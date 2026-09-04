"""The Krea 2 Turbo pipeline from the files on this box, as the quality oracle.

    python3 tools/pipeline.py --prompt "..." --seed 0 --out build/bf16.png
    python3 tools/pipeline.py --quant w4a4 --out build/w4a4.png      # same seed -> compare
    python3 tools/pipeline.py --fixture build/step0.pt              # dump one step's block I/O

diffusers' official Krea2Pipeline runs the text encoder, scheduler and VAE; the
transformer is the official module with the ComfyUI-format Turbo weights mapped
onto it, and with --quant its forward is routed through reference/krea2_ref.py so
the W4A4 ConvRot arithmetic can be judged on real images before any kernel exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import krea2_ref as R

MODELS = Path.home() / "krea2-models"
TURBO = MODELS / "krea2_turbo_bf16.safetensors"
QWEN = MODELS / "qwen3-vl-4b"
VAE = MODELS / "qwen-image" / "vae"
SCHEDULER = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=6400, max_shift=1.15,
                 num_train_timesteps=1000, shift=1.0, time_shift_type="exponential", use_dynamic_shifting=True)


def diffusers_state(comfy: dict[str, torch.Tensor], layers: int = 28) -> dict[str, torch.Tensor]:
    """ComfyUI checkpoint names -> diffusers parameter names (views, no copies)."""
    out = {}
    def attn(dst, src):
        out[f"{dst}.to_q.weight"] = comfy[f"{src}.wq.weight"]; out[f"{dst}.to_k.weight"] = comfy[f"{src}.wk.weight"]
        out[f"{dst}.to_v.weight"] = comfy[f"{src}.wv.weight"]; out[f"{dst}.to_gate.weight"] = comfy[f"{src}.gate.weight"]
        out[f"{dst}.to_out.0.weight"] = comfy[f"{src}.wo.weight"]
        out[f"{dst}.norm_q.weight"] = comfy[f"{src}.qknorm.qnorm.scale"]; out[f"{dst}.norm_k.weight"] = comfy[f"{src}.qknorm.knorm.scale"]
    def mlp(dst, src):
        for n in ("gate", "up", "down"):
            out[f"{dst}.{n}.weight"] = comfy[f"{src}.{n}.weight"]
    for i in range(layers):
        s, d = f"blocks.{i}", f"transformer_blocks.{i}"
        attn(f"{d}.attn", f"{s}.attn"); mlp(f"{d}.ff", f"{s}.mlp")
        out[f"{d}.norm1.weight"] = comfy[f"{s}.prenorm.scale"]; out[f"{d}.norm2.weight"] = comfy[f"{s}.postnorm.scale"]
        out[f"{d}.scale_shift_table"] = comfy[f"{s}.mod.lin"].view(6, -1)
    for kind, n in (("layerwise", 2), ("refiner", 2)):
        for i in range(n):
            s, d = f"txtfusion.{kind}_blocks.{i}", f"text_fusion.{kind}_blocks.{i}"
            attn(f"{d}.attn", f"{s}.attn"); mlp(f"{d}.ff", f"{s}.mlp")
            out[f"{d}.norm1.weight"] = comfy[f"{s}.prenorm.scale"]; out[f"{d}.norm2.weight"] = comfy[f"{s}.postnorm.scale"]
    out["text_fusion.projector.weight"] = comfy["txtfusion.projector.weight"]
    out["txt_in.norm.weight"] = comfy["txtmlp.0.scale"]
    out["txt_in.linear_1.weight"], out["txt_in.linear_1.bias"] = comfy["txtmlp.1.weight"], comfy["txtmlp.1.bias"]
    out["txt_in.linear_2.weight"], out["txt_in.linear_2.bias"] = comfy["txtmlp.3.weight"], comfy["txtmlp.3.bias"]
    out["img_in.weight"], out["img_in.bias"] = comfy["first.weight"], comfy["first.bias"]
    out["time_embed.linear_1.weight"], out["time_embed.linear_1.bias"] = comfy["tmlp.0.weight"], comfy["tmlp.0.bias"]
    out["time_embed.linear_2.weight"], out["time_embed.linear_2.bias"] = comfy["tmlp.2.weight"], comfy["tmlp.2.bias"]
    out["time_mod_proj.weight"], out["time_mod_proj.bias"] = comfy["tproj.1.weight"], comfy["tproj.1.bias"]
    out["final_layer.scale_shift_table"] = comfy["last.modulation.lin"].view(2, -1)
    out["final_layer.norm.weight"] = comfy["last.norm.scale"]
    out["final_layer.linear.weight"], out["final_layer.linear.bias"] = comfy["last.linear.weight"], comfy["last.linear.bias"]
    return out


class ReferenceForward:
    """Routes Krea2Transformer2DModel.forward through krea2_ref (compacting the text
    tokens by the padding mask, which is exact), optionally dumping one step's block
    inputs and outputs as the fixture for the Loom runtime."""
    def __init__(self, ref: R.Krea2Ref, fixture: Path | None):
        self.ref, self.fixture, self.calls = ref, fixture, 0

    def __call__(self, hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None,
                 attention_kwargs=None, return_dict=True):
        text = encoder_hidden_states
        if encoder_attention_mask is not None:
            keep = encoder_attention_mask[0].bool()
            text = text[:, keep]
        grid_h = int(position_ids[:, 1].max().item()) + 1
        grid_w = int(position_ids[:, 2].max().item()) + 1
        ref = self.ref
        with torch.no_grad():
            txt = ref.text_in(text); img = ref.image_in(hidden_states)
            x = torch.cat([txt, img], dim=1)
            temb, mod = ref.time_embed(timestep)
            cos, sin = R.rope_tables(R.position_ids(txt.shape[1], grid_h, grid_w, x.device))
            y = ref.blocks_forward(x, mod, cos, sin)
            out = ref.final(y[:, txt.shape[1]:], temb)
        if self.fixture is not None and self.calls == 0:
            mods = torch.stack([ref.block_modulation(i, mod)[0, 0] for i in range(ref.layers)])   # [28, 6, 6144]
            torch.save(dict(x=x[0].cpu(), mods=mods.cpu(), cos=cos.cpu(), sin=sin.cpu(), y=y[0].cpu(),
                            text_len=txt.shape[1], grid=(grid_h, grid_w), timestep=timestep.cpu()), self.fixture)
            print(f"fixture written: {self.fixture} (tokens {x.shape[1]}, text {txt.shape[1]})")
        self.calls += 1
        return (out,) if not return_dict else type("O", (), {"sample": out})()


def build(quant: str, fixture: Path | None, device="cuda"):
    from diffusers import Krea2Pipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
    from transformers import AutoTokenizer, Qwen3VLModel
    t0 = time.time()
    comfy = R.load_weights(str(TURBO))
    transformer = Krea2Transformer2DModel()
    missing, unexpected = transformer.load_state_dict(diffusers_state(comfy), strict=False)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    if quant != "none" or fixture is not None:
        ref = R.Krea2Ref(comfy, quant=quant, device=device, dtype=torch.bfloat16)
        transformer.forward = ReferenceForward(ref, fixture)
    text_encoder = Qwen3VLModel.from_pretrained(str(QWEN), torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN))
    vae = AutoencoderKLQwenImage.from_pretrained(str(VAE), torch_dtype=torch.bfloat16).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
    pipe = Krea2Pipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                         transformer=transformer, is_distilled=True)
    print(f"pipeline built in {time.time() - t0:.0f} s (quant={quant})")
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox sitting in fresh snow at dawn, soft light, photograph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--quant", default="none", choices=["none", "w4a4"])
    ap.add_argument("--out", default="build/out.png")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--latents-out", default=None, help="save the final packed latents for PSNR comparisons")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pipe = build(a.quant, Path(a.fixture) if a.fixture else None)
    gen = torch.Generator("cuda").manual_seed(a.seed)
    t0 = time.time()
    result = pipe(a.prompt, height=a.size, width=a.size, num_inference_steps=a.steps, guidance_scale=0.0,
                  generator=gen, output_type="latent" if a.latents_out else "pil")
    torch.cuda.synchronize()
    dt = time.time() - t0
    if a.latents_out:
        torch.save(result.images.cpu(), a.latents_out)
        print(f"latents saved: {a.latents_out}")
        image = pipe.vae  # decode separately when needed
    else:
        result.images[0].save(a.out)
        print(f"saved {a.out}")
    print(f"{a.steps} steps at {a.size}^2: {dt:.1f} s ({dt / a.steps:.2f} s/step incl. text encode and decode)")


if __name__ == "__main__":
    main()
