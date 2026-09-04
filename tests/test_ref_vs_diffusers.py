"""Structural check of reference/krea2_ref.py against diffusers' Krea2Transformer2DModel
at toy size with random weights: the two must agree to bf16 rounding on the full
forward, which validates the transcription (key mapping, modulation, RoPE, GQA,
gate, text fusion, final layer) without any 26 GB download."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import krea2_ref as R
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel


def comfy_state(model: Krea2Transformer2DModel) -> dict[str, torch.Tensor]:
    """diffusers parameter names -> ComfyUI checkpoint names."""
    sd = model.state_dict()
    out = {}
    def attn(dst, src):
        out[f"{dst}.wq.weight"] = sd[f"{src}.to_q.weight"]; out[f"{dst}.wk.weight"] = sd[f"{src}.to_k.weight"]
        out[f"{dst}.wv.weight"] = sd[f"{src}.to_v.weight"]; out[f"{dst}.gate.weight"] = sd[f"{src}.to_gate.weight"]
        out[f"{dst}.wo.weight"] = sd[f"{src}.to_out.0.weight"]
        out[f"{dst}.qknorm.qnorm.scale"] = sd[f"{src}.norm_q.weight"]; out[f"{dst}.qknorm.knorm.scale"] = sd[f"{src}.norm_k.weight"]
    def mlp(dst, src):
        for n in ("gate", "up", "down"):
            out[f"{dst}.{n}.weight"] = sd[f"{src}.{n}.weight"]
    for i in range(model.config.num_layers):
        s, d = f"transformer_blocks.{i}", f"blocks.{i}"
        attn(f"{d}.attn", f"{s}.attn"); mlp(f"{d}.mlp", f"{s}.ff")
        out[f"{d}.prenorm.scale"] = sd[f"{s}.norm1.weight"]; out[f"{d}.postnorm.scale"] = sd[f"{s}.norm2.weight"]
        out[f"{d}.mod.lin"] = sd[f"{s}.scale_shift_table"].reshape(-1)
    for kind, n in (("layerwise", model.config.num_layerwise_text_blocks), ("refiner", model.config.num_refiner_text_blocks)):
        for i in range(n):
            s, d = f"text_fusion.{kind}_blocks.{i}", f"txtfusion.{kind}_blocks.{i}"
            attn(f"{d}.attn", f"{s}.attn"); mlp(f"{d}.mlp", f"{s}.ff")
            out[f"{d}.prenorm.scale"] = sd[f"{s}.norm1.weight"]; out[f"{d}.postnorm.scale"] = sd[f"{s}.norm2.weight"]
    out["txtfusion.projector.weight"] = sd["text_fusion.projector.weight"]
    out["txtmlp.0.scale"] = sd["txt_in.norm.weight"]
    out["txtmlp.1.weight"], out["txtmlp.1.bias"] = sd["txt_in.linear_1.weight"], sd["txt_in.linear_1.bias"]
    out["txtmlp.3.weight"], out["txtmlp.3.bias"] = sd["txt_in.linear_2.weight"], sd["txt_in.linear_2.bias"]
    out["first.weight"], out["first.bias"] = sd["img_in.weight"], sd["img_in.bias"]
    out["tmlp.0.weight"], out["tmlp.0.bias"] = sd["time_embed.linear_1.weight"], sd["time_embed.linear_1.bias"]
    out["tmlp.2.weight"], out["tmlp.2.bias"] = sd["time_embed.linear_2.weight"], sd["time_embed.linear_2.bias"]
    out["tproj.1.weight"], out["tproj.1.bias"] = sd["time_mod_proj.weight"], sd["time_mod_proj.bias"]
    out["last.modulation.lin"] = sd["final_layer.scale_shift_table"].reshape(-1)
    out["last.norm.scale"] = sd["final_layer.norm.weight"]
    out["last.linear.weight"], out["last.linear.bias"] = sd["final_layer.linear.weight"], sd["final_layer.linear.bias"]
    return out


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    heads, kv, hd = 4, 2, 32                     # toy: hidden 128
    R.HEADS, R.KV_HEADS, R.HEAD_DIM, R.HIDDEN = heads, kv, hd, heads * hd
    R.AXES = (8, 12, 12)
    R.TXT_DIM, R.TXT_HEADS, R.TXT_LAYERS = 64, 2, 3
    model = Krea2Transformer2DModel(in_channels=16, num_layers=2, attention_head_dim=hd, num_attention_heads=heads,
                                    num_key_value_heads=kv, intermediate_size=192, timestep_embed_dim=32,
                                    text_hidden_dim=64, num_text_layers=3, text_num_attention_heads=2,
                                    text_num_key_value_heads=2, text_intermediate_size=96, axes_dims_rope=(8, 12, 12),
                                    rope_theta=1000.0).to(dev)
    with torch.no_grad():
        for p in model.parameters():          # zero-init tables get real values
            p.copy_(torch.randn_like(p) * 0.2)
    ref = R.Krea2Ref(comfy_state(model), quant="none", device=dev, dtype=torch.float32, layers=2)
    # the toy RoPE tables and the timestep width differ from the real model's constants
    R.timestep_embedding.__defaults__ = (32,)
    b, gh, gw, t_len = 1, 4, 6, 5
    latents = torch.randn(b, gh * gw, 16, device=dev)
    text = torch.randn(b, t_len, 3, 64, device=dev)
    timestep = torch.tensor([0.7], device=dev)
    pos = R.position_ids(t_len, gh, gw, dev)
    with torch.no_grad():
        want = model(hidden_states=latents, encoder_hidden_states=text, timestep=timestep, position_ids=pos, return_dict=False)[0]
        got = ref.forward(latents, text, timestep, gh, gw)
    err = (got.float() - want.float()).abs().max().item()
    rel = err / want.float().abs().max().item()
    print(f"toy forward: max_abs={err:.3e} rel={rel:.3e}")
    ok = rel < 1e-4
    print("  PASS" if ok else "  FAIL", "reference matches diffusers at toy size")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
