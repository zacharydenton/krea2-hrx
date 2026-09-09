#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy>=2",
#   "torch>=2.13,<2.15",
#   "triton-rocm",
#   "diffusers>=0.36",
#   "transformers>=4.57",
#   "safetensors>=0.6",
#   "pillow>=11",
# ]
#
# # Torch must be the ROCm build: PyPI's default wheels are CUDA and would give
# # this script a GPU it cannot use. This index matches the HIP 7.2 runtime on the
# # 8060S, and only publishes cp313 wheels, which is why requires-python is pinned
# # below 3.14 -- uv fetches a matching interpreter itself.
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm7.2"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# # Torch's ROCm build needs triton-rocm from this index; PyPI carries only an
# # unrelated 3.0.0rc1. It is named as a direct dependency because uv applies
# # tool.uv.sources to declared dependencies, not to ones reached through torch.
# triton-rocm = { index = "pytorch-rocm" }
# ///
"""Capture the unquantized BF16 ground truth that `scripts/parity.sh` gates against.

This is the half of the parity check that cannot live in the Rust test: it needs Torch,
Diffusers and the original unquantized `krea2_turbo_bf16.safetensors`, none of which this
repository depends on. The Rust gate only ever reads what this writes, and never creates or
updates its own ground truth -- which is exactly why the fixture has to be reproducible from
here rather than existing only as files somebody once made.

    ./scripts/capture_reference.py reference           the ground truth: noise, text, bf16 latent and image
    ./scripts/capture_reference.py accept --latents F  mint a new accepted W8A8 baseline (deliberate)
    ./scripts/capture_reference.py manifest            the hashes for crates/pipeline/tests/fixtures/unquantized.json

A `uv run` script: dependencies and the ROCm Torch index are in the header above and the
resolution is pinned by capture_reference.py.lock, so there is no environment to set up.

`reference` refuses to overwrite an existing fixture without --force: re-minting ground truth
from a build that has already drifted is the one mistake this gate cannot survive. A fresh
capture is a new fixture with new hashes, so `manifest` has to be run after it and the result
committed deliberately.

Weights live outside the repository (`~/krea2-models` by default, `--models` to move it):
`krea2_turbo_bf16.safetensors`, `qwen3-vl-4b/`, `qwen-image/vae/`.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Torch is imported by the stages that need it, not here: `manifest` is pure hashing and must
# stay runnable on a box that has neither Torch nor Diffusers installed.
ROOT = Path(__file__).resolve().parent.parent
MODELS = Path.home() / "krea2-models"

# diffusers' FlowMatchEulerDiscreteScheduler settings for Krea 2; `max_shift` is the
# exponential shift the sampler resolves to at this sequence length, and is written into
# job.json so the gate reads the schedule it was captured with instead of assuming one.
SCHEDULER = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=6400, max_shift=1.15,
                 num_train_timesteps=1000, shift=1.0, time_shift_type="exponential",
                 use_dynamic_shifting=True)
SHIFT = SCHEDULER["max_shift"]

# The files the Rust gate reads, in the order it reads them.
FIXTURE = ("job.json", "noise.npy", "text.npy", "bf16.npy", "bf16.png", "w8a8.npy", "w8a8.png")


def diffusers_state(comfy, layers: int = 28):
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

def cast_transformer_bf16(transformer):
    """Honor diffusers' fp32 parameter exceptions when loading with assign=True."""
    import torch

    keep = set(transformer._keep_in_fp32_modules)
    norms = {name: p.detach().float() for name, p in transformer.named_parameters()
             if keep.intersection(name.split(".")[:-1])}
    # ModelMixin.to warns because it cannot preserve these exceptions itself, so apply the
    # base cast and restore the saved fp32 tensors immediately afterward.
    torch.nn.Module.to(transformer, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, value in norms.items():
            transformer.get_parameter(name).data = value
    return transformer


def build(models: Path, checkpoint: Path, device: str = "cuda"):
    """The official Krea 2 modules with the unquantized ComfyUI-format weights mapped on."""
    import torch
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3VLModel

    # The checkpoint goes straight to the GPU (unified memory, one copy) and the module is
    # built on the meta device, so no f32 copy of 12.9B parameters is ever materialised.
    comfy = load_file(str(checkpoint), device=device)
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel()
    missing, unexpected = transformer.load_state_dict(diffusers_state(comfy), strict=False,
                                                      assign=True)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    transformer = cast_transformer_bf16(transformer)
    vae = AutoencoderKLQwenImage.from_pretrained(str(models / "qwen-image" / "vae"),
                                                 torch_dtype=torch.bfloat16).to(device)
    vae.enable_tiling()
    return Krea2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(**SCHEDULER), vae=vae,
        text_encoder=Qwen3VLModel.from_pretrained(str(models / "qwen3-vl-4b"),
                                                  torch_dtype=torch.bfloat16).to(device),
        tokenizer=AutoTokenizer.from_pretrained(str(models / "qwen3-vl-4b")),
        transformer=transformer, is_distilled=True)


def unpack(latents, height, width, vae_scale=8, p=2):
    """The C API's [1][tokens][64] back to the VAE's [1][16][1][H/8][W/8]."""
    b, _, c = latents.shape
    h, w = p * (height // (vae_scale * p)), p * (width // (vae_scale * p))
    x = latents.view(b, h // p, w // p, c // (p * p), p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c // (p * p), 1, h, w)


def decode(pipe, latents, size, path):
    """Tiled decode: a plain 1024^2 decode runs for many minutes on this stack."""
    import torch
    from PIL import Image
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).cuda().bfloat16()
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).cuda().bfloat16()
    z = unpack(latents.cuda().bfloat16(), size, size) / std + mean
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0]
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(rgb).save(path)
    print(f"decoded -> {path}")


def digests(work: Path) -> dict:
    return {name: hashlib.sha256((work / name).read_bytes()).hexdigest()
            for name in FIXTURE if (work / name).is_file()}


def command_reference(a) -> int:
    present = [name for name in FIXTURE if (a.work / name).is_file()]
    if present and not a.force:
        raise SystemExit(
            f"{a.work} already holds a fixture ({', '.join(present)}).\n"
            "Re-capturing ground truth from a build that has already drifted is the one\n"
            "mistake this gate cannot survive, so pass --force only when you mean to\n"
            "replace the reference and re-pin the manifest.")
    import torch

    checkpoint = a.checkpoint or a.models / "krea2_turbo_bf16.safetensors"
    if not Path(checkpoint).is_file():
        raise SystemExit(f"{checkpoint} does not exist: the unquantized checkpoint is the truth here")
    pipe = build(a.models, Path(checkpoint))
    with torch.no_grad():
        embeds, mask = pipe.encode_prompt(a.prompt, device="cuda")
        noise = pipe.prepare_latents(1, 16, a.size, a.size, torch.bfloat16, "cuda",
                                     torch.Generator("cuda").manual_seed(a.seed))
        a.work.mkdir(parents=True, exist_ok=True)
        # The native transformer takes the valid text rows only; the reference masks the rest.
        text = embeds[0][mask[0].bool()].float().cpu().numpy()
        np.save(a.work / "text.npy", np.ascontiguousarray(text, np.float32))
        np.save(a.work / "noise.npy", np.ascontiguousarray(noise[0].float().cpu().numpy(), np.float32))
        out = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, latents=noise,
                   height=a.size, width=a.size, num_inference_steps=a.steps,
                   guidance_scale=0.0, output_type="latent")
    latents = out.images.float().cpu()
    np.save(a.work / "bf16.npy", np.ascontiguousarray(latents.numpy(), np.float32))
    decode(pipe, latents, a.size, a.work / "bf16.png")
    (a.work / "job.json").write_text(json.dumps(
        dict(prompt=a.prompt, seed=a.seed, size=a.size, steps=a.steps,
             checkpoint=str(checkpoint), text_tokens=int(text.shape[0]), shift=SHIFT),
        indent=1) + "\n")
    print(f"reference: {text.shape[0]} text tokens, latents {tuple(latents.shape)} -> {a.work}")
    print("the accepted W8A8 baseline is not ground truth and is not written here; "
          "mint one with `accept`, then re-run `manifest`")
    return 0


def command_accept(a) -> int:
    """Promote a native run to the accepted baseline the gate must not regress against."""
    import torch

    if a.latents is None:
        raise SystemExit("--latents must name a KREA2_QUALITY_OUTPUT dump from the native run")
    meta = json.loads((a.work / "job.json").read_text())
    size, tokens = meta["size"], (meta["size"] // 16) ** 2
    raw = np.fromfile(a.latents, dtype="<f4")
    if raw.size != tokens * 64:
        raise SystemExit(f"{a.latents} holds {raw.size} floats, expected {tokens * 64}")
    latents = torch.from_numpy(raw.reshape(1, tokens, 64).copy())
    np.save(a.work / "w8a8.npy", np.ascontiguousarray(latents.numpy(), np.float32))
    pipe = build(a.models, Path(meta["checkpoint"]))
    decode(pipe, latents, size, a.work / "w8a8.png")
    truth = np.load(a.work / "bf16.npy").astype(np.float64).ravel()
    got = latents.numpy().astype(np.float64).ravel()
    cosine = float(got @ truth / (np.linalg.norm(got) * np.linalg.norm(truth)))
    print(f"accepted baseline: cosine {cosine:.6f}  "
          f"relative RMS {np.linalg.norm(got - truth) / np.linalg.norm(truth):.6f}")
    print("re-run `manifest` and commit the new hashes: the gate pins these files on purpose")
    return 0


def command_manifest(a) -> int:
    target = ROOT / "crates/pipeline/tests/fixtures/unquantized.json"
    found = digests(a.work)
    missing = [name for name in FIXTURE if name not in found]
    if missing:
        print(f"warning: {a.work} is missing {', '.join(missing)}")
    # The prose says what the fixture is, which no hash can. Carry the existing note forward
    # rather than dropping it, so re-pinning after a capture does not silently lose provenance.
    note = a.note
    if note is None and target.is_file():
        note = json.loads(target.read_text()).get("reference")
    manifest = {}
    if note:
        manifest["reference"] = note
    manifest["files"] = found
    text = json.dumps(manifest, indent=2) + "\n"
    if a.write:
        target.write_text(text)
        print(f"wrote {target}")
    else:
        print(text, end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["reference", "accept", "manifest"])
    ap.add_argument("--work", type=Path, default=ROOT / "build/quality")
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--checkpoint", default=None, help="the unquantized bf16 checkpoint")
    ap.add_argument("--latents", type=Path, default=None,
                    help="accept: a KREA2_QUALITY_OUTPUT dump of the native final latent")
    ap.add_argument("--prompt",
                    default="a red fox sitting in fresh snow at dawn, soft light, photograph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="reference: replace an existing fixture")
    ap.add_argument("--write", action="store_true", help="manifest: write the pinned file in place")
    ap.add_argument("--note", default=None,
                    help="manifest: replace the provenance line describing the fixture")
    a = ap.parse_args()
    return {"reference": command_reference, "accept": command_accept,
            "manifest": command_manifest}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
