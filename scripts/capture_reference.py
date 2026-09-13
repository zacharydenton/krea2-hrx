#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy>=2",
#   "torch>=2.13,<2.15",
#   "triton-rocm",
#   "diffusers>=0.36",
#   # The official transformer keeps some modules in fp32, and diffusers refuses to load
#   # that without accelerate: from_pretrained fails outright, it is not merely slower.
#   "accelerate>=1.0",
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
    ./scripts/capture_reference.py manifest            the hashes for tests/fixtures/unquantized.json

A `uv run` script: dependencies and the ROCm Torch index are in the header above and the
resolution is pinned by capture_reference.py.lock, so there is no environment to set up.

`reference` refuses to overwrite an existing fixture without --force: re-minting ground truth
from a build that has already drifted is the one mistake this gate cannot survive. A fresh
capture is a new fixture with new hashes, so `manifest` has to be run after it and the result
committed deliberately.

Model components use the standard Hugging Face cache. The optional --checkpoint
fallback replaces only the transformer; other components come from the pinned
official repository.
"""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Torch is imported by the stages that need it, not here: `manifest` is pure hashing and must
# stay runnable on a box that has neither Torch nor Diffusers installed.
ROOT = Path(__file__).resolve().parent.parent
REPO = "krea/Krea-2-Turbo"
# Immutable official snapshot present in the capture machine's Hugging Face cache.
# This pins future captures; it does not retroactively attribute older fixtures.
REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"

# diffusers' FlowMatchEulerDiscreteScheduler settings for Krea 2; `max_shift` is the
# exponential shift the sampler resolves to at this sequence length, and is written into
# job.json so the gate reads the schedule it was captured with instead of assuming one.
SCHEDULER = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=6400, max_shift=1.15,
                 num_train_timesteps=1000, shift=1.0, time_shift_type="exponential",
                 use_dynamic_shifting=True)
SHIFT = SCHEDULER["max_shift"]

# The files the Rust gate reads, in the order it reads them.
FIXTURE = ("job.json", "noise.npy", "text.npy", "bf16.npy", "bf16.png", "w8a8.npy", "w8a8.png")
REFERENCE = FIXTURE[:5]
BASELINE = FIXTURE[5:]


def model_revision(a):
    revision = a.revision or (REVISION if a.repo == REPO else None)
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit("--revision must be a full, immutable model commit SHA; "
                         "custom repositories require an explicit revision")
    return revision


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

def cast_transformer(transformer, dtype):
    """Honor diffusers' fp32 parameter exceptions when loading with assign=True."""
    import torch

    keep = set(transformer._keep_in_fp32_modules)
    norms = {name: p.detach().float() for name, p in transformer.named_parameters()
             if keep.intersection(name.split(".")[:-1])}
    # ModelMixin.to warns because it cannot preserve these exceptions itself, so apply the
    # base cast and restore the saved fp32 tensors immediately afterward.
    torch.nn.Module.to(transformer, dtype=dtype)
    with torch.no_grad():
        for name, value in norms.items():
            transformer.get_parameter(name).data = value
    return transformer


def build(a, dtype):
    """The pipeline the reference is defined by.

    By default this is the official diffusers repository loaded with from_pretrained, so the
    reference is the stock implementation on the stock weights and nothing here interprets
    the checkpoint. `--checkpoint` is a labelled fallback that maps a local ComfyUI-format
    file onto the same official modules; it is recorded in job.json as such, because a
    reference produced that way rests on this file's name mapping as well as on diffusers.
    """
    import torch
    from diffusers import Krea2Pipeline

    if a.checkpoint is None:
        revision = model_revision(a)
        pipe = Krea2Pipeline.from_pretrained(a.repo, revision=revision, dtype=dtype)
        return pipe.to(a.device), {"repo": a.repo, "revision": revision}

    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
    from safetensors.torch import load_file

    comfy = load_file(str(a.checkpoint), device=a.device)
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel()
    mapped = diffusers_state(comfy)
    # load_state_dict(assign=True) replaces parameters outright and so does not check
    # shapes: a mis-mapped name would be adopted silently. This mapping is the one part
    # that is not stock diffusers, so it is checked here.
    declared = dict(transformer.named_parameters())
    for name, tensor in mapped.items():
        want = declared.get(name)
        assert want is not None, f"{name} is not a parameter of the official module"
        assert tuple(want.shape) == tuple(tensor.shape), \
            f"{name}: checkpoint {tuple(tensor.shape)} but the module declares {tuple(want.shape)}"
    missing, unexpected = transformer.load_state_dict(mapped, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    transformer = cast_transformer(transformer, dtype)
    revision = model_revision(a)
    pipe = Krea2Pipeline.from_pretrained(
        a.repo, revision=revision, dtype=dtype, transformer=transformer)
    pipe.vae.enable_tiling()
    return pipe.to(a.device), {"checkpoint": str(a.checkpoint),
                              "loaded_by": "local ComfyUI name mapping",
                              "repo": a.repo, "revision": revision}


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
    where, dtype = vae.device, vae.dtype
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean).view(shape).to(where, dtype)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(shape).to(where, dtype)
    z = unpack(latents.to(where, dtype), size, size) / std + mean
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0]
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(rgb).save(path)
    print(f"decoded -> {path}")


def digests(work: Path) -> dict:
    return {name: hashlib.sha256((work / name).read_bytes()).hexdigest()
            for name in FIXTURE if (work / name).is_file()}


def publish(stage, work, names):
    # All expensive or fallible generation and validation happens before publication.
    # Replace individual files so unrelated archives in the fixture directory survive.
    for name in names:
        if not (stage / name).is_file():
            raise SystemExit(f"capture did not produce {name}")
    work.mkdir(parents=True, exist_ok=True)
    if names == REFERENCE:
        for name in BASELINE:
            (work / name).unlink(missing_ok=True)
    for name in names:
        (stage / name).replace(work / name)


def command_reference(a) -> int:
    present = [name for name in FIXTURE if (a.work / name).is_file()]
    if present and not a.force:
        raise SystemExit(
            f"{a.work} already holds a fixture ({', '.join(present)}).\n"
            "Re-capturing ground truth from a build that has already drifted is the one\n"
            "mistake this gate cannot survive, so pass --force only when you mean to\n"
            "replace the reference and re-pin the manifest.")
    a.work.parent.mkdir(parents=True, exist_ok=True)
    # Failed model loads, inference or decoding must leave the old fixture intact.
    # Staging also permits --reuse to point at the fixture being replaced.
    with TemporaryDirectory(prefix=".reference-", dir=a.work.parent) as temporary:
        staged = argparse.Namespace(**vars(a))
        staged.work = Path(temporary)
        capture_reference(staged)
        publish(staged.work, a.work, REFERENCE)
    print(f"reference -> {a.work}; old accepted baseline removed")
    print("pin with `manifest --reference-only --write`, run "
          "`KREA2_QUALITY_MINT=1 scripts/parity.sh`, then `manifest --write`")
    return 0


def capture_reference(a):
    import torch

    if a.checkpoint is not None and not Path(a.checkpoint).is_file():
        raise SystemExit(f"{a.checkpoint} does not exist")
    dtype = getattr(torch, a.dtype)
    pipe, source = build(a, dtype)
    with torch.no_grad():
        if a.reuse is not None:
            # Take another fixture's exact conditioning and noise, so that a capture on a
            # different device isolates the transformer and nothing else. Encoding the prompt
            # again would also move the text encoder, which is a second difference.
            prior = np.load(Path(a.reuse) / "text.npy")
            embeds = torch.from_numpy(prior).to(a.device, dtype)[None]
            mask = torch.ones(1, prior.shape[0], dtype=torch.bool, device=a.device)
            source["conditioning_from"] = str(a.reuse)
        else:
            embeds, mask = pipe.encode_prompt(a.prompt, device=a.device)
        # The noise is always drawn on the CPU, whatever the model runs on. A CUDA
        # generator and a CPU generator produce entirely different streams from the same
        # seed -- measured here at cosine 0.002, i.e. unrelated -- so tying the noise to
        # the compute device would make a CPU capture and a GPU capture incomparable
        # rather than merely differently rounded.
        if a.reuse is not None:
            prior = np.load(Path(a.reuse) / "noise.npy")
            noise = torch.from_numpy(prior).to(a.device, dtype)[None]
        else:
            noise = pipe.prepare_latents(1, 16, a.size, a.size, dtype, "cpu",
                                         torch.Generator("cpu").manual_seed(a.seed))
            noise = noise.to(a.device)
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
    # Why the CPU is the default: not GPU nondeterminism -- two GPU captures back to back
    # on one stack are bit-identical, as are two CPU captures. What moved was the stack.
    # Recapturing the archived fixture on today's Torch/Diffusers/ROCm shifted the latent
    # by relative RMS 0.019, about 80% of the W8A8 error the gate exists to measure, and
    # the archived job.json recorded no versions to attribute that to. A CPU reference
    # depends on far less -- no ROCm, no GPU, no vendor kernels -- so it can be reproduced
    # on a machine that has none of this hardware. The stack is recorded either way.
    import diffusers
    (a.work / "job.json").write_text(json.dumps(
        dict(prompt=a.prompt, seed=a.seed, size=a.size, steps=a.steps,
             text_tokens=int(text.shape[0]), shift=SHIFT, device=a.device, dtype=a.dtype,
             torch=torch.__version__, diffusers=diffusers.__version__, **source),
        indent=1) + "\n")
    print(f"reference: {text.shape[0]} text tokens, latents {tuple(latents.shape)} -> {a.work}")


def command_accept(a) -> int:
    """Promote a native run to the accepted baseline the gate must not regress against."""
    from PIL import Image

    if a.latents is None:
        raise SystemExit("--latents must name a KREA2_QUALITY_OUTPUT dump from the native run")
    meta = json.loads((a.work / "job.json").read_text())
    size, tokens = meta["size"], (meta["size"] // 16) ** 2
    # The native runner writes the receipt last, after exporting both final latents and
    # its own decoded RGB. Hashes bind those outputs to each other and to the reference.
    receipt_path = Path(str(a.latents) + ".json")
    if not receipt_path.is_file():
        raise SystemExit("missing native export receipt; re-run with KREA2_QUALITY_OUTPUT")
    receipt = json.loads(receipt_path.read_text())
    reference = {name: hashlib.sha256((a.work / name).read_bytes()).hexdigest()
                 for name in REFERENCE}
    if (receipt.get("version") != 1 or receipt.get("size") != size
            or receipt.get("reference") != reference):
        raise SystemExit("native export was not produced against this reference")
    raw_bytes = a.latents.read_bytes()
    image_bytes = Path(str(a.latents) + ".png").read_bytes()
    if (hashlib.sha256(raw_bytes).hexdigest() != receipt.get("latents_sha256")
            or hashlib.sha256(image_bytes).hexdigest() != receipt.get("image_sha256")):
        raise SystemExit("native export hashes do not match; latents/image pair changed")
    if len(raw_bytes) != tokens * 64 * 4:
        raise SystemExit(f"{a.latents} holds {len(raw_bytes)} bytes, expected {tokens * 64 * 4}")
    raw = np.frombuffer(raw_bytes, dtype="<f4")
    if not np.isfinite(raw).all():
        raise SystemExit(f"{a.latents} contains non-finite values")
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (size, size):
            raise SystemExit("native export must contain an RGB8 PNG at the fixture resolution")
        image.load()
    latents = raw.reshape(1, tokens, 64)
    truth = np.load(a.work / "bf16.npy").astype(np.float64).ravel()
    got = latents.astype(np.float64).ravel()
    cosine = float(got @ truth / (np.linalg.norm(got) * np.linalg.norm(truth)))
    with TemporaryDirectory(prefix=".accept-", dir=a.work.parent) as temporary:
        stage = Path(temporary)
        np.save(stage / "w8a8.npy", latents)
        (stage / "w8a8.png").write_bytes(image_bytes)
        publish(stage, a.work, BASELINE)
    print(f"accepted baseline: cosine {cosine:.6f}  "
          f"relative RMS {np.linalg.norm(got - truth) / np.linalg.norm(truth):.6f}")
    print("re-run `manifest` and commit the new hashes: the gate pins these files on purpose")
    return 0


def command_manifest(a) -> int:
    target = a.manifest
    found = digests(a.work)
    required = REFERENCE if a.reference_only else FIXTURE
    missing = [name for name in required if name not in found]
    if missing:
        raise SystemExit(f"{a.work} is missing {', '.join(missing)}")
    if a.reference_only and any(name in found for name in BASELINE):
        raise SystemExit("reference-only pinning requires both accepted baseline files absent")
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
    ap.add_argument("--repo", default=REPO,
                    help="the official diffusers repository the reference is defined by")
    ap.add_argument("--revision", default=None,
                    help=f"immutable model commit SHA (official repo default: {REVISION})")
    ap.add_argument("--checkpoint", default=None,
                    help="fallback: a local ComfyUI-format bf16 checkpoint, mapped onto the "
                         "official modules by this file (recorded in job.json as such)")
    ap.add_argument("--latents", type=Path, default=None,
                    help="accept: a KREA2_QUALITY_OUTPUT dump with its .png and .json sidecars")
    ap.add_argument("--prompt",
                    default="a red fox sitting in fresh snow at dawn, soft light, photograph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="reference: where to compute the truth (default cpu, so the "
                         "reference can be reproduced without this vendor's GPU stack)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--reuse", type=Path, default=None,
                    help="reference: take noise and conditioning from another fixture, so a "
                         "capture on a different device isolates the transformer alone")
    ap.add_argument("--force", action="store_true", help="reference: replace an existing fixture")
    ap.add_argument("--write", action="store_true", help="manifest: write the pinned file in place")
    ap.add_argument("--reference-only", action="store_true",
                    help="manifest: pin only the reference before minting a fresh baseline")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "tests/fixtures/unquantized.json",
                    help="manifest: destination (Rust override: KREA2_QUALITY_MANIFEST)")
    ap.add_argument("--note", default=None,
                    help="manifest: replace the provenance line describing the fixture")
    a = ap.parse_args()
    return {"reference": command_reference, "accept": command_accept,
            "manifest": command_manifest}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
