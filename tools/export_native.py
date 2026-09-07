"""Prepare the non-block weights and tokenizer for the standalone native runtime.

Python and torch are only needed during export. Reuses build/weights for the W4A4
blocks; no source model is loaded by Python during native inference.
"""
import argparse
import json
from pathlib import Path
import shutil
import os
import tempfile

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent


def export(sources, destination, select, transform=lambda name, value: value):
    destination.mkdir(parents=True, exist_ok=True)
    index, offset = {}, 0
    with (destination / "weights.bin").open("wb") as output:
        for path in sources:
            with safe_open(path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    name = select(key)
                    if name is None:
                        continue
                    tensor = transform(name, source.get_tensor(key))
                    # The checkpoints' dtypes are kept: bf16 stays bf16 and the float32 tensors
                    # (the transformer's RMSNorm scales) stay float32, as ComfyUI and diffusers keep them.
                    f32 = tensor.dtype == torch.float32
                    tensor = (tensor.float() if f32 else tensor.to(torch.bfloat16)).contiguous()
                    data = (tensor.view(torch.uint32) if f32 else tensor.view(torch.uint16)).numpy().tobytes()
                    if name in index:
                        raise ValueError(f"duplicate tensor: {name}")
                    index[name] = dict(offset=offset, shape=list(tensor.shape), bytes=len(data), dtype="f32" if f32 else "bf16")
                    output.write(data)
                    offset += len(data)
    (destination / "weights.json").write_text(json.dumps(index, sort_keys=True))
    print(f"{destination.name}: {len(index)} tensors, {offset / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path.home() / "krea2-models")
    ap.add_argument("--blocks", type=Path, default=ROOT / "build/weights")
    ap.add_argument("--out", type=Path, default=ROOT / "build/native")
    ap.add_argument("--link-blocks", action="store_true", help="use development symlinks instead of copying block weights")
    ap.add_argument("--model", choices=("turbo", "raw"), default="turbo", help="which checkpoint the blocks came from: sets the bundle's schedule and guidance defaults")
    ap.add_argument("--checkpoint", type=Path, default=None, help="the bf16 ComfyUI-format checkpoint for the non-block transformer weights (default <models>/krea2_<model>_bf16.safetensors)")
    args = ap.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        ap.error(f"output already exists: {destination}; select a new --out directory")
    args.checkpoint = args.checkpoint or args.models / f"krea2_{args.model}_bf16.safetensors"
    required = [args.checkpoint, args.models / "qwen3-vl-4b/tokenizer.json",
                args.blocks / "weights.bin", args.blocks / "manifest.txt"]
    for pattern in ("qwen3-vl-4b/*.safetensors", "qwen-image/vae/*.safetensors"):
        if not list(args.models.glob(pattern)):
            ap.error(f"no model files match {args.models / pattern}")
    for path in required:
        if not path.is_file():
            ap.error(f"missing input: {path}")
    args.final_out = destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        args.out = Path(temporary) / "bundle"
        args.out.mkdir()
        export_bundle(args)
        args.out.rename(destination)
    print(f"native bundle: {destination}")


def export_bundle(args):
    export([args.checkpoint], args.out / "transformer",
           lambda k: k if not k.startswith("blocks.") or k.endswith(".mod.lin") else None)
    export(sorted((args.models / "qwen3-vl-4b").glob("*.safetensors")), args.out / "text",
           lambda k: k.removeprefix("model.language_model.") if k.startswith("model.language_model.") else None)
    def decoder_tensor(name, value):
        # A single image is the first causal frame: earlier temporal taps see
        # zero padding. Temporal upsamplers skip their time convolution here.
        return value[:, :, -1] if value.ndim == 5 else value
    export(sorted((args.models / "qwen-image/vae").glob("*.safetensors")), args.out / "vae",
           lambda k: k if k.startswith(("decoder.", "post_quant_conv.")) and ".time_conv." not in k else None,
           decoder_tensor)
    shutil.copyfile(args.models / "qwen3-vl-4b/tokenizer.json", args.out / "tokenizer.json")
    shutil.copytree(ROOT / "kernels", args.out / "sources", dirs_exist_ok=True)
    blocks = args.out / "blocks"
    blocks.mkdir(exist_ok=True)
    for name in ("weights.bin", "manifest.txt"):
        target = blocks / name
        if args.link_blocks:
            target.symlink_to(os.path.relpath((args.blocks / name).resolve(), args.final_out / "blocks"))
        else:
            shutil.copyfile(args.blocks / name, target)
    (args.out / "native.json").write_text(json.dumps(dict(version=1, model=f"krea2-{args.model}", max_text_tokens=512)))


if __name__ == "__main__":
    main()
