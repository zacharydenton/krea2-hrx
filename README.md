# krea2-loom

Krea 2 Turbo's transformer blocks in **Loom**, AMD's kernel language from
[ROCm/hrx-system](https://github.com/ROCm/hrx-system), for the Radeon 8060S (gfx1151),
in **W4A4 ConvRot**: int4 weights and int4 activations on the part's `iu4` WMMA, which is
the only 2x path this silicon has (measured 117 TOPS against 54 for int8 and fp16).
A sibling of [dinov3-loom](https://github.com/zacharydenton/dinov3-loom),
[scrfd-loom](https://github.com/zacharydenton/scrfd-loom) and the integer GEMM work in
`loom-gemm`.

## What runs where

The 28 blocks (98% of the FLOPs) run in Loom through a small C runtime, ten launches per
block. Everything around them stays in torch through diffusers' official `Krea2Pipeline`:
the Qwen3-VL-4B text encoder, the text-fusion stack, embeddings and the final layer, the
flow-match scheduler, the Qwen-Image VAE. `tools/pipeline.py --backend loom` is the
whole thing; `--quant w4a4` runs the same arithmetic in torch for comparison; the
default is the bf16 reference.

## Correctness

`reference/krea2_ref.py` is diffusers' model transcribed onto the ComfyUI checkpoint
names, validated bit-for-bit against diffusers at toy size. Every kernel is tested
against it: the three prepare kernels reproduce its int4 codes exactly, the attention
kernel matches torch's to cosine 0.99999996 at the real sequence length, and
`tests/test_blocks.py` runs the native blocks on a fixture captured from a real
denoising step against the reference in both W4A4 and bf16.

## Status

Building. See `docs/notes.md` for decisions and results as they land.
