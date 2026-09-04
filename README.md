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

## Results so far (1024x1024, 8 Turbo steps, seed 0)

| | per forward of the 28 blocks | latent PSNR vs bf16 | image PSNR vs bf16 |
| --- | ---: | ---: | ---: |
| torch bf16 (diffusers) | ~7 s | | |
| reference W4A4 in torch | 87 s | 9.96 dB | 18.9 dB |
| Loom W4A4, first attention | 7.85 s | 10.09 dB | 19.0 dB |
| Loom W4A4, staged attention | 2.89 s (29.2 s per image) | 10.09 dB | 19.04 dB |
| Loom W4A4, Q resident + V transposed in LDS, raster groups | 2.31 s (25.4 s per image) | 12.92 dB | 23.23 dB |
| Loom W4A4, prepare kernels vectorised | 2.13 s (21.9 s per image, steady state) | | |

The per-image time is the second image of a process (`tools/pipeline.py --images 2`):
8 steps at 2.35 s plus text encoding and the tiled VAE decode. The first image pays the
Loom session build (about 6 s) and, once per machine, MIOpen's convolution kernel search
for the VAE decode (minutes; cached afterwards).

The three pictures (`build/bf16_seed0.png`, `build/w4a4_seed0.png`,
`build/loom_seed0.png`) are the same fox in the same pose and light; the deviation is
fur and snow detail, not artifacts. For comparison, a published int4 ConvRot checkpoint
of the same model that keeps 96 of its 224 block linears in int8 reports a minimum
image PSNR of 17.7 dB against bf16; this port quantises every block GEMM to int4 with
per-row scales and measures 19.0.

Where a forward goes now: the four GEMMs 68% (at 56-75 TOPS on the part's measured
117 TOPS int4 ceiling), attention 26% (20 TFLOP/s), the prepare kernels 5%.

The PSNR between two 8-step trajectories swings by several dB between numerically
near-identical runs (the sampler amplifies rounding); the per-block cosine against the
reference in `tests/test_blocks.py` is the metric that tracks kernel correctness.

See `docs/notes.md` for every decision and measurement.
