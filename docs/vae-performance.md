# VAE decode

The decoder uses implicit GEMM for packed 3×3 convolutions: image patches are
loaded into the GEMM's shared-memory tiles instead of materialized in global
memory. A 256×256 layer with 96 input channels otherwise needs a 108 MiB patch
buffer. Weights are packed `[out, ky, kx, in]` and carry an explicit layout;
ordinary row-major weights retain the im2col path.

The reduction order changes from channel-major to tap-major. Outputs are not
bit-identical to im2col. Tiling remains 32×32 latents with stride 24 and bf16
overlap blending.

## Measurements, 2026-09-08

One idle Radeon 8060S, fixed 1024×1024 latents, resident models, warm kernels,
profiling disabled. Loading and compilation are excluded.

| Path | Warm samples, seconds | Median |
| --- | --- | ---: |
| im2col + GEMM | 1.0658, 1.0654, 1.0679 | 1.0658 s |
| Implicit GEMM | 0.5927, 0.5856, 0.5815 | 0.5856 s |

These samples show **1.82× throughput, or 45% less decode latency**. They are a
small local measurement, not a prediction for every resolution or power setting.

| Convolution (spatial size, input/output channels) | im2col + GEMM | Implicit GEMM | Speedup | Maximum error vs float64, both paths |
| --- | ---: | ---: | ---: | ---: |
| 256×256, 96 | 2.086 ms | 0.685 ms | 3.05× | 0.00753 |
| 128×128, 192 | 1.213 ms | 0.591 ms | 2.05× | 0.01176 |
| 64×64, 384 | 0.771 ms | 0.471 ms | 1.64× | 0.01558 |

The isolated tests reported 99.8–99.95% element agreement, with one-bf16-ulp
differences elsewhere. Equal maximum errors on these fixtures establish
comparable measured accuracy, not identical outputs or general quality parity.

Full decoder mean RGB errors against Diffusers, in 8-bit channel levels:

| Image size | Before | Implicit GEMM |
| --- | ---: | ---: |
| 64×64 | 1.2226 | 1.2370 |
| 256×256 | 0.9047 | 0.9058 |
| 320×272 | 0.9324 | 0.9328 |

The earlier coalesced-im2col and fused-normalization optimization measured
1.294 → 1.038 s with identical RGB output. Its
[raw record](benchmarks/vae-2026-09-07.json) and
[whole-image record](benchmarks/vae-full-2026-09-07.json) describe that earlier
build, not implicit GEMM. The September 8 samples above were recorded in the
implementation report at `0a8651f`; no separate raw benchmark file was committed.

## Reproduce

Use fixed packed float32 latents in a `.npy` file, matching the requested
resolution. The two paths must run in separate processes because packing is
selected when weights load.


> These commands are recorded as they were run. The Python benchmark
> tooling and `scripts/env.sh` were retired in e33b171; the measurements
> stand, but reproducing them means recovering those tools from Git.
```sh
source scripts/env.sh
scripts/build.sh
KREA2_CONV_IM2COL=1 .venv/bin/python tools/bench_vae.py /path/to/latents.npy \
  --width 1024 --height 1024 --runs 4 --output build/vae-im2col.rgb
KREA2_CONV_IM2COL=0 .venv/bin/python tools/bench_vae.py /path/to/latents.npy \
  --width 1024 --height 1024 --runs 4 --output build/vae-implicit.rgb
```

Run zero warms kernels and is excluded from warm statistics. Each process
requires identical RGB on repeated decodes. Across implementations, compare
numerical errors against the reference rather than requiring identical RGB.
Repeat in alternating order on an idle GPU and retain the JSON output, library
hashes and latent hash.

Complete decoder comparisons now come from `scripts/parity.sh`, which decodes
both trajectories through the native VAE; the Rust convolution test covers
dispatch for both weight layouts. Changes to packing
also need loader fixtures for BF16, F32, fp8 and causal convolution weights.
Larger tiles may reduce overlap work, but need separate memory and quality
validation before adoption.
