# VAE decode optimization, 2026-09-07

An idle, alternating comparison of the native decoder at `a3b5040` and the
optimized build measures **1.294 s → 1.038 s** at 1024×1024: **1.246× throughput,
19.7% less decode latency**. All output RGB bytes are identical. These are
decoder timings, not complete generation timings.

| Build | Six warm samples, seconds | Median |
| --- | --- | ---: |
| Original | 1.276, 1.276, 1.279, 1.310, 1.312, 1.309 | 1.294 s |
| Coalesced im2col and fused wave normalization | 1.032, 1.032, 1.031, 1.045, 1.047, 1.053 | 1.038 s |

The [raw record](benchmarks/vae-2026-09-07.json) contains all samples, library
hashes and the fixed latent hash. Four resident sessions ran in A/B/B/A order,
each with one untimed-for-statistics warmup followed by three measured decodes.
Each session started after five consecutive idle GPU readings. Model loading
is outside the timer; latent upload, tile processing, blending and RGB output
are included. Profiling was disabled.

Earlier samples varied from 2.54 to 3.02 s with contention and are not used to
claim a speedup. The historical 3.2 s native / 0.66 s ComfyUI decoder comparison
is not a fresh baseline for this change. ComfyUI was not rerun here.

## Complete generation

With the original fp16 attention restored, 1024×1024 Turbo generation at eight
steps now measures **27.54 s warm median** (27.47, 27.54, 27.54 s). First generation
takes 29.29 s, with 1.28 s of model loading measured separately. Settings are
W8A8, prompt `a red fox in the snow`, seed 0, with text encoding and RGB output
included. Every repeat has exactly the same RGB hash as a complete generation
using the archived pre-optimization build.

This is about **1.31× faster than the recorded ComfyUI 36.08 s warm median** for
the same image size, step count and prompt. ComfyUI was measured earlier, so this
is not a freshly alternating backend comparison. The old build was run once here
to verify image equality; that first image is not a warm performance baseline.
See the [complete generation record](benchmarks/vae-full-2026-09-07.json).

## Changes

The decoder still uses the same 32×32 latent tiles, 24-pixel stride, overlap
blending, bf16 operations, weights and convolution accumulation order.

* **Convolution patches:** each workgroup reads a pixel's 3×3 neighborhood in
  contiguous channel order, transposes through shared memory, and writes the
  original `[channel, kernel_y, kernel_x]` patch order. This removes scattered
  global reads without changing any patch values. The path is selected for
  3×3 convolutions with channel counts divisible by 32 and at most 1024.
* **VAE normalization:** one wave handles a row, emulating the original 256
  virtual lanes and the same fp32 addition tree. Eight rows share a workgroup,
  eliminating workgroup barriers. The following SiLU is fused while preserving
  the intermediate bf16 rounding. The small attention normalization uses the
  same wave kernel without SiLU. Other normalization modes retain their kernels.

The synchronized diagnostic profile identified im2col as the largest cost,
followed by convolution GEMMs and normalization. Those per-kernel measurements
include synchronization overhead and must not be added to estimate warm decode
latency.

## Validation and reproduction

GPU tests require exact patch equality against an independently constructed
padding oracle, including rectangular inputs, single pixels and 32–1024
channels. Wave normalization and fused SiLU must match the original kernels
bit for bit at 3–1024 channels, with partial workgroups, zero rows and tiny
values. Full decoder comparisons require identical repeated RGB buffers on
real sampler latents and additional square/rectangular fixtures.
These checks pass at 64×64, 256×272, 512×512, 640×384 and 1024×1024, including
both the bf16 reference trajectory and the restored W8A8 trajectory at 1024×1024.
The complete `scripts/test.sh --quick` suite also passes, including generated
source parity, runtime checks, and all native, GEMM and attention kernel tests.

```sh
source scripts/env.sh
scripts/build_native.sh
env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/bench_vae.py \
  build/vae-opt/latents.npy --runs 4 --output build/vae.rgb
```

Use `--library` to point to an archived build, retaining its corresponding
`libkrea2.so` and runtime beside it. Compare the saved RGB buffers exactly.
`--width` and `--height` select rectangular images. The input is packed float32
latents in a `.npy` file. Run zero may compile kernels and is excluded from warm
statistics. `--profile --runs 2` enables synchronized per-kernel totals only
after the warmup, via `KREA2_NATIVE_KERNEL_PROFILE=1`.
