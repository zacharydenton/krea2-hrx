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
scripts/build.sh
env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/bench_vae.py \
  build/vae-opt/latents.npy --runs 4 --output build/vae.rgb
```

Use `--library` to point to an archived build, retaining its corresponding
`libkrea2.so` and runtime beside it. Compare the saved RGB buffers exactly.
`--width` and `--height` select rectangular images. The input is packed float32
latents in a `.npy` file. Run zero may compile kernels and is excluded from warm
statistics. `--profile --runs 2` enables synchronized per-kernel totals only
after the warmup, via `KREA2_NATIVE_KERNEL_PROFILE=1`.

## Implicit-GEMM convolution (2026-09-08)

The patch buffer is gone. `Ops::conv` builds a 3x3 convolution's operand inside
the GEMM by addressing the image, instead of writing an im2col matrix and
reading it back. At the decoder's largest stage that matrix is 113 MB per
convolution, written once and streamed once, for data the GEMM could work out
from the pixel and the tap.

Measured idle at 1024x1024 on fixed latents, same library, same seed:

| Path | Warm samples, seconds | Median |
| --- | --- | ---: |
| im2col + GEMM | 1.0658, 1.0654, 1.0679 | 1.0658 s |
| implicit convolution | 0.5927, 0.5856, 0.5815 | **0.5856 s** |

**1.82x, 45% less decode latency.** One convolution in isolation, against the
kernel the shape rules actually select on each side:

| Shape | im2col + GEMM | implicit | | Max error vs float64 |
| --- | ---: | ---: | ---: | --- |
| 256x256 x96 | 2.086 ms | 0.685 ms | 3.05x | 0.00753 both |
| 128x128 x192 | 1.213 ms | 0.591 ms | 2.05x | 0.01176 both |
| 64x64 x384 | 0.771 ms | 0.471 ms | 1.64x | 0.01558 both |

Accuracy is unchanged, not merely close: against a float64 oracle the two paths
have the same maximum error at every shape. They agree on 99.8-99.95% of
elements and differ by one bf16 ulp where they do, because the reduction runs
tap-major rather than channel-major. Against the diffusers VAE the decoder is
where it was: 1.2370, 0.9058 and 0.9328 per 255 at 64x64, 256x256 and 320x272,
against 1.2226, 0.9047 and 0.9324 before.

**Weights are packed `[out][ky][kx][in]`,** not the file's `[out][in][ky][kx]`.
That is the whole point of the change: with channels innermost, four consecutive
k are four consecutive channels of one tap, so the operand load stays the
contiguous four-element packet the GEMM already issues. With channels outermost
they would be four taps of one channel, strided by the channel count, and the
load would fall apart into four.

The layout travels on the `Weight`, and `Ops::conv` dispatches on it. It began
as a convention shared between the loader and the operation, and that cost two
bugs before it was written down as a type:

- Three of the decoder's convolutions (`resample.1`, after each upsample) are
  stored four-dimensional rather than five, so they have no temporal tap to
  reduce. The repack lived inside the staging step, which only runs for tensors
  that need staging for some other reason, so those three reached the device in
  the file's order and were read as though packed. The decode came out at
  72/255 against the reference. Nothing in the isolated kernel tests could see
  it: the kernel was correct at all ten shapes tried, and the fault was in which
  bytes reached it.
- A float32 convolution's bf16 copy was built from the unpacked bytes while the
  device received the packed ones, and any `Weight` a caller built by hand was
  reinterpreted according to a process-global setting rather than its contents.

`crates/ops/tests/arithmetic.rs` now builds the same convolution as a row-major
weight and a channels-last weight and requires them to agree; it fails by 1.04
on values up to 0.82 against the dispatch that ignored the layout.

`KREA2_CONV_IM2COL=1` keeps the file's order and the patch buffer, which is how
the two paths were compared here and how the `resample.1` bug was localised.

## Against ComfyUI, both measured 2026-09-08

Both sides on this box, idle, 1024x1024, Turbo, eight Euler steps, CFG 1,
seed 0, `a red fox in the snow`, prompt to RGB with text encoding and decode
inside the timer.

| | ComfyUI INT8 ConvRot | this runtime (W8A8) |
| --- | ---: | ---: |
| total, warm median | 36.31 s | **27.30 s** (1.33x) |
| denoise | 34.52 s | about 24 s (1.44x) |
| VAE decode | 0.652 s | **0.586 s** (1.11x) |

ComfyUI's samples were 36.83 and 35.80 s; ours 27.30, 27.13 and 28.06 s. This
replaces the 7 September comparison, whose 36.08 s figure it reproduces to
within 0.7% -- so that number was sound, but until now every ratio quoted here
was a fresh measurement of ours against a stale one of theirs.

The decode line is the one that moved. Before this change we were 1.038 s
against their 0.652 s, losing by 1.6x, and we still decode 1.72x the pixels they
do because tiles overlap by eight latents. The patch buffer was what forced
32x32 tiles; without it, larger tiles are affordable, and each step down in
overdraw is a further win in both time and seam error.
