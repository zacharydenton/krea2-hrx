# Native SA2-style attention on gfx1151

The Radeon 8060S runtime always uses tuned smoothed INT4 QK attention. It selects
eight waves below 8,192 tokens and four waves with explicit prefetch at longer
sequences. There are no attention selectors or fallback backends. Inference uses
the C ABI with no Python or Torch dependency.

## Arithmetic and hardware adaptation

[SageAttention2](https://arxiv.org/html/2411.10958v4) combines smoothed INT4 QK,
quantization groups matched to CUDA MMA threads, and FP8 PV. This implementation
adapts its Q/K smoothing and INT4 QK approach to gfx11 WMMA. It uses per-token
scales and fp16 PV; it is not an exact port of upstream SageAttention2.

`host/sage.cpp` prepares Q/K on the GPU using HIP and hipBLAS:

1. Compute the K mean over the actual sequence, independently per head/channel.
2. Center Q within 64-token groups, excluding padded rows from the mean.
3. Quantize centered Q and K symmetrically to INT4 codes in [-7, 7], with one
   absmax/7 scale per token/head and round-to-nearest-even packing.
4. Compute Q-mean times centered-K with fp16 inputs and fp32 accumulation/output.
   Four Q heads share one K head; their query groups become adjacent GEMM columns,
   so the centered K workspace is stored once per KV head.

The Loom kernel evaluates the centered QK product with INT4 WMMA, dequantizes it,
restores the Q-mean correction, and applies the usual 1/sqrt(128) scale. Omitting
the K-mean contribution changes each score row by a constant, which softmax
cancels. The correction's fp16 input rounding is an additional approximation.
Online softmax and accumulators remain fp32; probability fragments and V use fp16.

The original kernels (historical measurements below) processed 16 queries by 16 keys with four wave32 groups,
one wave per Q head sharing a K/V head. All eight packed Q fragments remain in
registers. There are eight INT4 QK WMMA operations per key tile and eight fp16 PV
operations. The native preprocessing and correction workspace are reused across
blocks and inference calls.

## Ping-pong prefetch

[FlashAttention-3](https://tridao.me/blog/2024/flash3/) overlaps data movement and
computation using Hopper's asynchronous hardware and scheduling. The transferable
idea implemented here is staging the next K/V tile while consuming the current
one. This is software prefetch with alternating LDS slots, not Hopper's TMA/WGMMA
or its complete warp-specialized ping-pong schedule.

The first tile is staged before the loop. Each iteration issues next-tile global
loads early in the current QK/softmax/PV work, then writes the prefetched data into
the alternate slot. A workgroup barrier protects reuse. One tile of zeroed
headroom makes the final lookahead safe. The arithmetic order is unchanged.
The compiled ISA places next-tile global loads among the current INT4 WMMAs and
keeps them outstanding during later QK and softmax instructions.

Compiled gfx1151 resource metadata:

| Variant | LDS bytes/workgroup | VGPRs | Private bytes/thread |
| --- | ---: | ---: | ---: |
| SA2 | 9,472 | 240 | 0 |
| SA2 ping-pong (initial build) | 16,896 | 240 | 0 |
| SA2 fast, eight waves | 18,944 | 240 | 0 |

The larger LDS footprint and additional scheduling work make ping-pong a choice
to measure, rather than an automatic replacement for the single-buffer kernel.

## Runtime selection and workspace

Both native and Python builders select the tuned source by sequence length.
Immutable caches fingerprint the source and configuration. `launch.txt` contains
`2 tokens m_group capacity attention_waves`; version 2 fixes the preprocessing
contract to 64-token query groups and transposed V. The host rejects old metadata
and incorrect wave counts before loading weights. GEMM groups are chosen internally
from 4, 3 and 2 to minimize padded rows. Deployment sources contain only the two
tuned attention kernels; the FP16 comparison kernel lives under `experiments/`.

SA2 supports 16 through 16,896 total tokens, head dimension 128 and four Q heads
per KV head. The pipeline uses 48 Q heads and 12 KV heads. Its largest additional
allocation is the fp32 correction:

`4 * 48 * ceil(tokens / 64) * capacity` bytes.

At 4,115 tokens this is about 50 MiB; at 16,384 it is about 770 MiB. Packed Q/K,
scales, means and the fp16 centered-K workspace add further storage. A future
fused correction kernel could reduce these global-memory costs.

## Validation and historical timing

```sh
scripts/test.sh --quick
env -u LD_LIBRARY_PATH .venv/bin/python tests/test_native_pipeline.py
env -u LD_LIBRARY_PATH .venv/bin/python tools/bench_sage_attention.py \
  --tokens 4115,8192,16384 --repeat 20
```

The kernel tests compare native preprocessing plus attention against an independent
Torch quantization oracle, exercise partial query/key tiles and constant centered
inputs, and require byte-identical outputs between the eight-wave and four-wave prefetch kernels. The oracle
uses fp32 correction inputs, so the comparison also checks the native correction's
fp16 approximation. Synthetic output cosine against the quantized oracle was
at least 0.9999999 on 16, 48, 65, 100 and 4,115 tokens. Against fp16 attention it was about
0.9991 at 4,115, 8,192 and 16,384 tokens; these synthetic inputs are not an image
quality benchmark.

A native eight-step 1024×1024 generation of “a red fox in the snow”, seed 0,
completed with ping-pong attention and produced a coherent image. RGB PSNR against
the native fp16-attention image with identical initial noise was 20.21 dB. Both
use the existing W4A4 block projections. This is one prompt, not evidence of
quality parity across prompts or denoising trajectories.

Before the H3-derived changes below, the following were observed mean milliseconds over 20 warmed calls, **with other
GPU workloads active**. They include all SA2 preprocessing. Contention varied
between variants, so these values cannot establish a speedup or rank the variants.

| Tokens | fp16 total | SA2 preparation + attention | Ping-pong preparation + attention |
| ---: | ---: | ---: | ---: |
| 4,115 | 29.04 | 7.43 + 18.91 = 26.34 | 9.06 + 26.73 = 35.80 |
| 8,192 | 112.72 | 16.04 + 84.22 = 100.26 | 15.84 + 73.14 = 88.98 |
| 16,384 | 608.19 | 57.00 + 397.36 = 454.36 | 54.16 + 442.13 = 496.29 |

Before selecting a default, repeat measurements on an idle GPU, alternate backend
order, and measure full-image latency with `tools/bench_native.py` under each
selector. Preprocessing can erase the INT4 QK savings, especially at shorter
sequences. Broader image comparisons across prompts, seeds and resolutions remain
necessary. All experimental modes are therefore opt-in.

## Tuning against minimax-h3-loom

The local H3 implementation (`../minimax-h3-loom/tools/gen_attention_i4qk.py`
and its development notes) supplied two useful scheduling ideas: transpose V
once globally so every attention tile uses wide LDS stores, and alternate LDS
slots to remove a barrier without retaining a prefetched tile in registers.
Krea's GQA ratio is four, unlike H3's MHA. Eight Krea waves therefore handle two
query tiles across four Q heads each, with one group loading K and the other V.
Both groups consume the same K/V tile. Inactive query tiles read a bounded tile
but never publish output, including a partial final query group.

The fixed gfx1151 dispatcher uses:

| Total tokens | Waves | Query rows/workgroup | Scheduling | V layout |
| ---: | ---: | ---: | --- | --- |
| 16–8,191 | 8 | 32 per Q head | alternating slots, one barrier/tile | transposed |
| 8,192–16,896 | 4 | 16 per Q head | explicit next-tile prefetch | transposed |

The second schedule won at both 8,192 and 16,384 tokens. The boundary is an
empirical choice from these sample lengths, not proof that it is optimal at every
shape or power setting. The earlier `sa2` and `sa2-pingpong` modes remain available
for comparison. All four kernels preserve the original quantized attention's
arithmetic and outputs.

SA2 preprocessing also changed. Q mean and quantization now use separate kernels;
this removes the previous 33 KiB LDS allocation and the 512-thread workgroup's
barrier tail. The sequential float32 mean and quantization arithmetic are
unchanged. K quantization now writes the centered fp16 K buffer in the same pass.
The fast mode additionally transposes V with a tiled HIP kernel once per block.
This costs `2 * capacity * 12 * 128` bytes of reusable workspace.

In an interleaved old/new preprocessing comparison at 4,115 tokens, the old
preparation took 12.02–13.06 ms and the new one 7.38–9.24 ms. Outputs matched
exactly. Other GPU work was active; these are observations in that run, not an
isolated-hardware throughput claim.

The initial schedule sweep used identical synthetic inputs and three rounds of
five warmed calls, reversing variant order between rounds. The following medians
are **attention only**, excluding preparation and the one-time V transpose:

| Tokens | Original SA2 | Original prefetch | Transposed V + prefetch | Transposed V + eight-wave double buffer |
| ---: | ---: | ---: | ---: | ---: |
| 4,115 | 20.26 ms | 18.52 ms | 17.39 ms | 17.04 ms |
| 8,192 | 88.81 ms | 78.05 ms | 65.92 ms | 68.82 ms |
| 16,384 | 400.60 ms | 405.54 ms | 247.77 ms | 269.95 ms |

Contention was still present. The long-sequence improvement appeared in every
round, while the short-sequence ranking varied. Eight waves increase K/V reuse
but do not automatically improve occupancy or latency. Padding LDS merely to cap
occupancy did not provide a consistent benefit. The shipping benchmark now
alternates order and includes the actual native V transpose and preprocessing;
use it for complete attention cost, rather than treating this schedule sweep as
an inference benchmark.

A follow-up with the shipping benchmark included the actual HIP transpose and
all preprocessing, two reversed-order rounds of three warmed calls. At 16,384
tokens the selected prefetch path totaled 342.20 and 321.52 ms, versus the
original SA2's 430.41 and 422.55 ms; outputs were identical. At 8,192 tokens the
selected path totaled 97.59 and 100.55 ms, close to the original's 102.92 and
91.77 ms. These loaded-system results reinforce the long-sequence improvement
and show why the sequence-length dispatch remains an empirical heuristic. All four kernels
matched exactly at both sequence lengths, including native V transposition.

H3's rotation-only quantization was not adopted: the earlier Krea first-block
probe strongly favored centering. Nor is INT4 PV enabled without a separate
accuracy study. The inexpensive score correction remains in place. A future
change to its arithmetic needs full-trajectory quality validation.

The native pipeline now also keeps the residual stream on the GPU at the block
boundary. Its bf16-to-fp16 and fp16-to-bf16 conversions use HIP kernels rather
than downloading, converting on the CPU and uploading the full stream twice per
step. The public C ABI and host-memory block API are unchanged. The native compiler
also now chooses a GEMM raster group that minimizes padded tile rows, matching
the Python builder: at 4,115 tokens, group 3 executes 33 tile rows instead of
group 4 executing 36. Earlier profiles
put those conversions at about 100 ms/step at 1024×1024. This removes those transfers from each denoising step.

## Native image checks after tuning

These measurements and image hashes predate the correction to CUDA scheduler
delta rounding described in [the runtime review](notes.md#native-runtime-review-fixes).

The native C-API benchmark generated 1024×1024 images, eight steps, prompt
“a red fox in the snow”, seed 0. Loading is excluded; the first generation
includes block-session preparation. `libkrea2_pipeline-before-h3.so` is the saved
library from before this tuning pass.

| Library and attention | First generation | Warm 1 | Warm 2 |
| --- | ---: | ---: | ---: |
| Saved library, fp16 | 57.84 s | 66.10 s | 72.11 s |
| Pre-simplification library, fp16 | 68.67 s | — | — |
| Tuned library, INT4 | 34.97 s | 21.78 s | 21.53 s |
| Saved library, sa2-pingpong | 48.97 s | 28.31 s | 38.04 s |

Other GPU work and changing memory pressure affected these sequential runs;
**do not interpret their ratios as isolated speedups**. The fast path's observed
warm mean is 21.65 seconds from two samples. All fp16 RGB checksums match the
saved fp16 image, and all SA2 RGB checksums match the saved SA2 image:

- fp16: `36293aa77377f808462b5b56112424d254cf89d434db47b8b938911c3e724ff7`
- SA2: `31a19cff459feb1f3ac36b496b50ad7b8a40e9bf4810d7b61591ce90f7ce26da`

Canonical generation, host/cache/API regressions and all SA2 kernel edge tests
passed. The full Torch-reference pipeline run passed through the complete
transformer, equal-area rectangles and three VAE sizes, but was intentionally
terminated when the shared machine exhausted RAM and nearly filled swap. Its
remaining scheduler/thread checks were not completed for this revision. The
native-only image comparisons above subsequently verified the full eight-step
path and both GPU storage conversions without loading a second reference model.

## Fixed-path validation

The runtime now uses the tuned path unconditionally. After removing the selector
and fallback branches, the quick suite passes and the first real block matches
the smoothed INT4 storage oracle at update cosine 0.999946. Native composition is
exact. A default C-API eight-step 1024×1024 image matches the saved SA2 RGB hash
above on both first and warm calls (208.95 and 88.24 seconds). Other video-model
and game work was active; these are correctness and session-reuse checks under
contention, not isolated throughput measurements.
The full Torch-reference pipeline suite was not rerun for this simplification.

## Earlier model-derived feasibility probe

Run:

```sh
env -u LD_LIBRARY_PATH .venv/bin/python tools/probe_attention_i4.py
```

This earlier probe uses 16-token query means, whereas the native backend uses 64.
Its numbers must not be attributed to the current kernel. The tool advances the fixture's residual stream with the native Loom blocks, then
projects Q/K/V with the existing Torch storage-boundary oracle. It evaluates 128
query positions against all 4,115 keys and all 48 query heads at timestep 1.0.
It samples zero-based blocks 0, 13 and 27. PV remains fp32 in this numerical probe
so the comparison isolates QK quantization. This is not a kernel benchmark or an
exact implementation of SageAttention2's per-thread quantization.

| Block | Raw per-token INT4 cosine | K centered | Q/K centered + correction | K centered + Hadamard |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.971123 | 0.993319 | 0.998870 | 0.936237 |
| 13 | 0.993636 | 0.993098 | 0.996786 | 0.996425 |
| 27 | 0.997013 | 0.997210 | 0.999117 | 0.998553 |

The worst first-block head improves from cosine 0.550766 with raw INT4 to 0.992433
with Q/K centering. The centered variant's worst middle-block head is 0.988733,
so aggregate cosine alone is insufficient. The tested deterministic Hadamard
transform is applied after RoPE, but performs poorly on the first block; it is not
FA3's complete low-precision recipe.
