# Larger K tiles in the INT8 GEMMs

The QKV/gate, gate/up and attention-output projections now stage 128 K elements
per iteration instead of 64. Their output tile remains 256x128 with eight waves.
For the production K=6144, this halves the loop iterations and staging barriers
from 96 iterations to 48. Eight 16-wide WMMA substeps consume each stage, and
registers prefetch the next stage while the current one is consumed.

The operands remain INT8, accumulation remains INT32, and all scale, BF16
rounding, residual and SwiGLU epilogues are unchanged. This is a scheduling change,
not a reduction in precision. The narrow kernels now require K divisible by 128;
the model's dimensions satisfy this. The down projection retains its existing
256x256 output tile and 64-element K step.

Each workgroup uses 55,296 bytes of LDS (384 rows with a 144-byte pitch), up from
30,720 bytes. On gfx1151 the compiler reports 256 VGPRs, no spills and 25%
occupancy, down from 37%. Lower occupancy is worthwhile here because it removes
enough staging and loop overhead. A 96-element K step, wider load packets,
additional scheduling fences, and a 512x128 output tile did not consistently
improve on the retained implementation.

## Whole-image measurement

On 2026-09-28, pooled median prompt-to-RGB latency fell from **25.960 seconds to
24.112 seconds: 7.1% less time**, or 1.85 seconds saved per eight-step image.

| Batch, in execution order | Images | Median (s) | Range (s) |
| --- | ---: | ---: | ---: |
| Baseline first | 3 | 26.222 | 26.065–26.933 |
| Candidate | 6 | 24.112 | 24.018–24.990 |
| Baseline repeat | 3 | 25.712 | 25.514–25.855 |

The candidate is 8.0% faster by latency reduction than the first baseline median
and 6.2% faster than the repeat baseline median. Every candidate sample was faster
than every baseline sample. All images were byte-identical across builds and
replays, with zero tracked warm allocations and unchanged reserved memory.
The [raw record](benchmarks/gemm-k128-2026-09-28.json) includes every whole-image
and isolated-kernel sample, source and binary hashes, and compiled resource reports.

Both builds include the earlier [preparation optimization](prepare-performance.md),
so the comparison isolates the three GEMM changes. Measurements use 1024x1024,
eight Turbo steps, seed 37 and prompt `a red ceramic cup on a wooden table`.
Elapsed wall time includes text encoding, denoising, VAE decoding and RGB
readback. Model loading, warm-up, file writing and output comparison are excluded.

Only one benchmark pipeline is resident at a time. Each process warms once, then
runs its measured batch. The order is three baseline images, six candidate images,
and three baseline images. No other benchmark or profiler from this task runs
during the timed batches. An unrelated application continues sharing the GPU,
so these results describe the measured shared-machine workload, not idle-machine
latency. Model loading between batches prevents a tightly interleaved comparison.

## Kernel measurement

On AMD Radeon 8060S / gfx1151 with HRX 0.8.11 and its pinned Loom compiler:

| Projection at 4115 tokens | Before (ms) | After (ms) | Reduction |
| --- | ---: | ---: | ---: |
| QKV/gate, K=6144, N=15360 | 19.802 | 18.731 | 5.4% |
| Gate/up + SwiGLU, K=6144, N=32768 | 39.436 | 38.129 | 3.3% |
| Attention output + residual, K=6144, N=6144 | 7.956 | 7.384 | 7.2% |

Each comparison rotates the baseline, candidate and register-only WMMA peak
through 21 measured rounds of five dispatches, after two warm-up rounds.
GPU-clock timings exclude host overhead. The harness checks baseline/candidate
output equality before timing. Residual timings repeatedly update the output;
the independent oracle checks the actual residual arithmetic separately.
Additional token counts 275, 1043 and 9235 are retained in the raw record.
Shared-memory contention particularly affected the largest QKV/gate case, so
its large isolated gain should not be generalized.

## Reproduction and correctness

```sh
git show 470f60c:kernels/gemm_i8_256.loom > /tmp/gemm-before.loom
cargo run --release --example gemm_ab -- \
  --shape 4115,6144,15360 --rounds 21 --json /tmp/gemm.jsonl \
  /tmp/gemm-before.loom kernels/gemm_i8_256.loom
cargo run --release --example bench_runtime -- /tmp/image.rgb 1024 3 8
cargo test --release --test quantized integer_gemms -- --ignored --nocapture
scripts/parity.sh
scripts/test.sh --cpu
```

For fused kernels, compare the corresponding historical and current sources;
pass `--residual` for the attention-output projection. Build and preserve each
whole-pipeline binary separately, changing only these three GEMM sources between
builds, then run baseline/candidate/baseline batches. `bench_runtime --interactive`
accepts `run` and `quit` after loading and warming, and reports each sample as JSON.

The expanded CPU-oracle GPU test covers all three epilogues, dense and padded
operand rows, one and three K iterations, partial row tiles, and multiple tiles
in both output axes. The full-image replay checks compare every RGB byte and
require stable reserved memory; the frozen BF16 parity gate independently checks
the complete denoising trajectory and decoded image.

All checks passed, including the CPU suite and the full eight-step frozen BF16
gate at 1024x1024 with the separate prompt `a red fox sitting in fresh snow at
dawn, soft light, photograph`. Latent cosine was 0.997044, relative RMS 0.076912,
and decoded image PSNR 31.777828 dB. Both latent and image regression against the
accepted baseline were 0.000000 dB. The reference and accepted baseline fixtures
were not regenerated.

## 1536x2048 follow-up

A separate shared-GPU run on 2026-09-28 measured **130.786 seconds median** for
1536x2048 at eight Turbo steps. The three samples after one warm-up were 130.786,
124.154 and 132.418 seconds. The prompt, seed and full prompt-to-RGB timing scope
match the 1024x1024 benchmark above, but these runs happened later under potentially
different external load. They are not a controlled measurement of resolution scaling
or of the K-tile optimization's benefit at this size.

[TheNoise 0.9.0's published table](https://github.com/lemonade-sdk/thenoise/blob/7633bfda1e3701b781d023f7ad42fe20304b040a/README.md#performance)
reports 99 seconds for INT8 ConvRot and 117.6 seconds for BF16 on Strix Halo at
1536x2048, eight steps, after warm-up. Our median here takes 32.1% longer than
their published INT8 time. This is a cross-project comparison, not a controlled
head-to-head test; power limits, prompts and GPU contention may differ.

The run exposed a host-side replay bug: a pooled modulation allocation can be
larger than its logical table after VAE buffer reuse. Copying its whole binding
into the fixed-size block input failed with `copy requires equal spans`. The
handoff now slices the source to the logical destination length, consistent with
HRX's pool contract. The focused GPU regression fails before this fix and passes
after it; both shared-bridge GPU tests and the CPU suite pass. No kernels or
arithmetic changed. All repeated images match each other and the successful
pre-fix first image byte for byte.

The benchmark accepts `WIDTHxHEIGHT` and reports per-image reserved memory rather
than rejecting any change in pooled capacity. Here it varied by 110,592 bytes
across the timed images around 24.1 GB, with zero new tracked warm allocations;
all reserved memory was released when the pipeline dropped.

```sh
cargo run --release --example bench_runtime -- /tmp/image.rgb 1536x2048 3 8
```

The [raw record](benchmarks/generation-1536x2048-2026-09-28.json) retains every
sample, memory readings, hashes and comparison metadata.

## Attention query sharing at large resolutions

A synchronized profile at 1536x2048 (12,301 tokens including the prompt) puts
attention at 8,483 ms of a 15,502 ms transformer step, or 54.7%. The new
`attention_gqa_lds_f16_wmma_q32` kernel lets two independent groups of sixteen
query rows share each staged K/V tile. Each wave retains the original FP16
matrix products and FP32 online softmax order. This is separate from the old
experimental query32 fragment repacking kernel.

The builder and runtime select the eight-wave kernel at 8,192 tokens and above.
Smaller sequences keep the original four-wave kernel. Both compile without
scratch spills. A sequential sweep on the shared gfx1151 GPU gave these medians:

| Tokens | Original attention (ms) | Two query groups (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 5,123 | 25.008 | 25.147 | 0.994x |
| 6,163 | 37.351 | 36.182 | 1.032x |
| 8,195 | 71.434 | 64.485 | 1.108x |
| 12,301 | 180.818 | 144.449 | 1.252x |

Every output element was bit-identical. Each size alternated kernels for seven
rounds with five samples per round. No other benchmark owned by this session ran
concurrently; an unrelated GPU application remained active. Earlier overlapping
screening runs are excluded. These are kernel timings, not whole-image gains.
The [raw samples](benchmarks/attention-query-groups-2026-09-28.json) retain the
measurement scope and source identity. Both GPU numerical and compiler regression suites pass, including independent
CPU softmax checks of both query tilings. The stricter comparison that poisons
output before each kernel also passes at 8,191, 8,192, 12,301 and 16,403 tokens.
At the two crossover sizes, the original benchmark used 32 extra padding rows.
The rerun with the actual 8,224-row capacity remains bit-identical: 73.735 to
67.375 ms at 8,191 tokens, and 88.834 to 65.100 ms at 8,192 tokens. Those raw
samples are included separately in the measurement record.

The local TheNoise comparison was interrupted by a GPU reset during its first
1536x2048 decode. After recovery, our full-image run completed 512x512 at 5.919 s
median, then stalled after the 1024x1024 warm-up with further driver queue errors.
The job was terminated; the GPU remained busy with no compute clients. The
[partial local record](benchmarks/thenoise-local-2026-09-28.json) distinguishes
completed measurements from failures. The recovered 1536x2048 eight-step warm image is byte-identical to the original
attention kernel (SHA-256 `66b46dac8451cc2461142d5fc2f5af636c0b20af64cd2a0486ebb1a873e3f982`).
A user-authorized driver reset at 12:36 CEST restored idle GPU activity and
passed copy/readback and attention compute checks without rebooting. The resumed
1536x2048 run measured **96.955 seconds median**, with samples 95.858, 97.607 and
96.955 seconds. All three outputs match the original attention image exactly.
The [full-image record](benchmarks/generation-q32-1536x2048-2026-09-28.json)
retains raw timings, hashes and source identity. This is 2.1% less time than
TheNoise's published 99 seconds. The resumed local run measured TheNoise at
186.204 seconds median (186.204, 192.040 and 185.153), including decoding,
with `MIOPEN_FIND_MODE=FAST`. Its pre-decode median alone was 168.084 seconds.
The apparent 1.92x full-generation speedup remains provisional: other GPU
workloads were active and the engines ran at different times. The user accepted
the measured optimization and ended the remaining work on 2026-09-28. The
[28-case size and boundary sweep](benchmarks/resolution-matrix-2026-09-28.json)
therefore remains partial, and the later drift recheck was cancelled. These
results do not establish superiority at every resolution.

| Resolution | HRX median (s) | Local TheNoise median (s) |
| --- | ---: | ---: |
| 512x512 | 7.255 | 9.374 |
| 1024x1024 | 27.086 | 36.362 |
| 1536x2048 | 96.955 | 186.204 |
| 2048x2048 | 144.920 | Not completed |

Each completed generation case uses eight steps and three timed images after
one warm-up, with exact within-engine replay. The frozen
1024x1024 BF16 quality gate passes at 31.777828 dB image PSNR with 0.000000 dB
regression in either latent error or image PSNR against the accepted baseline.
This fixture exercises the original attention tile; the independent CPU oracle
and exact large-sequence/full-image comparisons cover the new query grouping.

```sh
cargo run --release --example attention -- 12301 /tmp/attention-q32 --rounds 7 \
  kernels/attention_gqa_lds_f16_wmma.loom \
  q32:kernels/attention_gqa_lds_f16_wmma_q32.loom
cargo run --release --example bench_runtime -- /tmp/images \
  512,1024,1536x2048,2048 3 8
```
