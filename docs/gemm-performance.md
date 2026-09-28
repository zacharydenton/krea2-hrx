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
