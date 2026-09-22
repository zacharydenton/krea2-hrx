# Native Loom text-fusion result — 2026-09-22

The native port executes correctly but fails the performance gate. Auto remains
GPU. Explicit `--features npu --fusion-backend npu` selects this experiment.

| Completed work | GPU median | Native NPU + GPU handoffs median |
| --- | ---: | ---: |
| Real 156×2560→6912 fusion projection | 0.790 ms | 1407.579 ms |
| Full 256×256, two-step generation | 794.035 ms | 2193.577 ms |

Stage p95 was 1.231 ms on GPU and 1420.176 ms on NPU.
Stage measurements come from one fresh process with ten warmup pairs followed
by 100 pairs in alternating backend order; they include all packing, copies,
cache maintenance, native execution, GPU reduction/bias and completion.
Compilation and correctness readback are excluded. Five-process qualification
was stopped after this clear latency failure. An earlier exploratory timing
run overlapped CPU verification and is excluded from the table.

Each production backend ran in one fresh process with seven warm samples after
an initial generation. The prompt was “a red ceramic cup on a wooden table”,
seed 37, Turbo int8 ConvRot. These are small-case measurements on the shared
Strix Halo desktop, not peak NPU throughput or a broad workload benchmark.
GPU/NPU workloads were serialized.

Five synthetic shapes passed exact BF16 comparison to a scalar f64 oracle on
three changed-input replays each, including M/K/N tails, split K, absent/present
bias and a width crossing the 256-tile DMA chunk boundary. The real projection
passed 1,024 sampled f64 checks on original and changed inputs. Full-output
relative RMS versus GPU was 0.000139777 and
0.000127804, respectively.

Native tracked storage was 82,001,408 bytes. Eight tracked
allocations and three imports remained constant during warm replay. The
production NPU path replayed deterministically, retained a stable residency
budget and released it on teardown. Default and feature-enabled tests passed
(61 and 62 tests); Clippy passed with warnings denied.
The unchanged GPU production path still matched the frozen pre-migration
transformer and image fixture byte-for-byte, including cancellation/retry.

NPU and GPU images differ: their mutual PSNR was 32.375 dB.
This is not PSNR against the unquantized reference and does not establish a
quality pass or fail. The original immutable `build/quality` fixture is absent;
no replacement baseline was minted. Full reference-quality qualification is
therefore incomplete, independently of the failed latency gate.

The implementation uses one native BF16 worker. Four-worker routing exhausted
Loom stream channels; a graph with one native run per matrix tile exhausted
hardware contexts. Reusing one run and zero-offset staging avoids those failures,
while chunking width respects the 256-repeat shim DMA limit. This port establishes
a working native path; a faster tiling/layout and fewer handoffs are needed
before repeating qualification. The historical Chess implementation's different
shape/runtime measurements are not directly comparable.

[Raw samples, hashes and runtime identity](benchmarks/npu-fusion-native-2026-09-22.json)
are checked in. [Usage and reproduction](npu-fusion.md) describe the native path.
