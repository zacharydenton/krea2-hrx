# Recording the block loop

Each session records its transformer blocks once and replays them through HRX.
The session uses its creating stream and owns every recorded workspace buffer.
External residual and modulation inputs are copied into those fixed buffers;
RoPE contents are updated when their fingerprint changes.

Profiling (`KREA2_NATIVE_PROFILE=1`) replays a second recording built with
`Graph::finish_profiled`: HRX writes a GPU-clock timestamp before and after each
kernel, and the session sums the intervals per stage over all 28 blocks, with
the idle time between kernels as its own line. The markers add completion
barriers, so kernels run one at a time: the numbers are per-kernel costs, not
the latency of an unprofiled forward. It needs the native bundle of hrx-rs
0.8.10 or later, whose bridge exports HRX's profiling ABI; with an older bridge
a profiled forward fails with `Unsupported`. The direct `Session::run` is not
profiled.

`KREA2_PROFILE_JSON=FILE` also appends each profiled forward to `FILE` as one
JSON line: the sequence length, layer count, attention mode and HRX's raw
`DeviceProfile` (labelled intervals in device ticks, the counter frequency,
and the interval union, span and gaps). Compare two builds by their per-label
sums rather than by the printed table, which is rounded.

For kernel work, `cargo run --release --example gemm_ab -- BASELINE.loom
CANDIDATE.loom` times int8 GEMM sources the same way at the four transformer
shapes, interleaving them round by round with a register-only WMMA peak
(`experiments/peak_i8.loom`), so each result is a fraction of the ceiling
measured under the same contention. Candidates must match the baseline byte
for byte; `--no-check` admits diagnostic variants that deliberately skip work.

Replay has not demonstrated a consistent end-to-end speedup on gfx1151. It
reduces host recording work, but native barriers, partitioning and GPU resource
use determine the completed time. Grid size alone does not establish saturation.

## Dependencies

The block loop is a chain: each launch waits for the one before it, which orders
shared scratch reuse and residual updates.

Fan-ins go on their consumer directly rather than through an empty join node. In
the pinned native runtime an empty node splits the command buffer and adds a
queue barrier.
The scheduler considers additional workstreams only after the first 16
recordable nodes in a partition. Removing unnecessary barriers can help even
within one workstream; declaring a DAG does not guarantee simultaneous execution.
The native API currently exposes no partition or workstream counters.

## Measurements and limits

A previous full-pipeline A/B/B/A comparison measured 29.58 s direct against
29.21 s recorded, with the two rounds disagreeing on the sign. That establishes
no consistent win. Historical host submission measured 238 ns per dispatch;
it does not measure the GPU or give a fixed cost for graph edges.

The retired smoothed-attention preparation, recorded as a seven-kernel diamond,
measured 244.3 ms against 246.0 ms as a chain over 28 passes (2026-09-10):
declaring the concurrency was free but bought nothing.

The overlap benchmark measures two sessions with two blocks each. It initializes
distinct finite inputs, resets them before each sample and checks both outputs
against eager execution. Measurement order alternates, with three warmups and
nine samples per arm. Uploads, reset copies and checks are outside the timer.
A 2026-09-10 run produced these medians; every output matched eager execution.
The independent schedule did not improve these cases.

| Tokens | Serial | Independent |
| --- | ---: | ---: |
| 275 | 66.7 ms | 70.6 ms |
| 1043 | 184.1 ms | 211.3 ms |
| 4115 | 737.6 ms | 755.8 ms |

These figures describe two block prefixes. A full-forward overlap experiment
would need all 28 blocks and the surrounding pipeline work.

A separate historical VAE experiment measured 578 ms with queued tile downloads
against 569 ms with blocking downloads. That configuration showed no improvement;
it does not set a general limit on transfer overlap. Decoding now queues every
tile before reading any back, which is byte-identical; it has not been
re-measured on an idle GPU.

```sh
cargo run --release --example dispatch_cost
cargo test --release --lib -- --ignored --nocapture \
  two_independent_block_prefixes_are_priced_against_a_chain
```

## Correctness and ownership

`a_recorded_block_loop_replays_to_the_same_bytes` compares eager execution with
successive replays after changing input bytes, modulation, RoPE and external
allocation addresses. It also checks stream rejection before storage is changed,
profiling after recording, and destruction with a replay pending.

Graph instantiation retains native resources. Model pools must additionally keep
their Rust allocation owners alive while a recording can be replayed: native
retention prevents deallocation, but does not prevent pool reuse. This session
uses fixed, unpooled workspaces. HRX's completion-point destructor waits for the
last replay without flushing unrelated work on the stream.

On 2026-09-10 the replay regression passed in each attention mode offered at
the time against HRX 0.2.0. Locked workspace builds, CPU tests, clippy and rustdoc
with warnings denied also passed using the crates.io package.
