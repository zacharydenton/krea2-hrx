# Recording the block loop

Each session records its transformer blocks once and replays them through HRX.
The session uses its creating stream and owns every recorded workspace buffer.
External residual and modulation inputs are copied into those fixed buffers;
RoPE contents are updated when their fingerprint changes. Profiling dispatches
directly, including after a recording has been cached.

Replay has not demonstrated a consistent end-to-end speedup on gfx1151. It
reduces host recording work, but native barriers, partitioning and GPU resource
use determine the completed time. Grid size alone does not establish saturation.

## Dependencies

The block loop orders shared scratch reuse and residual updates. Sage's seven
preparation kernels form a DAG: the key and query branches have separate storage,
V transpose is independent of both, and correction consumes the two branches.
Attention waits directly for query quantization, V transpose and correction.
Correction already depends on K quantization, so a fourth edge is redundant.

There is no empty join node between preparation and attention. In the pinned
native runtime an empty node splits the command buffer and adds a queue barrier.
The scheduler considers additional workstreams only after the first 16
recordable nodes in a partition. Removing unnecessary barriers can help even
within one workstream; declaring a DAG does not guarantee simultaneous execution.
The native API currently exposes no partition or workstream counters.

## Measurements and limits

A previous full-pipeline A/B/B/A comparison measured 29.58 s direct against
29.21 s recorded, with the two rounds disagreeing on the sign. That establishes
no consistent win. Historical host submission measured 238 ns per dispatch;
it does not measure the GPU or give a fixed cost for graph edges.

The Sage benchmark records 28 preparation passes as either a diamond or a chain.
It alternates measurement order, uses three warmups and nine samples per arm,
and checks outputs against eager execution. On 2026-09-10 it measured 244.3 ms
for the diamond and 246.0 ms for the chain. These passes omit the transformer
kernels that would normally separate them, so their native schedule can differ
from a complete block loop.

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
it does not set a general limit on transfer overlap.

```sh
cargo run --release --example dispatch_cost
cargo test --release --lib -- --ignored --nocapture \
  the_declared_concurrency_is_priced_against_a_chain \
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

On 2026-09-10 the replay regression passed with `KREA2_ATTN_QK=16`, `8` and
`4` against HRX 0.2.0. Locked workspace builds, CPU tests, clippy and rustdoc
with warnings denied also passed using the crates.io package.
