# Recording the block loop

The 28 transformer blocks are recorded once per session as an HRX dependency
graph and replayed each forward, rather than dispatched launch by launch.
`Session::blocks_through` records on first use; profiling still dispatches
directly, because it synchronizes between stages and a replay cannot stop to do
that.

**This is deliberate infrastructure, not a measured win.** Everything below says
so. It is kept because the graph is where the runtime's scheduling work lands,
and because a recording states the pipeline's real dependency structure in a form
a future runtime can exploit. The measurements are here so the decision is
re-checkable rather than remembered.

## What it costs and what it saves

Host submission is **238 ns per dispatch** including the operation-cache lookup
and constant packing (`crates/kernels/examples/dispatch_cost.rs`). A forward is
280 dispatches, so replay can save at most **67 µs**. HRX documents about
**0.95 µs per graph edge**, so a 279-edge serial chain costs roughly **265 µs**.
Both are about 0.01% of a 3674 ms forward, which is why the end-to-end paired
comparison could not resolve a difference: A/B/B/A over two rounds gave medians
of 29.58 s direct against 29.21 s recorded, with the two rounds disagreeing on
the sign.

## Why there is no concurrency to expose

A graph pays when independent work can overlap. Three measurements, from
different directions, say this pipeline has none.

**The Sage preparation pass is a genuine DAG and it does not matter.** Its seven
launches form a diamond — the key side and the query side touch disjoint
buffers, the V transpose touches neither, and only the correction GEMM reads
both — for a critical path of four instead of seven. Recorded 28 deep, with no
checkpoint and no disk involved:

| | replay of 28 passes | median |
| --- | ---: | ---: |
| diamond | 191.5, 196.0, 190.6, 190.3 ms | 191.1 ms |
| chain | 192.1, 192.3, 191.3, 192.5 ms | 192.2 ms |

0.6%, inside the spread. The launch grids explain it: `sage_query_mean` is 3120
workgroups and `sage_key_partial` is 780, against 40 CUs. Each kernel already
fills the device.

**Two whole forwards do not overlap either.** Two sessions sharing one set of
weights but with their own activation buffers, recorded into one graph twice —
once as two chains the runtime may overlap, once as one chain end to end:

| tokens | image | overlapped | sequential | delta |
| ---: | --- | ---: | ---: | ---: |
| 275 | 256×256 | 44.06 ms | 43.92 ms | +0.3% |
| 1043 | 512×512 | 104.27 ms | 104.55 ms | −0.3% |
| 4115 | 1024×1024 | 417.52 ms | 421.12 ms | −0.9% |

Swept over sequence length because saturation is a property of the launch grids,
and those shrink with the image. It does not break down even at 256×256.

**The VAE's blocking readbacks were not costing anything either.** Decoding
1024×1024 waits on 36 tile downloads, each of which drains the stream. Replacing
them with queued readbacks and one wait measured 578 ms against 569 ms — inside
the spread — and was reverted. Each tile's host work is a 32 KB memcpy, so the
round trips cost single-digit milliseconds out of 570; the rest is real GPU work,
about 15.8 ms per tile for a hundred dispatches on a 256×256 image.

## The control

Equal timings would also be what a scheduler that never partitioned the work
produced, so the negative results depend on the mechanism working. HRX's own
`independent_nodes_beat_a_serial_chain` replays 64 tiny fills in about 88 µs
declared independent against 151 µs chained, and passes on this machine. The
runtime does overlap independent nodes. This workload gives it nothing to do.

## What would change the answer

- **Kernels fast enough for submission cost to matter.** At 238 ns against
  13 ms per dispatch, host cost is 0.002% today.
- **Work that leaves the device idle.** Every measurement above is against
  kernels whose grids fill a 40-CU GPU at every shape tested.
- **A runtime that schedules across streams or devices**, where a recording is
  the unit of work rather than a replay of one queue.

Re-run the evidence with:

```sh
cargo run --release -p krea2-kernels --example dispatch_cost
cargo test -p krea2-session --lib -- --ignored --nocapture \
  the_declared_concurrency_is_priced_against_a_chain \
  two_independent_forwards_are_priced_against_one_after_the_other
```

## Correctness

Replay must produce exactly what direct dispatch produces, and does:
`a_recorded_block_loop_replays_to_the_same_bytes` compares the two byte for byte
and asserts the recording was actually taken, and `scripts/parity.sh` reports the
full 1024×1024 trajectory bit-identical to the frozen baseline. The Sage DAG was
checked the same way at `KREA2_ATTN_QK=4` and `=8`, three runs each, to rule out
a race that happened to win.

Two constraints the recording imposes, both load-bearing:

- **Every binding must be session-owned**, because a recording fixes addresses.
  `run_device` therefore copies the caller's modulation into the session's own
  buffer, as it already did for the residual stream.
- **`Session.graph` is declared before the buffers it records**, because fields
  drop in declaration order.

Releasing a graph while a replay is still running frees native structures the
device is reading; it surfaced here as an AMDGPU memory access fault during an
unrelated session's weight upload. Fixed upstream in `hrx-rs` f1b62a9, where
`GraphExec::drop` now drains its stream first.
