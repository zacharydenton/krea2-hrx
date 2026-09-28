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
