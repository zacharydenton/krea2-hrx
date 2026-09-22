# Shared HRX runtime

Krea uses [`hrx-rs` 0.8](https://crates.io/crates/hrx-rs) for keyed kernel
requests and the coordinated graph API.
The manifest renames it to `hrx`, so call sites read `hrx::`.
GPU, NPU and Loom now use the unified HRX 0.8 native bundle.
`Cargo.lock` pins the complete dependency
graph. The shared implementation owns native loading, status conversion,
allocation, streams, dispatch lifetimes and compiler caching.
The unified native bundle loads dynamically; no runtime rpath is embedded in the binary.

Each pipeline owns an explicit HRX `ModelContext`; `Pipeline::open_in` lets an
application share that context with other models. Transformer block plans use
one private inference slot each, retaining their native stream, graph and RoPE
tables. HRX's bounded `PlanCache` keeps two idle-evictable shapes, keyed by image
width, height and text length, so equal token counts do not alias different RoPE
geometries. Immutable block weights are shared across those plans.

Auxiliary models and their tensor pool use a separate ordered native stream.
Owned device-copy handoffs drain each boundary without reading intermediate
latents back to the host. A failed handoff quarantines its stream and captured
owners; later calls reject reuse. Calls sharing one pipeline remain serialized.
`Pipeline::context` exposes coordinated statistics, which do not include native
model weights or auxiliary pools and are not a total GPU memory measurement.
When the context has a `memory_budget`, all pipeline GPU streams use it before
allocating, including native weights, auxiliary pools, block workspace and
transfer staging. Residency statistics include those native allocations;
coordinated Runtime statistics still cover only tracked tensors. Driver/compiler memory
and native allocator rounding remain outside it.

Transfers and dispatch use HRX `View` regions. Uploads copy into HRX-owned staging
and queue work on the stream. Weight loading uses chunks of at most 16 MiB;
unpadded checkpoint rows go directly from the mapping to HRX staging, while
padded rows are assembled in a reusable host buffer. Blocking reads wait for
completion. Profiling and progress callbacks add explicit waits when needed.

Tensor storage returns to the model's pool when its final owner is dropped.
One pool serves one ordered stream and refuses another. Buffers are device-scoped, so HRX permits a block on any stream and
`record_event`/`wait_event` order that use — but a pooled block returns to the
free list while the work reading it is still queued, and reissuing it is safe
only because the stream that runs that work also runs whatever writes it next.
Across streams there is no such order, and no event can impose one on a reuse
already handed out.

The compiler and runtime come from HRX's public, verified native bundle:

Install the matching CLI from crates.io:

```sh
cargo install --locked hrx-rs --version 0.8.0
hrx prepare
cargo build --release
```

For local HRX development, add a temporary patch in `Cargo.toml`:

```toml
[patch.crates-io]
hrx-rs = { path = "../hrx-rs" }
```

Cargo updates the lockfile for a path override. Restore the registry dependency
before committing a lockfile generated with a local override.

`HRX_RUNTIME_DIR` selects a trusted native directory and `HRX_OFFLINE=1` refuses
network provisioning. The artifact cache follows XDG: `$XDG_CACHE_HOME/hrx`,
else `$HOME/.cache/hrx`. `HRX_LOOM_LIBRARY` or an explicit model compiler argument
selects a compiler override.

Krea's shape/bundle metadata and model-specific operation builders remain here.

Auxiliary dispatches use HRX's `KeyedKernels` with the kernel name, dimensions,
grid and report setting as their key. Only a miss constructs a specialization
and hashes the embedded source. HRX owns the key index and the artifact cache;
different keys for the same artifact share one loaded executable. Failed
requests remain retryable, and cache hits still check the device.

On gfx1151 with HRX revision `cd64a45`, an optimized warm-lookup benchmark on
2026-09-10 measured:

| Kernel | HRX source lookup | Krea through HRX's key index |
| --- | ---: | ---: |
| Unary | 1,859 ns | 307 ns |
| Wide GEMM | 37,547 ns | 532 ns |

These are host lookup costs, not end-to-end inference timings. The source path
already has its specialization constructed before timing; the keyed path
includes Krea's request-key construction. Each figure is the median of nine
samples after three warmups, with alternating measurement order. Reproduce with
`cargo run --release --example kernel_lookup`.
Both auxiliary and block compilation use the stream's target and `hrx::loom`.
Compiler selection is cached by library and target. Its bounded module cache is
kept across guidance shapes, avoiding repeated parsing and indexing. Prepared
operation sets retain loaded kernels for their lifetime; there is no global
loaded-kernel cache holding model resources after teardown. Artifacts load from
owned bytes without a compiler subprocess or an extra filesystem round trip.

Kernel sources live in `kernels`; tokenizer assets live in
`assets`. Generators and tests use these paths directly.
Package builds carry the same assets without depending on files outside the package.

Applications combining model crates should resolve to one HRX package version
and source, sharing a `ModelContext` when they need coordinated ownership.

## Interface

The `krea2` library is the interface. Consuming applications use its
`krea2::pipeline` and `krea2::session` modules directly; an Elixir application
wraps `krea2::pipeline::Pipeline` with Rustler. There is no C
ABI, no generated header and no error-buffer protocol: arguments are Rust types,
failures are `Result`, and Rustler contains panics at the NIF boundary as the C
boundary once did.

Handles must outlive the calls that use them, which ownership already enforces.
A pipeline or block session serializes its own calls and a panic poisons it, so
the next call reports that rather than running on unknown state. Generation
returns contiguous RGB8 in HWC order. Progress is a per-call closure rather than
a registered callback; returning false abandons the image. It runs on the calling
thread while the pipeline's lock is held, so it must not call back into the same
pipeline.

## Integration checks, 2026-09-09

Workspace CPU tests, clippy with warnings denied, and rustdoc passed. The 21
selected native tests cover operation-cache lifetime, compiler targets, uploads
crossing a 16 MiB boundary, gathered and padded weights, pooled storage, and
numerical operations. Repeat them with:

```sh
HRX_OFFLINE=1 cargo test --test dispatch -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo test --test uploads -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo test -- --ignored --test-threads=1
HRX_OFFLINE=1 cargo run --release --example dispatch_cost
```

On Ryzen AI MAX+ 395 / gfx1151 with Rust 1.95 nightly and hrx-rs 0.1.0, three
release runs measured 127–137 ns of host time per direct dispatch and 227–243 ns
including prepared-cache lookup, configuration construction and scalar packing.
Each run takes the median of nine 2,048-launch batches after three warmups and
checks the output. Completed batches averaged 2.1–2.25 µs per tiny kernel.
These are wall-clock costs, not GPU timestamps or whole-model latency. Image
quality, checkpoint-scale load time and peak memory were not remeasured.

The [legacy NPU text-fusion pilot](npu-fusion.md) is retired. Auto uses GPU;
a native replacement requires fresh latency and quality qualification.
