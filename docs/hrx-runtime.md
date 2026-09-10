# Shared HRX runtime

Krea uses [`hrx-rs`](https://crates.io/crates/hrx-rs) 0.2.0 from crates.io.
The manifest renames it to `hrx`, so call sites read `hrx::`.
Its features are explicit and `Cargo.lock` pins the complete dependency
graph. The shared implementation owns native loading, status conversion,
allocation, streams, dispatch lifetimes and compiler caching.
There is no link-time libhrx dependency or runtime rpath in the binary.

Each pipeline owns an ordered stream. Block sessions and their tensor pools use
that same stream; a standalone session uses its caller's stream. Dispatches,
fills and copies borrow the stream; transfers and waits need it mutably. Calls sharing
one pipeline are serialized. Native command buffers retain recorded resources,
so releasing an ordinary buffer does not wait for GPU execution.

Transfers and dispatch use HRX `View` regions. Uploads copy into HRX-owned staging
and queue work on the stream. Weight loading uses chunks of at most 16 MiB;
unpadded checkpoint rows go directly from the mapping to HRX staging, while
padded rows are assembled in a reusable host buffer. Blocking reads wait for
completion. Profiling and progress callbacks add explicit waits when needed.

Tensor storage returns to the model's pool when its final owner is dropped.
This pool is intentional: HRX's `Stream::recycle` needs mutable stream access,
which tensor destructors do not have. One pool serves one stream, and refuses
another. Buffers are device-scoped, so HRX permits a block on any stream and
`record_event`/`wait_event` order that use — but a pooled block returns to the
free list while the work reading it is still queued, and reissuing it is safe
only because the stream that runs that work also runs whatever writes it next.
Across streams there is no such order, and no event can impose one on a reuse
already handed out.

The compiler and runtime come from HRX's public, verified native bundle:

```sh
cargo install --locked hrx-rs --version 0.2.0 --features runner
hrx prepare
cargo build --release
```

For local HRX development, use an ignored `.cargo/config.toml`:

```toml
[patch.crates-io]
hrx-rs = { path = "../hrx.rs" }
```

Cargo updates the lockfile for a path override. Restore the registry dependency
before committing a lockfile generated with a local override.

`HRX_RUNTIME_DIR` selects a trusted native directory and `HRX_OFFLINE=1` refuses
network provisioning. The artifact cache follows XDG: `$XDG_CACHE_HOME/hrx`,
else `$HOME/.cache/hrx`. `HRX_LOOM_LIBRARY` or an explicit model compiler argument
selects a compiler override.

Krea's shape/bundle metadata and model-specific operation builders remain here.
Both auxiliary and block compilation use the stream's target and `hrx::loom`.
Compiler selection is cached by library and target. Its bounded module cache is
kept across guidance shapes, avoiding repeated parsing and indexing. Prepared
operation sets retain loaded kernels for their lifetime; there is no global
loaded-kernel cache holding model resources after teardown. Artifacts load from
owned bytes without a compiler subprocess or an extra filesystem round trip.

Kernel sources live in `kernels`; tokenizer assets live in
`assets`. Generators and tests use these paths directly.
Package builds carry the same assets without depending on files outside the package.

H3, Krea and kernel test libraries can coexist: HRX initializes under an OS lock
across Rust crate copies, uses the same native library, and never unloads or
globally shuts it down on model teardown.

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
