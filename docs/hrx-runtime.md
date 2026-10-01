# Shared HRX runtime

Krea uses [`hrx-rs` 0.8.15](https://crates.io/crates/hrx-rs) or a compatible 0.8 release for keyed kernel
requests and the coordinated graph API.
The manifest renames it to `hrx`, so call sites read `hrx::`.
The GPU runtime and Loom come from the unified HRX 0.8 native bundle.
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

`Pipeline::open_with_adapter_in(files, &context, compiler, &adapter, strength)`
uses the same context for LoRA inference, including strength zero. Standalone
`open` and `open_with_adapter` create a private context and delegate to these
entry points. An application's root `[patch.crates-io]` applies to Krea's HRX
dependency too; keep every model on the same resolved HRX package so their
`ModelContext` types and native runtime agree.

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

Training provides `Trainer::open_in(config, &context)` and
`Trainer::resume_in(checkpoint, &context)`. Preparation provides
`training::prepare::prepare_in(&config, &context)` for both its VAE and text
encoder passes. Existing standalone entry points create a private context.
All native streams select the supplied context's GPU; these entry points do not
move native training operations into the graph scheduler. The embedding
application still controls job concurrency.

The context's budget is optional: `RuntimeOptions::default()` has none. Configure
one explicitly when the application needs a global ceiling:

```rust,ignore
let residency = hrx::residency::ResidencyManager::new(global_bytes)?;
let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
    memory_budget: Some(residency.budget()),
    ..Default::default()
})?;
let trainer = krea2::training::Trainer::open_in(config, &context)?;
```

HRX 0.8 supports one allocation counter per native stream. Training preserves
its hard `memory_gib` ceiling by reserving that **entire allowance** in the
context's global budget before loading, then enforcing the allowance on its
private stream. The global counter reports reserved capacity, not instantaneous
training usage. Admission can fail even when the estimated model would fit if
the requested allowance plus other live reservations exceeds the global ceiling.
Choose `memory_gib` accordingly; no code silently shrinks the allowance or enables
offloading. Without a context budget, the private ceiling still applies.

Each uncached preparation pass reserves and releases its allowance separately.
A fully cached preparation needs no native stream or allowance. On normal drop
or a drained load failure, all weights, scratch and stream staging are released
before the global allowance. If HRX quarantines allocations after a device
failure, the global allowance remains reserved rather than advertising that
memory as reusable. Inference streams charge their allocations directly.

The library retains training's 8 GiB system RAM floor and startup headroom checks.
It never changes `oom_score_adj`; that process-wide policy remains exclusively
in the standalone `krea2-train` binary.

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
cargo install --locked hrx-rs --version 0.8.9
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
grid and report setting as their key. The key holds the embedded kernel's
`'static` name and configuration names, so a hit allocates nothing and takes no
lock beyond HRX's own. Only a miss constructs a specialization and hashes the
embedded source. HRX owns the key index and the artifact cache;
different keys for the same artifact share one loaded executable. Failed
requests remain retryable, and cache hits still check the device.

Both auxiliary and block compilation use the stream's target and `hrx::loom`.
Compiler selection is cached by library and target. Its bounded module cache is
kept across guidance shapes, avoiding repeated parsing and indexing. Prepared
operation sets retain loaded kernels for their lifetime; there is no global
loaded-kernel cache holding model resources after teardown. Artifacts load from
owned bytes without a compiler subprocess or an extra filesystem round trip.

Kernel sources live in `kernels`; tokenizer assets live in `assets`. `build.rs`
embeds every kernel and tests read the same paths, so package builds carry the
assets without depending on files outside the package.

Applications combining model crates should resolve to one HRX package version
and source, sharing a `ModelContext` when they need coordinated ownership.

## Interface

The `krea2` library is the interface. Consuming applications use its
`krea2::pipeline` and `krea2::session` modules directly; an Elixir application
wraps `krea2::pipeline::Pipeline` with Rustler. There is no C
ABI, no generated header and no error-buffer protocol: arguments are Rust types,
failures are `Result`, and Rustler contains panics at the NIF boundary as the C
boundary once did.

Every function returns `krea2::Error`, re-exported as each module's `Error`.
`Error::is_invalid_argument` separates a request the caller can correct from a
failure; `Error::Cancelled` reports a progress callback that asked to stop and
`Error::Poisoned` an object an earlier panic left unusable. Runtime failures
keep their `hrx::Error`, so a lost device or a busy slot stays distinguishable
all the way to the caller.

Handles must outlive the calls that use them, which ownership already enforces.
A pipeline or block session serializes its own calls and a panic poisons it, so
the next call reports that rather than running on unknown state. Generation
returns contiguous RGB8 in HWC order. Progress is a per-call closure rather than
a registered callback; returning false abandons the image. It runs on the calling
thread while the pipeline's lock is held, so it must not call back into the same
pipeline.
