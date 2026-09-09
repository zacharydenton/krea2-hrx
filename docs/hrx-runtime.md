# Shared HRX runtime

Krea depends on the published [`hrx-rs`](https://crates.io/crates/hrx-rs) crate
directly, renamed to `hrx` in the workspace `Cargo.toml` so call sites read
`hrx::`. Its features are explicit and `Cargo.lock` pins the complete dependency
graph. The shared implementation owns native loading, status conversion,
allocation, streams, dispatch lifetimes and compiler caching.
There is no link-time libhrx dependency or runtime rpath in the binary.

A pipeline selects its own ordered stream. Nested block sessions inherit it;
standalone block sessions open a stream of their own. Calls sharing one pipeline
or block session are serialized and a panic poisons that session. Tensor pools
are bound to the stream that first uses them. Direct kernel arguments retain
referenced allocations through synchronization, so buffer Drop does not wait
for GPU execution. Allocations use the native device allocator without draining
pending commands. Native operations already insert their own ordering barriers.

The default runtime and in-process compiler come from HRX's pinned native bundle.
The HRX repository is currently private, so download with an authenticated GitHub
CLI and prepare the verified cache:

```sh
cargo install --locked --git https://github.com/zacharydenton/hrx.rs --rev be89b44652af6adf17c5c950d0759f92c2e88582 --features runner hrx
mkdir -p build
gh release download native-ecaaf7376f7d-loomc --repo zacharydenton/hrx.rs \
  --pattern hrx-linux-x86_64-gfx1151.tar.gz --dir build
hrx prepare build/hrx-linux-x86_64-gfx1151.tar.gz
cargo build --release --workspace
cargo install --locked --path cli
```

To develop Krea against a working copy of `hrx.rs`, override the pinned
revision without editing the manifest. `.cargo/config.toml` is ignored by git:

```toml
[patch."https://github.com/zacharydenton/hrx.rs"]
hrx = { path = "../hrx.rs" }
```

The override rewrites `Cargo.lock` to the path source, so remove the file and
`cargo build` again — or `git checkout Cargo.lock` — before committing.

`HRX_RUNTIME_DIR` chooses a trusted native directory (`KREA2_RUNTIME` remains an
alias). `HRX_CACHE_DIR` controls the shared cache. `HRX_OFFLINE=1` refuses network
provisioning. `HRX_LOOM_LIBRARY` or an explicit model compiler argument overrides
the compiler. Normal builds do not require `scripts/runtime.sh`; that script
remains available to stage a developer's native build for an override.
The shared crate README documents provisioning, native lifetime and bundle format.

Krea's shape/bundle metadata and model-specific operation builders remain here.
Both auxiliary and block compilation delegate to `hrx::loom`. Cache identity now
includes the compiler's content hash as well as source, symbol, target and
configuration. Sessions receive typed prepared artifacts and load their owned
executable bytes directly. There is no precompiled-directory interface or
compiler subprocess.

Kernel sources live in `crates/kernels/kernels`; tokenizer assets live in
`crates/tokenizer/assets`. Generators and tests use these paths directly.
Package builds carry the same assets without depending on files outside the package.

H3, Krea and kernel test libraries can coexist: HRX initializes under an OS lock
across Rust crate copies, uses the same native library, and never unloads or
globally shuts it down on model teardown.

## Interface

The workspace crates are the interface. `krea2-pipeline` and `krea2-session` are
ordinary Rust libraries and consuming applications depend on them directly; an
Elixir application wraps `krea2_pipeline::Pipeline` with Rustler. There is no C
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
