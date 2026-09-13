# Contributing to krea2-hrx

The model host, tests, and tooling are Rust. Checked-in `.loom` files are the
source of truth: edit them directly and test the affected operation against an
independent CPU reference. Reference-capture Python scripts are separate, opt-in tools; normal builds and
tests do not need Python.

```sh
scripts/test.sh --cpu
scripts/test.sh --gpu
```

The CPU suite runs formatting, Clippy, rustdoc and CPU tests. GPU tests are explicitly
ignored by default and must be requested on gfx1151; once requested, missing
hardware, compiler, or runtime is a failure, never a silent pass. HRX provisions
and caches the compiler and runtime. `HRX_OFFLINE=1` requires an existing bundle.

Use the shared HRX crate for native loading, allocation, scalar packing, dispatch,
compilation and caching. Model code owns its source selection, shapes, weight
layout and numerical semantics. The `krea2` library is the public interface:
consuming applications depend on it directly, and a Rustler adapter lives in
the application rather than behind a C boundary here.

See [test coverage](docs/testing.md) for the numerical cases and remaining limits.
Record dimensions, precision, toolchain and GPU when reporting performance.

Open an issue for bugs or proposed features, and include a minimal reproduction.
For inference failures, include the commit, GPU, Linux/kernel and Rust versions,
model filenames, command, and relevant error output. Remove credentials and
private prompts from logs before sharing them.

Keep pull requests focused and explain the behavior change and validation.
Run `scripts/test.sh --cpu` before submitting; kernel changes also need the
relevant GPU checks. State any hardware or model-dependent checks you could not
run. Maintainers can use the [release checklist](docs/releasing.md).
