# Contributing to krea2-hrx

The model host, tests, and tooling are Rust. Checked-in `.loom` files are the
source of truth: edit them directly and test the affected operation against an
independent CPU reference. Reference-capture Python scripts are separate, opt-in tools; normal builds and
tests do not need Python.

```sh
scripts/test.sh --cpu
scripts/test.sh --gpu
```

The CPU suite runs formatting, Clippy, rustdoc, the pinned shipped-dependency check and CPU tests. CI also builds on the
`rust-version` in `Cargo.toml` and checks RustSec advisories. Every `unsafe`
block needs a `// SAFETY:` comment; Clippy enforces it. GPU tests are explicitly
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

## Benchmarks

The repository contains runnable Criterion benches only. Keep benchmark results,
profiler reports and experimental harnesses outside the checkout. Criterion's
local output defaults to ignored `target/criterion`; `CRITERION_HOME` can point
it outside the repository.

```sh
cargo bench --bench kernels
cargo bench --bench generation -- generation/1024x1024
# Compile every bench without running the GPU workload:
cargo bench --bench kernels --bench generation --no-run
```

Both suites require gfx1151 and the HRX runtime/compiler. Generation also needs
the Turbo INT8 ConvRot checkpoint, BF16 text encoder and VAE in the Hugging Face
cache; it never downloads weights. It uses eight steps, guidance disabled, a
fixed prompt and seed, and covers 256×256, 1024×1024 and 1536×2048. Loading,
compilation and the first generation are outside the measured interval. Each
sample includes text encoding, denoising, decoding and RGB readback. Flat
sampling limits repetition for these long-running cases.

Kernel benches cover the four transformer GEMMs at 1043, 4095, 4096, 4109,
12301 and 16397 tokens, including the wide-GEMM scheduling boundary. They also
cover both production attention tiles at 4109 and 12301 tokens. They measure completed
wall-clock dispatch latency, including submission and synchronization. Compilation,
uploads and residual resets are excluded. Attention starts with transposed V;
these isolated timings exclude preparation and cannot establish whole-image speed.
Kernel setup is retained across Criterion samples. Independent numerical
validation lives in the GPU oracle and parity tests.

Disable `KREA2_NATIVE_PROFILE` for timings and avoid concurrent GPU workloads.
Record the commit, dimensions, precision, model/runtime/compiler versions and GPU
alongside externally stored results. Run a quick executable check with
`cargo bench --bench kernels -- --test` or
`cargo bench --bench generation -- generation/256x256 --test`.

Use kernel benches during tuning; run the image-quality gate and full generation
bench only for the final candidate. Compare timings on an uncontended GPU,
and repeat the baseline after candidate batches to check for drift.

```sh
cargo bench --bench kernels -- gemm/gemm_gu/4109 --save-baseline before
KREA2_BENCH_SOURCE=/path/to/candidate.loom \
KREA2_BENCH_REPORT_DIR=/path/to/reports \
  cargo bench --bench kernels -- gemm/gemm_gu/4109 --baseline before
```

`KREA2_BENCH_SOURCE` accepts trusted local kernel code with the selected kernel's
export, configuration, buffer ABI and launch dimensions. Filter to that kernel.
The bench checks its output byte for byte against the embedded source before
measurement. `KREA2_BENCH_REPORT_DIR` saves HRX Details-mode compiler reports;
keep that directory outside the repository. These checks complement the
independent CPU oracles; matching the embedded implementation alone cannot
establish its correctness.
