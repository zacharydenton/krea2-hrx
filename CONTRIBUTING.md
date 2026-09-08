# Contributing

Include the problem, the change, and the validation in a pull request. For bugs,
include the revision, GPU, runtime/compiler versions, model names, command and
error output. A small reproduction is preferable to a full model run.

## Build

Follow the [runtime setup](docs/hrx-runtime.md#checkout-build), then:

```sh
source scripts/env.sh
scripts/build.sh
```

For subsequent Rust changes, `cargo build --workspace` is sufficient. The
packaging script copies release binaries and test runners into `build/`.

## Tests

```sh
source scripts/env.sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

The GPU/reference suite additionally needs a `.venv` with ROCm PyTorch,
Diffusers with Krea 2 support, Transformers, NumPy, safetensors and Pillow.
The HRX build must provide `loom-format` and `loom-compile`.

| Command | Coverage and additional inputs |
| --- | --- |
| `scripts/test.sh --quick` | Generated-source parity, Rust/Python runtime tests, kernel oracles and scheduler tests |
| `scripts/test.sh` | Adds block-stack comparisons using the checkpoint and `build/fixture_step0.pt` |
| `scripts/test.sh --quick --native` | Adds model, tokenizer, transformer and VAE comparisons; requires model files |
| `scripts/test.sh --quick --quality` | Adds an eight-step trajectory comparison against an accepted quality baseline |

Some Rust tests return early when the GPU, compiler or model fixtures are
unavailable. State which hardware-dependent tests actually ran; a passing
`cargo test` alone does not establish kernel or image correctness.

## Kernels and numerical changes

Edit the relevant `tools/gen_*.py` generator and regenerate its checked-in
`.loom` files. Do not hand-edit generated kernels. The quick suite regenerates
sources in a temporary directory and compares them with the checkout.

Preserve buffer bounds, synchronization, weight layouts and rounding contracts.
Test irregular shapes and repeated calls, as well as the main production shape.
Changes to loading must exercise the production loader with representative
dtypes; a manually packed kernel fixture cannot validate it.

Attention changes require the full trajectory gate. Set
`KREA2_QUALITY_BASELINE` to an archived accepted run, or retain a new report with:

```sh
.venv/bin/python tools/quality_vs_bf16.py regression \
  --baseline /path/to/accepted-run --work build/quality-candidate
```

The gate permits at most 0.1 dB loss in latent and image PSNR on identical
reference inputs. It covers a fixed fixture, not general image quality.

## Benchmarks

Use an idle GPU, warm caches and identical inputs. Alternate baseline and
candidate runs; report all samples, revisions, compiler/runtime versions and
timer boundaries. Exclude loading and compilation from warm inference results.
Verify repeated outputs and compare against an independent numerical reference.
Require exact equality when arithmetic is unchanged; explain and quantify any
rounding differences when it changes.

Keep benchmark records in `docs/benchmarks/`. Document current behavior and
reproducible results rather than a chronological debugging log. Comments should
explain contracts, safety, layouts and non-obvious decisions; preserve `# Safety`
sections on unsafe APIs.
