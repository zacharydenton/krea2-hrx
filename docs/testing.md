# Native test coverage

Run `scripts/test.sh --cpu` for formatting, Clippy and workspace tests;
`scripts/test.sh --gpu` also runs the explicitly ignored gfx1151 tests. GPU tests
use the same HRX compiler/cache and runtime as inference. Missing prerequisites
are errors when GPU tests are requested. There is no Python environment.

| Coverage | Rust location |
| --- | --- |
| BF16 ties, NaNs and conversion boundaries | `crates/numerics` |
| All 36,050 Turbo/Raw sigma values from Diffusers, including the 589-token rounding regression | `crates/pipeline/tests/schedule.rs` and immutable `fixtures/sigmas.f32le` |
| Exact BF16 Euler and guidance arithmetic; activations, broadcast, normalization and fused SiLU | `crates/ops/tests/arithmetic.rs` |
| Dense matmul, bias, ragged tiles, convolution layouts, grouped/causal attention, rotary embedding and upsampling | `crates/ops/tests/arithmetic.rs` |
| Repeated resident softmax, including causal masking and non-tile-aligned lengths | `crates/ops/tests/softmax_repeat.rs` |
| INT4/INT8 GEMM tiles, padded operand pitch, zero scales, BF16 residual and SwiGLU ordering | `crates/loom/tests/quantized.rs` |
| Production FP16 attention and quantized preparation against independent softmax and Hadamard references | `crates/loom/tests/quantized.rs` |
| Shared compiled artifact identity, allocation bounds and native library loading | `crates/loom/tests/dispatch.rs` |
| Invalid checkpoint/metadata rejected before GPU startup | `crates/session/tests/constructor.rs` |
| C ABI null handles, output clearing, bounded error buffers and panic containment | `crates/abi` |

Checked-in Loom source is authoritative. The former Python generators, model
wrappers, benchmark orchestration and reference implementations are retired.
Historical measurements remain in the documentation and old tools remain in Git.
The GPU oracles compute independent CPU arithmetic; they do not reproduce kernel
implementations line for line. The scheduler fixture is captured from Diffusers,
not generated from the Rust scheduler.

Full-checkpoint comparisons against ComfyUI/Diffusers and image-trajectory quality
runs formerly driven by Python are not equivalent to these operation tests and
are not claimed as migrated. Existing historical image and latency results are
unchanged by this test migration.

## Parity against ComfyUI

Everything above checks that the model agrees with its own earlier output.
`scripts/parity.py` is the only check against something outside itself, so it is
worth keeping even though it cannot run unattended: it needs the checkpoint, a
GPU, and a directory of dumps that `scripts/comfy_dump.py` produces inside the
ComfyUI environment. It is deliberately not part of `scripts/test.sh`; run it
before a release.

```sh
/opt/venv/bin/python scripts/comfy_dump.py --dump-steps --out build/comfy_parity  # in ComfyUI
python3 scripts/parity.py gate --require    # the release gate
python3 scripts/parity.py steps             # every evaluation, to localize a failure
python3 scripts/parity.py image             # our run and ComfyUI's latent, both to PNG
```

The gate asserts the final latent from a full run on ComfyUI's noise. It does not
pass today: the last recorded measurement puts that latent at cosine 0.806 while
the per-evaluation velocity agrees at 0.9981–0.9999, because the Euler step at the
low sigmas subtracts two large terms and amplifies a 1.4% velocity difference into
a 55% latent one. The velocity is reported beside the gate to localize a failure,
never in place of it — the threshold stays on the number that says whether this
host produces ComfyUI's image.
