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

The parity gate uses the **original, unquantized BF16 checkpoint** as ground truth,
not another quantized implementation. Run it explicitly:

```sh
scripts/parity.sh
```

This invokes `crates/pipeline/tests/unquantized_parity.rs`. The saved reference
uses the original `krea2_turbo_bf16.safetensors` through the Diffusers pipeline at
1024×1024, seed 0, eight steps. Both implementations receive identical packed
noise, tapped text states and scheduler settings. The Rust test executes the
transformer trajectory with the production GPU Euler kernel, then decodes
its final latent with the native VAE. It checks latent error and RGB PSNR against
the unquantized reference, allowing at most the existing 0.1 dB loss from the
accepted baseline in either metric. Agreement with quantized ComfyUI is not a
release criterion.

The default fixture directory is `build/quality`; `KREA2_QUALITY_FIXTURE` can point
to another copy of the same frozen fixture. `KREA2_CHECKPOINT` selects the native
candidate's checkpoint. Missing fixtures or weights fail explicitly. Reference
file hashes are pinned in `crates/pipeline/tests/fixtures/unquantized.json`; the
runner never creates or updates its own ground truth. The test needs no Python,
NumPy or Torch.

Because it never creates its ground truth, the fixture has to come from somewhere,
and `build/quality` is not versioned. `scripts/capture_reference.py` is that
somewhere: the half of the gate that needs Torch, Diffusers and the unquantized
checkpoint, so it runs in an environment this repository does not otherwise depend
on. Without it a lost `build/` would end the gate permanently.

```sh
python3 scripts/capture_reference.py reference               # ground truth: noise, text, bf16 latent and image
python3 scripts/capture_reference.py accept --latents FILE   # promote a native run to the accepted baseline
python3 scripts/capture_reference.py manifest --write        # re-pin the hashes after either
```

`reference` refuses to overwrite an existing fixture without `--force`, because
re-minting truth from a build that has already drifted is the one mistake this gate
cannot survive. `accept` takes a `KREA2_QUALITY_OUTPUT` dump and is the only way to
move the baseline; the pinned hashes make that a deliberate, reviewable commit
rather than something a passing run can do to itself. A fresh capture is a new
fixture with new hashes, so `manifest` follows either of the other two.

The fixture contains `job.json`, `noise.npy` (`[4096,64]`), `text.npy`
(`[19,12,2560]`), `bf16.npy` and accepted `w8a8.npy` (`[1,4096,64]`), plus
`bf16.png` and accepted `w8a8.png`. NPY tensors are little-endian float32 in C order;
the two latent NPY files are lossless exports of the original saved PT tensors.
New prompts, seeds or resolutions require independently captured unquantized
outputs and explicit fixture review. This single fixture is not a quality sweep
and uses saved conditioning rather than testing the text encoder.

The original BF16 timestep embedding and Euler arithmetic are retained. Kernel
rounding changes are accepted only if their error against the original model
does not increase. The quantized ComfyUI parity scripts have been removed.

The verified 1024×1024 fixture improves as follows:

| Against the unquantized BF16 reference | Previous native implementation | Updated kernels |
| --- | ---: | ---: |
| Final-latent cosine | 0.998254 | 0.999717 |
| Relative latent RMS error | 0.059111 | 0.023806 |
| Image PSNR | 33.6673 dB | 36.9982 dB |

These measurements include the actual GPU Euler kernel and native VAE decoder.
