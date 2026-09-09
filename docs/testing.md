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

## Parity against the unquantized model

The parity gate uses the **original, unquantized BF16 checkpoint** as ground truth,
not another quantized implementation. Run it explicitly:

```sh
scripts/parity.sh
```

This invokes `crates/pipeline/tests/unquantized_parity.rs`. The reference is the
official `krea/Krea-2-Turbo` diffusers repository loaded with `from_pretrained`,
run **on the CPU** at 1024×1024, seed 0, eight steps, guidance 0. Nothing in this
repository interprets the checkpoint on that path: the pipeline, transformer,
scheduler and VAE are all stock diffusers, on the published weights. Both
implementations receive identical packed noise, tapped text states and scheduler
settings. The Rust test executes the transformer trajectory with the production
GPU Euler kernel, then decodes its final latent with the native VAE. It checks
latent error and RGB PSNR, allowing at most 0.1 dB loss from the accepted
baseline in either metric. Agreement with quantized ComfyUI is not a release
criterion.

The reference runs on the CPU because that is what makes it reproducible by
someone who does not have this GPU. It is not about nondeterminism: two GPU
captures taken back to back are bit-identical, as are two CPU captures. The noise
is always drawn on the CPU whatever the model runs on, because a CUDA generator
and a CPU generator produce unrelated streams from one seed (measured at cosine
0.002), which would otherwise make CPU and GPU captures incomparable rather than
differently rounded.

Against that reference the current kernels measure:

| Against the official CPU reference | value |
| --- | ---: |
| Final-latent cosine | 0.997044 |
| Relative latent RMS error | 0.076912 |
| Image PSNR | 31.7778 dB |

That number is not yet decomposed, and two attempts to decompose it were wrong,
so what follows is only what has been measured cleanly.

| Measured | rel RMS |
| --- | ---: |
| Everything on GPU vs everything on CPU (diffusers, unquantized, same noise) | 0.074973 |
| Text sequence layout alone: 19 compacted rows vs 512 padded rows, same device, same values | 0.035054 |
| Our own text encoder instead of the reference's, through our transformer | 0.016092 |
| Our kernels with the reference's text (what the gate reports) | 0.076472 |

The layout row is the one to be careful about. The reference computes its
trajectory from 512 padded text rows and a mask; the fixture stores only the 19
valid rows, and the native transformer consumes those. `prepare_position_ids`
takes `text_seq_len`, so the two layouts give the image tokens different rotary
coordinates -- it is a different computation, worth 0.035 on its own, and some of
the gate's 0.0769 is that rather than quantization.

`scripts/capture_reference.py --reuse FIXTURE` reuses a fixture's noise and
conditioning, which is how the layout term above was measured, but it feeds the
compacted rows and so is not a drop-in control for a padded-layout reference.
Attributing the gate's error properly needs a reference captured at the same text
layout the native path uses; until then, treat 0.0769 as a single number.

The earlier figures of cosine 0.999717 and 37.00 dB were measured against a
GPU-captured reference and are not comparable with these: that reference shared
device arithmetic *and* its conditioning with the candidate. The kernels did not
regress; the measuring stick moved onto firmer ground.

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
./scripts/capture_reference.py reference               # ground truth: noise, text, bf16 latent and image
KREA2_QUALITY_MINT=1 scripts/parity.sh                 # mint the accepted baseline for a new fixture
./scripts/capture_reference.py manifest --write        # re-pin the hashes after either
```

A freshly captured reference has no accepted baseline, and one cannot exist until
the native trajectory has been run, so the gate mints it — but only when
`KREA2_QUALITY_MINT` is set. A missing baseline is otherwise a hard failure, never
a quietly passing gate. `--checkpoint` remains as a labelled fallback that maps a
local ComfyUI-format file onto the same official modules; it is recorded in
`job.json` as such. Captured both ways on the same machine, the two agree
bit-for-bit, so the mapping is exact -- but only the `from_pretrained` path is
free of this repository's own interpretation of the weights.

It is a `uv run` script: the dependencies, the pinned interpreter and the ROCm
Torch index live in its own header, and `scripts/capture_reference.py.lock` fixes
the resolution, so there is no environment to create and nothing to activate.
Torch comes from the ROCm index rather than PyPI, whose default wheels are CUDA;
that index publishes cp313 wheels only, which is why the script pins Python below
3.14 and lets uv fetch a matching interpreter.

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

The kernel improvement that motivated this section was measured against the
earlier GPU-captured reference, and is recorded here as history rather than as a
current figure. On that fixture the update moved final-latent cosine from 0.998254
to 0.999717, relative latent RMS from 0.059111 to 0.023806, and image PSNR from
33.6673 dB to 36.9982 dB. Those numbers are not comparable with the ones above:
the reference has since moved to the official repository on the CPU, and the
GPU-captured reference shared its device arithmetic with the candidate. Both sets
include the actual GPU Euler kernel and native VAE decoder.
