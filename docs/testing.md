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

Quantization is the dominant term, and the cleanest way to see it is a single
transformer forward rather than a trajectory. Given identical latents, identical
compacted text and the same device, our W8A8 transformer and the unquantized
diffusers one agree at **cosine 0.999885, relative RMS 0.015220**. Nothing
accumulates in that measurement and nothing else varies.

Eight Euler steps turn that 1.5% per-forward difference into the 0.0769 the gate
reports. The amplification is expected: at the low sigmas the step subtracts two
large terms, so a small velocity difference becomes a large latent one. The
unquantized model moved to the GPU stays within 0.0144 of the CPU reference over
the same eight steps, because its per-step difference is much smaller and
compounds far less.

Two other terms exist and are worth knowing, but neither explains the gate:

| Change, everything else held fixed | rel RMS |
| --- | ---: |
| Per transformer forward: our W8A8 vs unquantized | 0.015220 |
| Text sequence layout: 19 compacted rows vs 512 padded, same device and values | 0.035054 |
| Transformer device, CPU to GPU, at the compacted layout | 0.033368 |
| Unquantized on GPU vs the CPU reference, eight steps | 0.014415 |

The layout row deserves attention on its own account. The reference samples from
512 padded text rows and a mask while the native transformer consumes the 19 valid
rows, and `prepare_position_ids` takes `text_seq_len`. Measured separately the
layout and device terms are each about 0.034, yet combined they leave the
trajectory only 0.0144 from the reference, so they substantially cancel. That is
worth understanding before either number is quoted alone.

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
./scripts/capture_reference.py manifest --reference-only --write  # pin the new reference first
KREA2_QUALITY_MINT=1 scripts/parity.sh                 # mint the accepted baseline for a new fixture
./scripts/capture_reference.py manifest --write        # pin the complete fixture, including the baseline
scripts/parity.sh                                    # run the ordinary regression gate
```

A freshly captured reference has no accepted baseline, and one cannot exist until
the native trajectory has been run, so the gate mints it — but only when
`KREA2_QUALITY_MINT=1` is set, both baseline files are absent, and the five reference
files match a reference-only manifest. The ordinary gate requires all seven pinned
files. Minting creates a baseline; it does not check for a regression. For separate
fixtures, use `--work DIR` and `--manifest FILE` on the capture commands and
`KREA2_QUALITY_FIXTURE=DIR KREA2_QUALITY_MANIFEST=FILE` on the Rust runner.
`--checkpoint` remains as a labelled fallback that maps a
local ComfyUI-format file onto the same official modules; it is recorded in
`job.json` as such. Captured both ways on the same machine, the two agree
bit-for-bit, so the mapping is exact -- but only the `from_pretrained` path is
free of this repository's own interpretation of the weights.

It is a `uv run` script: the dependencies, the pinned interpreter and the ROCm
Torch index live in its own header, and `scripts/capture_reference.py.lock` fixes
the resolution, so there is no environment to create and nothing to activate.
The official model is pinned to commit
`98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7`; new captures record that revision in
`job.json`. `--revision` accepts a full commit SHA and is required for custom
repositories. Existing frozen fixtures retain their original provenance.
Torch comes from the ROCm index rather than PyPI, whose default wheels are CUDA;
that index publishes cp313 wheels only, which is why the script pins Python below
3.14 and lets uv fetch a matching interpreter.

`reference` refuses to overwrite an existing fixture without `--force`, because
re-minting truth from a build that has already drifted is the one mistake this gate
cannot survive. Capture stages all new reference files before replacing the old
ones and removes both old baseline files on successful recapture. A failed capture
leaves the existing fixture intact; interrupted publication fails hash validation.

To deliberately promote a candidate, run
`KREA2_QUALITY_OUTPUT=/tmp/candidate.f32 scripts/parity.sh`, then
`./scripts/capture_reference.py accept --latents /tmp/candidate.f32` and
`./scripts/capture_reference.py manifest --write`. The runner exports the raw
latent, its native decoded image (`.f32.png`), and a receipt (`.f32.json`) binding
both to the verified reference hashes. `accept` validates the pair and copies the
native image without invoking Torch or another decoder. Legacy latent-only dumps
must be exported again. Review the quality measurements before promoting a run;
exports are available even when the regression assertions fail.

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

## Separating the scheme from our implementation of it

`scripts/parity.sh` has two points to compare, so it can say how far this host is
from the unquantized model but not how much of that is the W8A8 ConvRot scheme and
how much is our kernels. `scripts/quantized_reference.py` supplies the third point:
the same int8 ConvRot checkpoint run by a plain torch transcription of
`Krea2Transformer2DModel`, with no kernel of ours involved. Like the capture script
it is a `uv run` script and not part of `scripts/test.sh`.

```sh
./scripts/quantized_reference.py forward --checkpoint CKPT --quant w8a8 --out v.npy
```

Prefer `forward` over `trajectory`: one evaluation accumulates nothing, so the number
is the arithmetic rather than eight steps of amplification. Measured that way at
1024×1024 on the pinned fixture, everything on the GPU, identical inputs:

| One transformer forward | rel RMS |
| --- | ---: |
| The scheme: torch W8A8 against the identical torch code unquantized | 0.012905 |
| Our kernels against that scheme | 0.011656 |
| Our kernels against unquantized | 0.015702 |
| Two independent *unquantized* implementations (this transcription vs diffusers) | 0.007002 |

The last row is the floor: two honest implementations of the same maths already
disagree by 0.0070, so our 0.0117 deviation from the scheme is under twice the noise
between implementations. Decoded through one untiled VAE the scheme reaches 30.93 dB
against the reference and our kernels 31.13 dB — marginally closer than the scheme's
own reference implementation.

SSIM disagrees with PSNR on that last point: 0.9705 for the torch scheme against
0.9459 for our kernels. Our kernels are structurally a little further from the
reference while being no further in mean-square terms, which is the shape the fp16
WMMA attention default would produce. Neither metric should be quoted alone.

The script is slow by construction — dequantisation and the Hadamard rotations run in
eager torch, about two minutes per forward — and it needs the ComfyUI-format files
rather than the diffusers repository, because `w8a8` reads the packed rows and
`weight_scale` tensors that the diffusers export does not carry.
