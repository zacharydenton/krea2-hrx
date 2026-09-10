# Architecture and numerical contracts

## Pipeline

Krea 2 uses a 28-block transformer over one sequence of text and image tokens.
The hidden width is 6144, with 48 query heads, 12 key/value heads, head dimension
128 and SwiGLU width 16384. At 1024×1024 there are 4096 image tokens plus text.
Qwen3-VL-4B encodes the prompt; Qwen-Image VAE decodes the resulting latents.

| Crate | Responsibility |
| --- | --- |
| `hrx` | Device, allocations, argument packing and dispatch |
| `krea2-kernels` | Embedded kernel sources, the specializations they compile to and the launch shapes |
| `krea2-checkpoint` | Safetensors mapping and block-weight upload plans |
| `krea2-numerics` | bf16 and fp8 conversions |
| `krea2-tokenizer` | Hugging Face tokenization and Krea's prompt template |
| `krea2-ops` | Device tensors, allocation pool and auxiliary operations |
| `krea2-session` | Resident transformer blocks and attention preparation |
| `krea2-models` | Model discovery, text encoder, outer transformer graph and VAE |
| `krea2-pipeline` | Conditioning, noise, scheduling and image generation |

`cli/` provides argument handling and image output. `crates/kernels/kernels/`
contains the checked-in Loom sources, which are authoritative since their
generators were retired. `crates/tokenizer/assets/tokenizer.json` is embedded
and read by the Rust `tokenizers` crate.

## Transformer

Each block performs normalization/modulation, group-256 Hadamard rotation,
per-token quantization, fused QKV/gate projection, QK normalization and RoPE,
attention, output projection, and the SwiGLU MLP. Gated residual updates are
fused into the output and down-projection GEMM epilogues.

The published int8 ConvRot checkpoints run with int8 weights and activations
(W8A8). Loading concatenates Q/K/V/gate rows and interleaves MLP gate/up rows in
16-row groups. Weights keep their checkpoint quantization and float32 norm
scales; the runtime does not quantize a bf16 checkpoint into W8A8 or W4A4.
The int4 kernel family remains available for experiments.

Operand rows with an 8192-byte pitch receive padding to avoid cache aliasing.
The upload plan, kernel configuration and launch metadata must agree on the
pitch. Shape rules live in `crates/kernels/src/shape.rs`, pinned by its own tests. The
Python kernel builder that once mirrored them, and the parity test between the
two, were retired with that layer.

## Precision and sampling

The residual stream and auxiliary activations are bf16. Preserve rounding at
modulation, residual addition, guidance and Euler updates: changing these
boundaries can change the full denoising trajectory even when block cosine is
high. Norm scales stored as float32 must not be rounded to bf16 before use.

Turbo defaults to eight unguided steps with a fixed timestep shift. Raw defaults
to 52 steps and guidance 3.5, with a resolution-dependent shift. Guidance follows
Krea's `cond + g * (cond - uncond)` convention. Native noise uses ChaCha8 and a
standard normal distribution; a seed does not reproduce Torch's initial noise.
Use supplied initial latents for backend comparisons.

## VAE

The single-image decoder takes the last temporal tap of causal 3D convolution
weights. Dense 3×3 weights are packed as `[out, ky, kx, in]` for implicit GEMM;
the logical shape remains `[out, in, ky, kx]`. `Weight::layout` determines the
convolution path. Both the float32 copy and the bf16 copy must use that layout.
Row-major weights use im2col, and 1×1 convolutions use GEMM directly.

Decode uses 32×32 latent tiles at stride 24 when tiling is needed. Overlap is
blended in bf16 before conversion to RGB8. Tile geometry affects normalization,
attention context and seams; increasing tile size is a numerical change as well
as a memory/performance change.

## Validation lessons

- **Full trajectories matter.** Query32 attention passed block checks but lost
  7.39 dB of image PSNR on the accepted fixture. It remains experimental.
- **Exercise the loader.** Kernel tests with manually packed weights cannot
  detect missing or inconsistent repacking in checkpoint loading.
- **Repeat resident calls.** A softmax race appeared only after repeated VAE
  decodes. Separate LDS regions now hold maximum and sum reductions.
- **Carry layout and ownership.** Tensor views retain allocations; convolution
  layout belongs to the weight; host and device session APIs share RoPE upload
  bookkeeping.

See [attention](native-attention.md), [VAE measurements](vae-performance.md),
[graph recording](graph-recording.md) and
[contribution requirements](../CONTRIBUTING.md). Historical raw measurements
remain under `docs/benchmarks/`; the chronological development log is in Git history.
