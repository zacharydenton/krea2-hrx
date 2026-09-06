# krea2-loom

Krea 2 Turbo's native GPU pipeline in **Loom**, AMD's kernel language from
[ROCm/hrx-system](https://github.com/ROCm/hrx-system), for the Radeon 8060S (gfx1151),
in **W4A4 ConvRot**: int4 weights and int4 activations on the part's `iu4` WMMA, which is
the only 2x path this silicon has (measured 117 TOPS against 54 for int8 and fp16).
A sibling of [dinov3-loom](https://github.com/zacharydenton/dinov3-loom),
[scrfd-loom](https://github.com/zacharydenton/scrfd-loom) and the integer GEMM work in
`loom-gemm`.

## What runs where

There are two entry points:

- **Standalone C ABI:** `libkrea2_pipeline.so` runs prompt-to-RGB inference without
  Python, Torch or diffusers. All GPU kernels use Loom, including the text encoder,
  text fusion, embeddings, VAE, scheduler and Sage preprocessing. The host calls
  HRX's public C API directly; there is no HIP launcher or BLAS dependency.
  Tokenization, Unicode normalization and cache hashing run in C++.
- **Python reference/integration:** `tools/pipeline.py --backend loom` uses the
  same Loom blocks inside diffusers. `--quant w4a4` and the default bf16 backend
  remain available for comparison.

The standalone path supports the single-image Turbo model, without classifier-free
guidance, on Linux with a Radeon 8060S (gfx1151). Dimensions must be multiples of
16 in 64–2048; practical memory and latency depend on resolution. This is a native
runtime with a C interface, specialized for this GPU.

## Standalone build and use

Build dependencies are a C++17/C compiler, nlohmann JSON headers, HRX's public
headers/library and a compatible HSA GPU driver. The scripts use the local
`~/code/hrx-system/build-cuda` build and `~/.local/rocm-hrx` provider, and package
runtime libraries under `build/runtime`. They use ordinary C++, without `hipcc`,
HIP, hipBLAS, ICU, PCRE2 or OpenSSL. Python/Torch are used for model export and
reference tests. See [HRX runtime details](docs/hrx-runtime.md).
The model layout defaults to `~/krea2-models/{krea2_turbo_bf16.safetensors,qwen3-vl-4b,qwen-image/vae}`.

```sh
# Once, if build/weights has not already been exported:
env -u LD_LIBRARY_PATH .venv/bin/python tools/export_weights.py
# Export to a new directory; existing bundles are never overwritten.
env -u LD_LIBRARY_PATH .venv/bin/python tools/export_native.py
scripts/build_native.sh

export LOOM_COMPILE="$HOME/code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile"
env -u LD_LIBRARY_PATH build/krea2-generate \
  --bundle build/native --prompt "a red fox in the snow" \
  --width 1024 --height 1024 --steps 8 --seed 0 --out build/native-fox.ppm
```

The exporter copies all weights into a self-contained bundle (about 15 GB).
`--models`, `--blocks` and `--out` override preparation paths; `--link-blocks` creates
smaller development bundles whose weight symlinks must be dereferenced for deployment.
Export publishes the bundle only after preparation succeeds.

The first request for a new total token count invokes the native `loom-compile`
executable and caches the GPU binaries inside the bundle. Completed caches have
source/configuration fingerprints and artifact checksums. Auxiliary Loom sources
are embedded in the library and specialize on tensor shapes; their binaries live
under `$XDG_CACHE_HOME/krea2-loom/native-gfx1151-v1` (default `~/.cache`). Once all
needed shapes are cached, inference needs no compiler. The bundle may be read-only;
the auxiliary cache currently requires write access for its lock. New shapes need
a compiler and writable caches. Compiler upgrades do not automatically invalidate
caches; clear both caches when changing compiler builds.

Any language with C FFI can use [host/krea2_pipeline.h](host/krea2_pipeline.h):
create a session, call `krea2_generate` into a caller-owned RGB8 buffer, then destroy
the session. Component APIs expose tokenization, text encoding, transformer inference
and VAE decoding. Calls on one session are serialized; destruction must not race a
call. Errors return a nonzero status and a caller-owned error string.
[examples/generate.c](examples/generate.c) is compiled as C into
`build/krea2-c-example`; it generates a 256×256 eight-step image:

```sh
env -u LD_LIBRARY_PATH build/krea2-c-example build/native "a red fox" build/c-fox.ppm
```

Deploy the bundle, `libkrea2_pipeline.so`, `libkrea2.so`, the adjacent `runtime/`
directory and your executable;
no Python environment or source checkpoints are needed. RGB encoding belongs to the
caller; the example CLI writes PPM. Native seeds are repeatable within this backend
but do not reproduce Torch's RNG. Pass identical packed initial latents through the
C API when comparing backends.

## gfx1151 attention

The native runtime always uses tuned SA2-style INT4 QK with fp16 PV and fp32
softmax/accumulation. Loom preprocessing and attention run entirely on the GPU
through HRX. Quantization uses four channels per lane and wave shuffles, with no
workgroup barriers. V is transposed once per block. Below 8,192 tokens, eight waves share
K/V across two query tiles with alternating LDS slots; longer sequences use four
waves with explicit prefetch. Kernel selection and GEMM grouping are automatic.

```sh
scripts/build_native.sh
export LOOM_COMPILE="$HOME/code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile"
env -u LD_LIBRARY_PATH build/krea2-generate \
  --bundle build/native --prompt "a red fox in the snow" \
  --width 1024 --height 1024 --steps 8 --seed 0 --out build/fox.ppm
```

Export bundles with the current sources using `tools/export_native.py`. Both the
native pipeline and Python block wrapper use the same fixed attention path and
version-2 launch metadata. The supported range is 16–16,896 total tokens. The
score correction uses `4 * 48 * ceil(tokens/64) * capacity` bytes of GPU workspace.
INT4 QK changes image bytes relative to FP16 QK; measured quality and timing are
recorded in [the implementation notes](docs/native-attention.md).

## Native performance

Each native pipeline retains one device copy of block weights across resolution changes. The
initial upload reads an mmap of the weight file without a full host vector.
Text, outer-transformer and VAE weights likewise use one mapped upload and one
resident allocation per bundle, with tensor views into it. Block
modulation is assembled on the GPU from resident tables, and scheduler updates
stay on the GPU with the real pipeline’s bf16 delta/product rounding.

Sessions reuse temporary GPU buffers, retaining at most 512 MiB of unused buffers
until session destruction. Active tensors and model weights are additional memory.
Rotary-position data is cached by image geometry and text-token count. The native
pipeline keeps its residual stream on the GPU and converts between bf16 and fp16
there, removing the previous CPU conversions and full-stream transfers per step.
The following timings predate the HRX/Loom auxiliary port. The earlier
buffer/rotary-reuse pass preserved numerical operations. In its
1024×1024 eight-step comparison, two warm runs averaged 30.89 seconds before and
27.72 seconds after buffer/rotary reuse
(about 10% less latency), with identical RGB checksums. See `docs/notes.md` for the
measurement conditions.

With the HRX/Loom port, a fresh local comparison at 1024² and eight steps measured
about **18.5 s warm** versus **19.3 s** for the saved HIP build. Consolidating weight
allocations reduced HRX model loading from 6.31 s to 1.43 s without changing the
RGB checksum. A separate UI comparison using identical initial noise measured
19.36 s native generation versus 53.73 s in ComfyUI INT8 ConvRot. These are local
measurements, not a broad benchmark; model loading is excluded from generation
times. See `docs/hrx-runtime.md` for the conditions.

Auxiliary WMMA kernels now use wider memory operations, reuse operand storage for
their results, and select larger tiles for image projections and VAE convolutions.
They preserve the accumulation and rounding order. The resident-buffer A/B
benchmark and rejected INT4/prefetch experiments are documented in
[HRX runtime details](docs/hrx-runtime.md#auxiliary-gemm-optimization).

A later wide INT4 down-projection candidate remains experimental: kernel timing
improved, but an integrated warm repeat changed the image checksum. See the
[measurements and unresolved repeat-image check](docs/down-gemm-experiment.md).

For stage timings, set `KREA2_NATIVE_PROFILE=1` when running the CLI. This adds
synchronization and prints text encoding, denoising, VAE and transformer-stage
timings to stderr. Leave it unset for performance comparisons.

```sh
env -u LD_LIBRARY_PATH -u KREA2_NATIVE_PROFILE python3 tools/bench_native.py \
  --bundle build/native --size 1024 --steps 8 --runs 3
```

This standard-library-only Python benchmark calls the native C API and prints
loading time, per-image time and an RGB checksum. The first image includes block
session preparation; later images reuse the resident session. `--library` selects
a saved library for comparisons. Repeated calls must retain the same checksum or
the benchmark fails. The native runtime itself still needs no Python.

## Local comparison UI

Run `python tools/compare_web.py` and open `http://localhost:7865`. The UI also
listens on the local network. Generate a shared-prompt, shared-noise pair with
the native Loom runtime and the installed ComfyUI INT8 ConvRot setup, inspect
the images side by side or with a wipe slider, and download the original PNGs.
History, images and worker logs are saved under `build/comparison-ui/`.

This uses the same local ComfyUI container, models and dependency overlay as
[the performance comparison](docs/comfyui-performance.md). It starts the
`amd-strix-halo-comfyui` container when needed and leaves it available for
subsequent requests. Backends run sequentially in separate processes to release
model memory between them. Each generation starts fresh, so displayed times
include first-run preparation and are not warm throughput measurements.
The Python web server calls Loom's native C API; native inference still has no
Torch dependency. Precision, text encoding and sampler arithmetic differ between
the pipelines even though their initial float32 noise is identical.

## Validation and Python integration

`scripts/test.sh --quick` checks the host, Python API and kernels, including failure
cleanup, both tuned attention kernels against the quantization oracle, and the
FP16 research reference. `scripts/test.sh` also checks
the full block stack using `build/fixture_step0.pt`, exported weights and the model files.
`scripts/test.sh --native` additionally builds and compares the standalone tokenizer,
text encoder, outer transformer layers, full transformer, tiled VAE and scheduler
against the installed Python references. It requires `build/native`,
`build/native-deploy` and the models. For the focused scheduler, modulation and
weight-reuse checks without loading a Torch model, run
`source scripts/env.sh` followed by
`.venv/bin/python tests/test_native_regressions.py` after `scripts/build_native.sh`.
Tests combining Torch and HRX in one process must use the same HSA provider at
startup; `scripts/env.sh` sets that search path. Standalone C callers need no such
setting. The quick suite also checks actual loaded libraries after HRX dispatch.

After changes to the block host, `scripts/build_host.sh` rebuilds `libkrea2.so` and
the kernel test runner. The block API uses ABI 2; the full pipeline API uses ABI 1.
`Krea2Blocks` builds or reuses a fingerprinted kernel bundle automatically.
Both builders choose among 4, 3 and 2 to minimize padded GEMM tile rows. Launch metadata fixes the choice for each session.

The Python pipeline processes batches sample by sample, uses each prompt's padding
mask, recreates the resident block session when token counts change, preserves fp32
normalization weights and enables tiled VAE decoding.

## Correctness

`reference/krea2_ref.py` is diffusers' model transcribed onto the ComfyUI checkpoint
names, validated bit-for-bit against diffusers at toy size. The prepare kernels are
checked against its quantization, allowing occasional one-code differences from
fp16 preparation rounding. The tuned attention kernels match the smoothed INT4 Torch oracle above
0.9999 cosine similarity and produce identical outputs to each other.
`tests/test_blocks.py` runs the native blocks on a fixture captured from a real
denoising step against `reference/loom_ref.py` (W4A4 projections, smoothed INT4 attention, native fp16 storage and
fp32 fused operations) and the original bf16 model. Every block is checked on
identical inputs with the original 0.99 update-cosine threshold, and composing
individual native blocks must exactly reproduce a complete native forward. The
bf16 trajectory comparison includes quantization and fusion/storage differences.

The HRX/Loom component checks measure text-encoder cosine 0.999847, text-fusion
cosine 0.999979 and complete-transformer cosine 0.990253 against the independent
outer-model implementation with Loom blocks. VAE RGB mean absolute errors are
1.223/255 at 64², 0.905/255 at 256² and 0.932/255 at 320×272. The saved HIP VAE
produces similar errors; HRX and HIP differ by 0.16–0.17/255 on these inputs. The earlier scheduler exactness claim used CPU sigmas
and did not test the real pipeline’s CUDA rounding. The corrected test uses CUDA
sigmas: all 5,050 steps across step counts 1–100 match exactly, as does a
64×64 two-step image driven by the native components. See
`docs/notes.md` for thresholds and the remaining image-quality measurement.

## Results so far (1024x1024, 8 Turbo steps, seed 0)

These are historical measurements from before the manual loader was corrected to
preserve fp32 normalization weights. Quality figures need a new baseline with the
corrected loader; the scripts now propagate failures before reporting completion.

| | per forward of the 28 blocks | latent PSNR vs bf16 | image PSNR vs bf16 |
| --- | ---: | ---: | ---: |
| torch bf16 (diffusers) | ~7 s | | |
| reference W4A4 in torch | 87 s | 9.96 dB | 18.9 dB |
| Loom W4A4, first attention | 7.85 s | 10.09 dB | 19.0 dB |
| Loom W4A4, staged attention | 2.89 s (29.2 s per image) | 10.09 dB | 19.04 dB |
| Loom W4A4, Q resident + V transposed in LDS, raster groups | 2.31 s (25.4 s per image) | 12.92 dB | 23.23 dB |
| Loom W4A4, prepare kernels vectorised | 2.13 s (21.9 s per image, steady state) | | |
| Loom W4A4, SwiGLU fused into the gate/up GEMM | same (exact; 270 MB per block less traffic) | | |

The per-image time is the second image of a process (`tools/pipeline.py --images 2`):
8 steps at 2.35 s plus text encoding and the tiled VAE decode. The first image pays the
Loom session build (about 6 s) and, once per machine, MIOpen's convolution kernel search
for the VAE decode (minutes; cached afterwards).

The three pictures (`build/bf16_seed0.png`, `build/w4a4_seed0.png`,
`build/loom_seed0.png`) are the same fox in the same pose and light; the deviation is
fur and snow detail, not artifacts. For comparison, a published int4 ConvRot checkpoint
of the same model that keeps 96 of its 224 block linears in int8 reports a minimum
image PSNR of 17.7 dB against bf16; this port quantises every block GEMM to int4 with
per-row scales and measures 19.0.

Where a forward goes now: the four GEMMs 68% (at 56-75 TOPS on the part's measured
117 TOPS int4 ceiling), attention 26% (20 TFLOP/s), the prepare kernels 5%.

The PSNR between two 8-step trajectories swings by several dB between numerically
near-identical runs (the sampler amplifies rounding); the per-block cosine against the
reference in `tests/test_blocks.py` is the metric that tracks kernel correctness.

See `docs/notes.md` for every decision and measurement.
