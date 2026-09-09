# ComfyUI comparison

## Latest measurements

Measured on one idle Radeon 8060S on 2026-09-08. Both backends used Krea 2 Turbo
int8 ConvRot, 1024×1024, eight Euler steps, batch one, seed 0 and prompt
`a red fox in the snow`. ComfyUI used CFG 1; the native Turbo path was unguided.

| Warm timing | ComfyUI INT8 ConvRot | krea2-loom W8A8 |
| --- | ---: | ---: |
| Generation median | 36.31 s | 27.30 s |
| Denoising | 34.52 s | About 24 s |
| VAE decode | 0.652 s | 0.586 s |

Warm generation samples were 36.83 and 35.80 s for ComfyUI, and 27.30, 27.13
and 28.06 s for krea2-loom. The generation ratio is 1.33× on these samples.
The [VAE report](vae-performance.md) describes the decoder change.
September 8 samples are recorded in commit `0a8651f`; separate machine-readable
logs were not committed. [September 7 raw logs](benchmarks/comfyui-2026-09-07.txt)
and [W8A8 results](benchmarks/w8a8-2026-09-07.txt) are historical baselines.

## What is being compared

ComfyUI loads the checkpoint through its own APIs, using bf16 transformer
compute, PyTorch attention, Comfy Kitchen GEMMs and DynamicVRAM. The native
runtime uses the same int8 checkpoint rows with per-token int8 activations,
fp16 attention and tiled VAE decoding. Model loading and kernel warmup are
outside warm timings; text encoding and decode run on every iteration.

The timers are not identical: the ComfyUI timer ends after GPU decode, before
RGB8 conversion and download; the native timer includes RGB8 output. Both
exclude PNG encoding and UI/server overhead. ComfyUI's reported Torch memory
peak excludes DynamicVRAM allocations and is not a total-memory comparison.

Matching seeds do not produce matching noise across backends. Repeated RGB
checksums establish repeatability within each backend, not image equivalence.
For quality comparisons, supply the same initial latents, conditioning and
schedule. ComfyUI is not the accuracy reference: it runs the same quantized
checkpoint, so agreeing with it measures two approximations against each other.
The release gate compares against the original unquantized BF16 model instead --
see [the parity gate](testing.md#parity-against-the-unquantized-model).

## Reproduce

Run the ComfyUI command in an environment with that checkout's dependencies and
ROCm PyTorch. Replace the paths with your installations:

```sh
python /path/to/krea2-loom/tools/bench_comfyui.py \
  --comfy /path/to/ComfyUI --models /path/to/comfy-models \
  --size 1024 --steps 8 --runs 4 --output-dir /path/to/comfy-results
```

From the krea2-loom checkout:


> These commands are recorded as they were run. The Python benchmark
> tooling and `scripts/env.sh` were retired in e33b171; the measurements
> stand, but reproducing them means recovering those tools from Git.
```sh
source scripts/env.sh
scripts/build.sh
.venv/bin/python tools/bench_native.py \
  --model /path/to/comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors \
  --size 1024 --steps 8 --runs 4
```

The ComfyUI harness bypasses graph-output caching and supports `--width` and
`--height` for rectangles. It saves initial noise, the latest PNG and metadata
when `--output-dir` is supplied. Redirect stdout to retain every timing sample.

For a new comparison, record revisions and runtime versions, disable profiling,
exclude run zero, alternate backends and keep the GPU idle. Report sample counts
and timer differences with any speedup claim.
