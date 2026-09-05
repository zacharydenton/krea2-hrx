# Local ComfyUI comparison, 2026-09-05

The installed ComfyUI INT8 setup averaged **105.74 s/image**, versus **31.88 s**
for the current native Loom runtime: **3.32× throughput** on the two measured
warm samples. A separate video-generation job was active on the GPU throughout
these sequential benchmark batches. Loom's samples varied substantially, so this
is a provisional local comparison, not an isolated hardware speedup claim.

| Backend | First image | Warm 1 | Warm 2 | Warm mean |
| --- | ---: | ---: | ---: | ---: |
| ComfyUI INT8 ConvRot | 118.73 s | 105.85 s | 105.62 s | 105.74 s |
| Native Loom INT4 | 40.09 s | 20.82 s | 42.94 s | 31.88 s |

Settings: 1024×1024, eight Euler steps, CFG 1, batch one, prompt
`a red fox in the snow`, seed 0. Native seed 0 and ComfyUI seed 0 do not produce
identical initial noise. Precision, implementation and sampler rounding differ;
this measures performance and does not establish image-quality equivalence.
All three outputs within each backend had identical RGB checksums.

ComfyUI uses `/home/zach/code/ComfyUI` at `250b2e95`, with its default
DynamicVRAM setup, PyTorch attention and automatic HIP Comfy Kitchen operations.
The local model directory is `/home/zach/comfy-models` (a symlink into `/mnt/usb`);
`/home/zach/comfyui-models` was not present. Model files:

- `diffusion_models/krea2_turbo_int8_convrot.safetensors`
- `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `vae/qwen_image_vae.safetensors`

ComfyUI uses bf16 DiT compute, fp16 text compute and bf16 VAE compute. Its sampler
is `euler` with the standard `simple` schedule and the model's 1.15 shift. Loom
uses its fixed smoothed INT4-QK/fp16-PV attention and W4A4 block projections,
with the corrected scheduler and resident weight/modulation paths.

The ComfyUI benchmark invokes the standard loader, encoder, sampler and VAE APIs
directly. It re-executes inference every time, excluding server/UI, previews,
custom nodes and PNG encoding. It includes text encoding on each call. ComfyUI
warm stage means were 1.71 s text, 100.43 s denoising and 3.59 s VAE. With cached
text conditioning, its sampling-plus-decoding mean would be 104.02 s. ComfyUI's
last float-to-RGB8 conversion is outside its timer; the native C-API timer includes
RGB8 output. Torch's reported allocation peak excludes DynamicVRAM allocations
and must not be used as a total-memory comparison.

Initialization was 0.94 s for ComfyUI and 10.69 s for Loom. These are not equivalent:
ComfyUI defers most weight loading until the first image; Loom loads eagerly.
Including setup, the first completed images took approximately 119.68 s and
50.78 s respectively. Kernel/compiler and OS caches were not cleared.

ComfyUI ran in the existing `amd-strix-halo-comfyui` toolbox with PyTorch
`2.14.0a0+rocm7.15.0a20260721`. The toolbox had comfy-aimdo 0.4.15; the checkout
requires 0.5.2. Exact requirement versions were installed with `--no-deps` into
`build/comfy-bench-deps`, leaving the existing installation unchanged. The
benchmark enabled checkpoint `--disable-mmap`, as recommended by the local
Strix Halo toolbox documentation. The ComfyUI checkout was not edited. The
previously stopped toolbox was stopped again after measurement.

Reproduction:

```sh
# Inside the ComfyUI toolbox, after making its dependencies available:
PYTHONPATH=/home/zach/code/krea2-loom/build/comfy-bench-deps \
  /opt/venv/bin/python /home/zach/code/krea2-loom/tools/bench_comfyui.py

# On the host, after the ComfyUI benchmark exits:
source scripts/env.sh
env -u LD_LIBRARY_PATH -u KREA2_NATIVE_PROFILE \
  python3 tools/bench_native.py --bundle build/native-deploy --runs 3
```

Raw logs and machine-readable results are in `build/comfy-comparison/`.
