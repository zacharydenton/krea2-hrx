# krea2-hrx

Krea 2 Turbo and Raw image generation in Rust for AMD Strix Halo.
The text encoder, diffusion transformer and VAE use custom
[Loom](https://github.com/ROCm/hrx-system) kernels through
[HRX](https://github.com/zacharydenton/hrx-rs).

## Gallery

[![A figure on a basalt sea cliff beneath a ringed planet](docs/images/planetrise.png)](docs/gallery.md)

Turbo, 1344×768, eight steps, seed 64.
[Prompts, commands and more images](docs/gallery.md).

## Features

- Turbo and Raw checkpoints, with automatic model downloads or local files.
- ComfyUI INT8 ConvRot weights, W8A8 transformer products and FP16 attention.
- Qwen3-VL-4B text encoding and Qwen-Image VAE decoding.
- Rust library and CLI, with reusable model sessions.
- Experimental [LoRA and Raw training](docs/training.md); GPU qualification is incomplete.

## Quick start

Requires Linux, Rust 1.91+ and Radeon 8060S (`gfx1151`) with amdgpu/KFD access.
The transformer checkpoint alone is about 13 GB; allow additional memory for
the text encoder, VAE and workspace. [Runtime requirements](docs/hrx-runtime.md).

```sh
git clone https://github.com/zacharydenton/krea2-hrx.git
cd krea2-hrx
cargo install --locked --path .
krea2 -p "a red fox in the snow" --out fox.png
```

First use downloads missing weights and prepares the HRX runtime and Loom kernels.
The default is Turbo at eight steps. Raw defaults to 52 steps with guidance 3.5:

```sh
krea2 --checkpoint raw -p "a mountain lake at dawn" --out lake.png
krea2 -p "a lighthouse at dusk" \
  --width 768 --height 1024 --seed 7 --out lighthouse.png
```

Dimensions must be multiples of 16 between 64 and 2048, within the token limit.
Use `--images N` for consecutive seeds and numbered outputs.
[Model files, offline use and generation options](docs/usage.md).

## Rust library

```toml
[dependencies]
krea2 = { package = "krea2-hrx", git = "https://github.com/zacharydenton/krea2-hrx" }
```

Pin a revision for reproducible builds. `krea2::pipeline` handles prompt-to-RGB
generation; `krea2::session` exposes resident transformer sessions.
[API, errors and checkpoint selection](docs/usage.md#rust-library).

## Development

```sh
cargo bench --bench kernels
cargo bench --bench generation -- generation/1024x1024
```

Generation benchmarks need cached weights and `gfx1151`. The parity gate compares
complete trajectories with the unquantized BF16 model using identical initial
latents. [Tests](docs/testing.md) · [Benchmark setup](CONTRIBUTING.md#benchmarks) ·
[Architecture](docs/notes.md) · [Contributing](CONTRIBUTING.md)

## License

[Code license](LICENSE). The bundled tokenizer has separate
[attribution](assets/README.md). Model weights retain their upstream licenses.
