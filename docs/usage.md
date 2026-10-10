# Models, generation and Rust API

Commands run from the repository root.

## Models

Models from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) are downloaded
and reused in the **standard Hugging Face cache**, normally
`~/.cache/huggingface/hub`. No separate models directory is needed:

```sh
krea2 -p "a red fox in the snow"                     # Turbo by default
krea2 --checkpoint raw -p "a mountain lake at dawn"
```

`HF_HUB_CACHE` overrides the cache directory. Otherwise it is `$HF_HOME/hub`,
with `HF_HOME` defaulting to `$XDG_CACHE_HOME/huggingface` when set, or
`~/.cache/huggingface`. Existing cached downloads are reused before network access.
`HF_HUB_OFFLINE=1` requires cached or local weights and disables downloads.
HF boolean variables follow the [standard convention](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#boolean-values):
`1`, `ON`, `YES`, and `TRUE` enable them (case-insensitive); other values, including
`false`, leave them disabled. The bundled tokenizer needs no download.

The checkpoint, Qwen3-VL-4B text encoder, and Qwen-Image VAE all resolve through
the cache automatically, pinned to upstream revision
[`e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96`](https://huggingface.co/Comfy-Org/Krea-2/tree/e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96).
The default text encoder is always BF16. To use FP8, pass its file explicitly
with `--text-encoder`; cache contents never change the default precision.

To select a checkpoint by name explicitly:

```sh
krea2 --model krea2_turbo_int8_convrot -p "a red fox in the snow"
```

For individual files outside the cache, pass explicit paths. They can live
anywhere; no directory layout is required. Components without an override
continue to use the HF cache.

```sh
krea2 --model /path/to/checkpoint.safetensors --checkpoint turbo \
  --text-encoder /path/to/encoder.safetensors --vae /path/to/decoder.safetensors \
  -p "a lighthouse at dusk"
```

## Generate

```sh
krea2 -p "a red fox in the snow" --out fox.png
krea2 -p "a lighthouse at dusk" \
  --width 768 --height 1024 --seed 7 --out lighthouse.png
printf '%s\n' "a mountain lake at dawn" | krea2 --checkpoint raw \
  --negative "blurry" --guidance 3.5 --out lake.png
```

| Checkpoint | Default steps | Default guidance |
| --- | ---: | --- |
| Turbo | 8 | Disabled |
| Raw | 52 | 3.5, using `cond + g * (cond - uncond)` |

`--steps` overrides the step count. `--images N` uses consecutive seeds and
numbered output files. Dimensions must be multiples of 16 between 64 and 2048;
text plus image tokens must fit the 16,896-token limit. Output supports `.png`
and `.ppm` (case-insensitive); unsupported extensions fail before model loading.

Use `--model`, `--text-encoder` and `--vae` for individual files. Custom checkpoint
paths require `--checkpoint turbo` or `--checkpoint raw`; the two built-in model
identifiers select their corresponding sampler. Filenames are never guessed.
`krea2 --help` lists all options. Invalid arguments exit with code 2; model,
runtime, and file I/O failures exit with code 1. `KREA2_NATIVE_PROFILE=1` prints
synchronized stage timings and a per-kernel breakdown of the transformer blocks,
measured on the GPU clock where the runtime supports it; profiled runs are
slower, so do not time them.

## Rust library

`cargo build --release` produces `target/release/krea2`; `scripts/build.sh` also
copies it to `build/krea2`. To use the library from another Rust project:

```toml
[dependencies]
krea2 = { package = "krea2-hrx", git = "https://github.com/zacharydenton/krea2-hrx" }
```

Pin a `rev` for reproducible downstream builds. The Rust import remains `krea2`:
`krea2::pipeline` provides prompt-to-RGB generation and the individual
pipeline components; `krea2::session` provides resident transformer block sessions.
For custom checkpoint paths, library callers select the sampler with
`Files::of(path).distilled(Some(true))` for Turbo or `Some(false)` for Raw.
Every call returns `krea2::Error`: `is_invalid_argument()` separates
the caller's mistakes from runtime failures, which keep their `hrx::Error`, and
`Error::Cancelled` reports a progress callback that stopped generation.

There is no C ABI. Consuming applications bind the Rust library directly — an
Elixir application wraps `krea2::pipeline::Pipeline` with Rustler, which needs no
C boundary, no generated headers and no error-buffer protocol. See the
[runtime and deployment guide](../docs/hrx-runtime.md) for lifetimes and threading.
