# Character LoRA training

The experimental `krea2-train` binary implements RAW character training in
Rust, Loom and HRX. Dataset decoding, cropping, tokenization, VAE encoding,
cached conditioning, differentiation, AdamW, serialization and inference are
native. No Python generator, Python runtime, PyTorch, CUDA or external training
framework is required.

**GPU numerical validation, a complete character trial, and ComfyUI loading
validation are still pending.** Compilation and CPU tests do not establish
training correctness or character quality. Attention currently uses a streaming
correctness implementation; its speed has not been measured. Adapter-enabled
INT8 inference also uses separate projections and has not been performance tuned.

## Configuration

Keep datasets, downloaded weights, caches, checkpoints, logs, evaluation images
and benchmark reports outside the checkout. Create a JSON configuration such as
`~/training/character.json`:

```json
{
  "model": "models/krea2_raw_bf16.safetensors",
  "dataset": "character/images",
  "trigger": "mycharacter",
  "output": "character/run",
  "resolution": 1024,
  "rank": 32,
  "alpha": 32,
  "steps": 1500,
  "accumulation": 1,
  "learning_rate": 0.0001,
  "save_every": 250,
  "keep_checkpoints": 4,
  "seed": 37,
  "memory_gib": 96,
  "offline": true
}
```

Paths resolve relative to the configuration file. Supply an original-basis
Krea 2 RAW BF16 checkpoint. The trainer rejects quantized main-block weights;
the user must select RAW rather than Turbo. BF16 text encoder and Qwen VAE
weights resolve from the existing pinned Hugging Face cache, or from explicit
`text_encoder` and `vae` paths. Set `offline` to false to allow missing auxiliary
weights to download.

Every JPEG or PNG needs a same-stem `.txt` caption containing the trigger.
The scanner checks for missing captions and byte-identical duplicate images.
It applies EXIF orientation, chooses among seven aspect buckets at the requested
512, 768 or 1024 target area, then resizes and center-crops without flips.
Keep held-out references outside the image directory. Preparation writes
`crops.png` for inspecting the actual crops.

The default physical batch is one image. `accumulation` averages several
microbatches per update. Defaults are rank/alpha 32/32, constant learning rate
1e-4, AdamW betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and global gradient
norm cap 1. These are starting settings, not established character-quality
recommendations. Training steps count optimizer updates, not epochs.

## Workflow

```sh
# CPU only: validate model projection shapes, image/caption pairs and buckets.
cargo run --locked --bin krea2-train -- inspect --config ~/training/character.json

# GPU: cache posterior moments and fully fused RAW conditioning, in separate phases.
cargo run --release --locked --bin krea2-train -- prepare --config ~/training/character.json

# GPU: train with both preprocessing models released.
cargo run --release --locked --bin krea2-train -- run --config ~/training/character.json

# Resume the FP32 master weights, moments, sample order and random generator.
cargo run --release --locked --bin krea2-train -- run \
  --resume ~/training/character/run/checkpoints/step-000250
```

Training freezes the base model and text encoder. Each of the 28 main transformer
blocks has adapters on `attn.wq`, `attn.wk`, `attn.wv`, `attn.gate`, `attn.wo`,
`mlp.gate`, `mlp.up` and `mlp.down`. Factors execute in BF16 with FP32 master
weights, parameter gradients and optimizer moments. Intermediate activation
gradients use BF16. Block inputs are retained and each block is recomputed during
backpropagation. Grouped-query attention stores linear-sized statistics and
computes gradients without floating-point atomics.

The memory budget limits HRX allocations during preparation and training. A
conservative training estimate is checked before opening its stream. The initial
target is a 128 GiB Strix Halo system, not the guide's 16 GB NVIDIA setup.

Caches bind image bytes, captions, buckets, model contents, tokenizer, host and
kernel source, dependency lockfile and compiler identity. Changed data or
preprocessing requires a fresh output directory. Resume also checks the device
target. A failed training update cannot be retried or saved through the same
trainer; reopen a complete checkpoint.

Each complete checkpoint directory contains a portable BF16
`adapter.safetensors`, an FP32 `optimizer.safetensors`, and `state.json`.
Directories become visible atomically after writing; retention removes only
older complete checkpoints. The portable adapter uses original-basis
`diffusion_model.blocks.*.{projection}.lora_A.weight`, `lora_B.weight` and
explicit `alpha` tensors. Its compatibility with ComfyUI still needs validation.

## Evaluate

```sh
cargo run --release --locked --bin krea2-train -- evaluate \
  --run ~/training/character/run --model krea2_turbo_bf16 --strengths 0.7,0.9,1.1

cargo run --release --locked --bin krea2 -- \
  --lora ~/training/character/run/checkpoints/step-001250/adapter.safetensors \
  --lora-strength 0.9 -p 'mycharacter wearing a blue suit in a library' --out ~/training/check.png
```

Evaluation uses two fixed seeds and six prompts covering portrait, profile,
expression, changed clothing, changed background and full body. Override prompts
with `validation_prompts`. Compare retained checkpoints against held-out
references and assess identity, prompt response and artifacts. Loss alone does
not choose the best checkpoint. The adapter branch uses original-basis inputs
for both BF16 and INT8 ConvRot base projections. A strength of zero selects
the unchanged base inference path.

## Checks and Criterion benches

```sh
scripts/test.sh --cpu
# Uses the provisioned compiler, but opens no GPU stream:
cargo test --locked --test training_compile -- --ignored --test-threads=1
# Only when the GPU is available:
cargo test --locked --test training_gpu -- --ignored --test-threads=1

cargo bench --locked --bench training -- training/projection
cargo bench --locked --bench training -- training/attention
cargo bench --locked --bench training -- training/adamw
KREA2_TRAIN_BENCH_CONFIG=~/training/character.json \
  cargo bench --locked --bench training -- training/prepared
```

The prepared-step benchmark requires existing caches. It warms compilation,
keeps the model resident, advances temporary in-memory optimizer state, and
excludes checkpoint serialization. It writes no training checkpoints. All
benchmarks in this feature are runnable Criterion benches; measurements remain
under the ignored Cargo target directory.
