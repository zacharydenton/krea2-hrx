# Character LoRA training

The experimental `krea2-train` binary implements RAW character training in
Rust, Loom and HRX. Dataset decoding, cropping, tokenization, VAE encoding,
cached conditioning, differentiation, AdamW, serialization and inference are
native. No Python generator, Python runtime, PyTorch, CUDA or external training
framework is required.

**A complete character trial and ComfyUI loading validation are still pending.**
Primitive GPU tests cover flow loss, LoRA forward/backward, AdamW, normalization,
rotary gradients, grouped attention and original-basis adapters on ConvRot
projections. A two-update RAW run with prepared 1024-area inputs passed exact
checkpoint/resume equivalence, and its exported adapter loaded into native Turbo
inference. These checks do not establish character quality or full-model reference
parity. Attention uses tiled BF16 matrix operations for four query heads per
KV head, with FP32 softmax, saved statistics and gradient accumulation.
Probabilities and score gradients are split into a BF16 high part and residual
before the output and gradient products. Backward recomputes each score tile
and sums shared key/value gradients without atomics or a quadratic score buffer.
Saved output uses full FP32 division, and its softmax correction uses the same
matrix reduction as backward to avoid false gradients in saturated attention.
The changed reduction order is not bitwise equivalent to scalar attention;
checkpoint/resume equivalence is checked within the same implementation.
Other head ratios and sequences shorter than 16 tokens use the scalar streaming
implementation. Adapter-enabled
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
  "targets": "all",
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
CPU inspection rejects captions exceeding 512 conditioning tokens, including
the assistant suffix but excluding the 34-token system prefix. Training never
silently truncates captions.
Keep held-out references outside the image directory. Preparation writes
`crops.png` for inspecting the actual crops.

The default physical batch is one image. `accumulation` averages several
microbatches per update. Defaults are rank/alpha 32/32, constant learning rate
1e-4, AdamW betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and global gradient
norm cap 1. These are starting settings, not established character-quality
recommendations. Training steps count optimizer updates, not epochs.

Timesteps follow a logit-normal distribution with Krea's resolution-dependent
shift, matching Musubi's `krea2_shift` with sigmoid scale 1. This differs from
AI Toolkit's unshifted `timestep_type: sigmoid`. The sampled FP32 flow time is
retained through sinusoidal embedding; only the resulting features are rounded
to BF16. Inference keeps its existing timestep rounding for trajectory parity.

## Workflow

```sh
# CPU only: validate model projection shapes, image/caption pairs and buckets.
cargo run --locked --bin krea2-train -- inspect --config ~/training/character.json

# GPU: cache posterior moments and frozen Qwen taps, in separate phases.
cargo run --release --locked --bin krea2-train -- prepare --config ~/training/character.json

# GPU: train with both preprocessing models released.
cargo run --release --locked --bin krea2-train -- run --config ~/training/character.json

# Optional short trial: save after 25 updates without changing the final step target.
cargo run --release --locked --bin krea2-train -- run \
  --config ~/training/character.json --stop-after 25

# Resume the FP32 master weights, moments, sample order and random generator.
cargo run --release --locked --bin krea2-train -- run \
  --resume ~/training/character/run/checkpoints/step-000250
```

Training freezes the base model and text encoder. Each of the 28 main transformer
blocks has adapters on `attn.wq`, `attn.wk`, `attn.wv`, `attn.gate`, `attn.wo`,
`mlp.gate`, `mlp.up` and `mlp.down`. Select `"targets": "all"` for all 264 DiT
linears: those 224, the 32 projections in the four text-fusion blocks,
`txtfusion.projector`, `txtmlp.1`, `txtmlp.3`, `tmlp.0`, `tmlp.2`, `tproj.1`,
`first`, and `last.linear`. Qwen, the VAE, normalization scales and modulation
tables remain frozen. Full training caches the twelve Qwen taps before the
trainable fusion tower. Broadcast modulation gradients accumulate in FP32 across
all 28 main blocks, then feed the timestep projections.

Omitting `targets` retains the existing `main_blocks` profile with 224 targets,
which caches fully fused text conditioning. The two profiles have distinct cache
identities; use a fresh output directory when switching. Initialization keeps
the same main-block RNG sequence in both modes. Resume restores the complete
selected target set, and native adapter inference applies auxiliary targets too.
Full-target mode retains the small auxiliary graph even when main-block
checkpointing is enabled; its storage is included in the allocation estimate.
Factors execute in BF16 with FP32 master
weights, parameter gradients and optimizer moments. Intermediate activation
gradients use BF16. By default, block inputs are retained and each block is
recomputed during backpropagation. Set `gradient_checkpointing` to `false` to
retain block activations and skip recomputation when the memory estimate fits
your budget. `scratch_pool_mib` controls the free device storage retained for
reuse (default 2048 MiB, maximum 16384 MiB). A larger pool can avoid repeatedly
allocating activation tapes when checkpointing is disabled; its full capacity
is included in the allocation estimate. Each block completes before advancing,
bounding temporary storage retained by queued dispatches. Grouped-query attention
retains linear-sized statistics and computes gradients without floating-point
atomics.

Optimizer updates replay prepared HRX graphs for all parameter norms and AdamW
operations. The complete global norm is read and checked before any parameter
update; a nonfinite gradient cancels the update without changing optimizer state.

The memory budget limits HRX allocations during preparation and training. A
conservative training estimate is checked before opening its stream. The
estimate counts BF16 weights once and includes the additional execution copies
of F32 weights, saved activations, adapter state and scratch capacity. Checkpoint
file size is not doubled to account for upload staging; HRX bounds staging
independently. Temporary host buffers are estimated separately from the largest
weight conversion and adapter checkpoint serialization, taking the larger of
those phases rather than adding them. They do not count against the HRX device
budget, but do count toward required system RAM. The initial
target is a 128 GiB Strix Halo system, not the guide's 16 GB NVIDIA setup.
The trainer requires available system RAM to cover device allocations, temporary
host buffers and one 8 GiB system reserve. It aborts between blocks if available
RAM falls below the same 8 GiB floor. That reserve is a policy allowance for the
desktop and other processes, not a measured model requirement. Swap is not
counted as GPU capacity. These checks cannot reserve RAM
against other processes; the `run` and `prepare` commands also mark themselves
as preferred OOM victims so system pressure targets training before the desktop.
Builds and full-model trials should run separately on a shared machine.

Use `KREA2_NATIVE_PROFILE=1` for synchronized forward, recomputation, backward,
and optimizer stage timings. For individual dense kernels, the runnable Criterion
bench exports detailed Loom compiler reports and HRX GPU-clock intervals when
`KREA2_BENCH_REPORT_DIR` points outside the checkout:

```sh
KREA2_BENCH_REPORT_DIR=~/.local/state/krea2-hrx/training-reports \
  cargo bench --locked --bench training -- 'training/dense/.*1043'
```

GPU-clock probes run during benchmark setup. Criterion measures ordinary,
uninstrumented dispatches afterward; profiling markers serialize execution and
must not be treated as end-to-end timing. Realistic token counts include text
tokens, so the benches cover ragged shapes such as 1043 and 4115.

`KREA2_HOST_PROFILE=1` reports per-update host wall time for training matrix
operations, transposes and optimizer submissions. These include compilation,
allocation and queue backpressure, and do not measure kernel execution time.
The `bf16_allocation` category is a subset of the matrix-operation timings;
do not add it to those totals. Both profiling switches are off by default.

Caches bind image bytes, captions, buckets, model contents, tokenizer,
preprocessing source, dependency lockfile and compiler identity. Changed data or
preprocessing requires a fresh output directory. Resume also checks the device
target and the complete training implementation. Optimizer/backward changes can
reuse preprocessing caches but cannot resume old optimizer state. A failed training
update cannot be retried or saved through the same trainer; reopen a complete
checkpoint.

`run --stop-after N` counts additional updates, including when resuming. It saves
the final update before exiting, even between regular checkpoint intervals.

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
# Full-model update/checkpoint/resume equivalence, using external prepared caches:
KREA2_TRAIN_TEST_CONFIG=~/training/character.json \
  cargo test --release --locked --test training_resume -- --ignored --test-threads=1 --nocapture
# Re-encode cached samples and inspect reconstructions in the run's validation/vae directory:
KREA2_TRAIN_TEST_CONFIG=~/training/character.json \
  cargo test --release --locked --test training_vae -- --ignored --test-threads=1 --nocapture

cargo bench --locked --bench training -- training/projection
cargo bench --locked --bench training -- training/attention
cargo bench --locked --bench training -- training/fusion_attention
cargo bench --locked --bench training -- training/frozen_backward
cargo bench --locked --bench training -- training/adamw
cargo bench --locked --bench training -- training/optimizer
KREA2_TRAIN_BENCH_CONFIG=~/training/character.json \
  cargo bench --locked --bench training -- training/prepared
```

The prepared-step benchmark requires existing caches. It warms compilation,
keeps the model resident, advances temporary in-memory optimizer state, and
excludes checkpoint serialization. It writes no training checkpoints. All
benchmarks in this feature are runnable Criterion benches; measurements remain
under the ignored Cargo target directory.
The optimizer comparison uses all 448 rank-32 parameter tensors, including
gradient norm readback and clipping, and needs roughly 2 GiB of device storage.
