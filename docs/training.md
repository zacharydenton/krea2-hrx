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
atomics. Tiled attention also retains padded Q/K/V and transposed Q/K for the
reverse pass. A fused packing kernel writes both layouts and zeroes tail rows
without changing BF16 bits; backward only packs the incoming gradient. The
allocation estimate includes these cached layouts. This reuses forward data
without quantizing attention or changing the gradient arithmetic.
For sequences up to 1536 tokens, dQ keeps its Q tile in registers; dK/dV
splits its K tile between registers and shared memory. This avoids repeated
tile loads from shared memory and allows more resident waves without spilling
registers. Longer sequences keep both tiles in shared memory, which benchmarks
favor at larger working sets. Loom specializes this choice at compile time.

Optimizer updates replay prepared HRX graphs for all parameter norms and AdamW
operations. The complete global norm is read and checked before any parameter
update; a nonfinite gradient cancels the update without changing optimizer state.

MLP and attention gate derivatives each use one fused dispatch. It reads the
cached forward activation and writes both branch gradients, preserving the BF16
rounding of the intermediate product without materializing that temporary tensor.
With all adapter targets enabled, each modulation stage also fuses its scale,
shift and residual-gate gradient reductions. It preserves BF16 product rounding
and the FP32 row summation order while avoiding two product tensors.
The 128-channel Q/K RMSNorm reverse pass retains each lane's inputs and scaled
gradients in registers across its reduction. Each row keeps its original
summation and shuffle order. Other widths use the general two-pass kernel.
The model combines inverse rotary and Q/K norm backward in one dispatch. Adjacent
lanes exchange rotary partners and round the result to BF16 in registers before
the norm derivative, avoiding the intermediate gradient matrix.
The main-block norm reverse paths fuse broadcast modulation scaling and residual
addition into the norm dispatch. BF16 rounding still occurs after adding one to
the modulation, scaling the gradient, computing the norm gradient and adding the
residual; the temporary vectors and matrices are not materialized.
The main-block forward path also fuses RMSNorm, modulation scaling and shift.
It retains the normalized BF16 values needed by modulation gradients, preserving
the original FP32 reduction tree and all intermediate BF16 roundings.
Q/K forward normalization and rotary embedding also share one dispatch. Each
wave keeps a 128-channel head in registers and exchanges rounded normalized
values between adjacent lanes; backward retains the original Q/K inputs.
Forward sigmoid/SiLU gates fuse activation with multiplication, retaining the
rounded activation for backward. Residual gates fuse broadcast multiplication
and addition while preserving the product's BF16 rounding. Both paths use
four-element vector loads, with scalar-operation fallbacks for ragged extents.
Backward gates also process four BF16 elements per thread when the element count
is divisible by four. Ragged counts retain the single-dispatch scalar kernel;
both variants preserve the product rounding and recompute the FP32 derivative.

Both main-block and auxiliary tapes retain each adapter's rank-sized forward
activation for its parameter gradient, avoiding a duplicate input projection in
backward. These activations expire with the tape before the optimizer update;
the memory estimate includes them for retained and recomputed blocks.

Rank-32/64 adapter output projections and input gradients fuse their matrix
product with the base-result addition. Vector loads stage both weight layouts
directly, and the product still rounds to BF16 before scaling and adding.
When input gradients from multiple branches meet, the same output stage adds
the preceding gradient after rounding the current branch to BF16. This avoids
four full-size intermediate gradient tensors and four dispatches per main block,
without changing the branch addition order or parameter-gradient accumulation.
Other ranks and dimensions retain the separate operations.

The memory budget limits HRX allocations during preparation and training. A
conservative training estimate is checked before opening its stream. The
estimate counts BF16 weights once and includes the additional execution copies
of F32 weights, saved activations, adapter state and scratch capacity. Checkpoint
file size is not doubled to account for upload staging; HRX bounds staging
independently. Temporary host buffers are estimated separately from the largest
weight conversion and adapter checkpoint serialization, taking the larger of
those phases rather than adding them. Auxiliary weights load before the FP32
host adapter is created. Checkpointing releases the portable
adapter's FP32 host copy before collecting optimizer state. These temporary
buffers do not count against the HRX device
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

Set `KREA2_BENCH_KERNEL_DIR` to a directory of candidate `<kernel-name>.loom`
files to compare dense and adapter-gradient kernel changes without rebuilding
the benchmark. Every selected kernel must have a source file in that directory.
Normal training continues to use the embedded kernels; keep candidate files and
reports outside the checkout. The `training/adapter_gradient` cases measure
direct FP32 parameter-gradient accumulation in both projection directions.
They check exact FP32 agreement across three accumulations into nonzero buffers
before timing. `KREA2_BENCH_REFERENCE_DIR` enables alternating paired timings
against a reference `train_gemm_tn_accumulate.loom`; each side has its own output
buffer so both see the same accumulation history.

For dense kernels, `KREA2_BENCH_REFERENCE_DIR` adds a reference source directory.
Reference kernels must use the same names, binding ABI and launch tile geometry.
Before timing, paired dense benchmarks require exact BF16 output agreement on
the resident inputs; comparison and readback are outside the measured interval.
Each iteration runs the reference and candidate on the same resident tensors,
alternating their order. Criterion records candidate time; a paired summary
reports both averages and their ratio across warm-up and measured iterations.
The paired ratio is descriptive, without a confidence interval. Check identical
sources and swap reference/candidate roles when investigating small differences.
This helps compare kernels under changing GPU contention, but does not replace an
uncontended end-to-end measurement.

`training/attention_pack` measures the bit-preserving padded row-major and
transposed layouts used by attention. It checks both complete outputs against
CPU indexing before timing, including zero headroom. It accepts the same
candidate and reference directory settings for `train_attention_pack.loom`.
Four-value transfers cover aligned columns and capacities; scalar guards handle
other shapes.

`training/narrow` compares the rank-32/64 output GEMM with the previous general
tiles in both operand layouts. It checks exact BF16 parity before alternating
timings, and accepts `KREA2_BENCH_KERNEL_DIR` for a candidate
`train_gemm_narrow.loom` with the same 16×32 output tile and 64-thread workgroup.
The production path uses this tile for at least 512 rows and full 64-element K
tiles; other shapes retain the general fallback.

`training/frozen_backward/cached_transpose` measures reusing a frozen weight
transpose created during setup. Set `KREA2_BENCH_PAIR_FROZEN_DIRECT=1` to pair it
with the direct reverse GEMM on the same inputs. This is an experiment; training
does not retain these weight transposes. `training/gated_backward` always pairs
the fused gate derivative with the original three-operation sequence. Both use
the same alternating-order timing and summary as the dense comparison.
`training/modulation_backward` compares the fused modulation reduction with its
original two products and three reductions using the same paired timing.
`training/norm_backward` pairs the specialized Q/K kernel with the general
per-row kernel, checks exact BF16 output agreement and exports both compiler
reports through `KREA2_BENCH_REPORT_DIR`.
`training/modulated_norm_backward` pairs the fused main-block path with its
original four-operation sequence, including allocation and dispatch overhead.
`training/modulated_norm_forward` compares fused normalization/modulation with
the separate forward operations and checks both outputs for exact agreement.
`training/rope_norm_backward` pairs fused inverse rotary and Q/K norm backward
with the separate operations for both head counts. It checks exact BF16 agreement
and exports compiler reports through `KREA2_BENCH_REPORT_DIR`.
`training/norm_rope_forward` does the same for forward normalization and rotary,
including the normalized intermediate's BF16 rounding.
`training/forward_gates` pairs fused sigmoid, SiLU and residual gate sequences
with their separate operations, including retention of the cached activations.
`training/gated_backward_packed` pairs scalar and vectorized backward kernels,
checks exact BF16 agreement, and exports Details-mode compiler reports through
`KREA2_BENCH_REPORT_DIR`.
`training/projection_fused` and `training/projection_fused_rank64` pair complete
adapter forward/backward passes against separate matrix products and additions,
with cached forward activations on both sides. Parameter-gradient accumulation
is included; frozen-base GEMMs are outside these measurements.
`training/projection_residual` and its `_rank64` variant compare folding the
preceding input gradient into that output stage against a separate addition.
`training/projection_cached` pairs complete LoRA forward/backward passes with and
without retaining the rank-sized activation, using the same resident inputs.

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
cargo bench --locked --bench training -- training/narrow
cargo bench --locked --bench training -- training/attention
cargo bench --locked --bench training -- training/attention_pack
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

## Full RAW fine-tuning

`mode: "full"` trains all 430 transformer tensors (12,820,073,036 parameters),
including text fusion, projection biases, RMSNorm scales and modulation tables.
Qwen and the VAE remain frozen and unload after preparation. The cache holds VAE
posterior moments and twelve Qwen hidden-state taps; text fusion is recomputed
with the current weights each microbatch. `mode` defaults to `"lora"` for existing
configurations. Full mode ignores `rank`, `alpha` and `targets`.

A starting configuration for the 128 GiB Strix Halo is:

```json
{
  "mode": "full",
  "model": "/models/krea2_raw_bf16.safetensors",
  "dataset": "/datasets/domain",
  "output": "/runs/domain-full",
  "resolution": 512,
  "steps": 100,
  "accumulation": 1,
  "gradient_checkpointing": false,
  "learning_rate": 0.00001,
  "weight_decay": 0.01,
  "save_every": 50,
  "keep_checkpoints": 1,
  "memory_gib": 112,
  "scratch_pool_mib": 8192,
  "seed": 37,
  "validation_prompts": ["a photograph describing the new domain"]
}
```

The Rust API exposes the same memory/precision starting point as
`TrainConfig::full_preset()`. JSON configuration uses the ordinary defaults for
fields omitted from the file, so include the full-mode settings above explicitly.
The physical batch size is one. This preset retains activations, with an 8 GiB
scratch pool and the existing 8 GiB system RAM reserve; it never silently enables
offloading or recomputation. The startup estimate includes optimizer state,
activation tapes, reduction buffers, and the largest FP32 weight-gradient
accumulator. Longer captions increase the activation requirement.

Captions are optional in full mode. Missing or whitespace-only `.txt` files use
the empty Krea prompt; existing unreadable or invalid UTF-8 caption files are
errors. Mixed captioned and uncaptioned examples are accepted. Captions help bind
new visual knowledge to words; uncaptioned training alone does not establish
that the new concepts can be requested by name. Set representative, fixed
`validation_prompts` for evaluation; loss alone does not establish knowledge or
image quality.

Large matrices use authoritative BF16 weights and accumulated BF16 gradients,
with FP32 optimizer arithmetic and two blockwise 8-bit moments (256 elements per
block, separate signed/unsigned dynamic codebooks and FP32 maxima). There is no
FP32 master copy of large matrices. Small matrices below 4096 elements and all
biases, scales and modulation tables retain FP32 masters, gradients and moments,
plus BF16 execution copies. A single stream-ordered FP32 accumulator is reused
for large weight gradients. Both gradient storage and large-weight updates use
Philox stochastic BF16 rounding keyed by parameter, seed, update and microbatch,
independently of the data-sampling RNG. All weights remain unchanged until the
entire backward pass finishes and the averaged global norm is validated/clipped.

Preparation and training use the existing commands:

```sh
cargo run --release --bin krea2-train -- inspect --config /runs/full.json
cargo run --release --bin krea2-train -- prepare --config /runs/full.json
cargo run --release --bin krea2-train -- run --config /runs/full.json --stop-after 2
cargo run --release --bin krea2-train -- run --resume /runs/domain-full/checkpoints/step-000002
cargo run --release --bin krea2-train -- evaluate --run /runs/domain-full
```

Each full checkpoint contains `model.safetensors`, `optimizer.safetensors`,
versioned `state.json`, and an `artifact.json` provenance manifest. Small model tensors are saved in FP32 and large matrices
in BF16; optimizer files preserve the exact moment bytes and scales. Serialization
uses at most 64 MiB of application readback staging. Files and the temporary
directory are synced before atomic publication, and retention pruning happens
only after the new checkpoint is complete. A checkpoint is about 48 GiB; allow
about 110 GiB for old and new checkpoints plus margin. Startup and each save check
available output space. Failed writes leave previous completed checkpoints intact.
Resume requires matching software/compiler/device, dataset, original base model,
and configuration, as with LoRA training.

Full-mode evaluation opens each saved model as **RAW**, using the RAW defaults
(52 steps, guidance 3.5), fixed seeds 37/38 and the configured prompts. Turbo and
adapter strength are only relevant to LoRA evaluation. Full fine-tuning does not
produce a Turbo-compatible adapter. Per-update EMA, LoRA merging, Qwen/VAE
training and Turbo distillation are outside this implementation. Offline
averaging of saved full-model snapshots is described below.

Validation includes scalar/GPU optimizer and rounding oracles, norm-scale
reductions, small-state exact resume and update rejection, and an opt-in real RAW
one-block/conditioning test. The full 100-step model trial, full-size resume,
peak-memory measurement and RAW renders still require a sufficiently free machine
and output volume; the small fixtures do not establish full-run speed or quality.

```sh
cargo test --test full_training_gpu -- --ignored --test-threads=1
cargo test --lib training::full::parameters::tests -- --ignored
KREA2_RAW_CHECKPOINT=/models/krea2_raw_bf16.safetensors \
  cargo test --lib training::full::parameters::real_model_tests -- --ignored
# Works with either LoRA or full-mode prepared configs:
KREA2_TRAIN_TEST_CONFIG=/runs/full.json \
  cargo test --test training_resume -- --ignored
cargo bench --bench training -- training/full_adamw
cargo bench --bench training -- training/full_weight_gradient
```

The optimizer bench resets weights, gradients and moments outside each timed
interval. The weight-gradient bench measures the FP32 TN product; it excludes
BF16 staging, the rest of backward, and optimizer work. Full optimizer compiler
reports can be saved through `KREA2_BENCH_REPORT_DIR` outside the checkout. Use
`KREA2_TRAIN_BENCH_CONFIG=/runs/full.json cargo bench --bench training -- training/prepared/step`
for the final end-to-end number.

## Model-only snapshots and offline averaging

Full training has independent retention for resumable checkpoints and inference
models. `keep_checkpoints` controls complete resume directories. Add
`"snapshot_every": 50, "keep_snapshots": 2` to the 100-step example above to retain
two model-only snapshots under `snapshots/`. JSON defaults disable automatic
snapshots; `TrainConfig::full_preset()` enables them every 250 optimizer updates
and keeps two. `null` disables the schedule. Pinning protects extra artifacts
outside those retention counts. A snapshot alone cannot resume training.

Snapshots contain the original model precision (BF16 large matrices, FP32 small
tensors), without optimizer or sampler state. Coincident snapshot/checkpoint
saves hard-link the immutable model file where the filesystem permits; otherwise
they copy it with bounded buffers and check available space. They do not read
weights back from the GPU twice. Removing either directory leaves the other
usable. Treat model files as immutable: editing a hard-linked file would change
both views. Exports include provenance, source checksums and the evaluation
configuration; they still require the frozen text encoder and VAE.

```sh
# Use the built target/release/krea2-train binary for these commands.
krea2-train checkpoint list --run /runs/domain-full
krea2-train checkpoint pin --checkpoint /runs/domain-full/snapshots/step-000050
krea2-train checkpoint export \
  --checkpoint /runs/domain-full/checkpoints/step-000100 \
  --output /runs/domain-full/exports/final

# Equal weight per saved model:
krea2-train checkpoint average \
  --checkpoints /runs/domain-full/snapshots/step-000050 /runs/domain-full/snapshots/step-000100 \
  --output /runs/domain-full/exports/average
# Gap-aware exponential averaging of the same snapshots:
krea2-train checkpoint average \
  --checkpoints /runs/domain-full/snapshots/step-000050 /runs/domain-full/snapshots/step-000100 \
  --half-life-steps 50 --output /runs/domain-full/exports/ema
krea2-train evaluate --checkpoint /runs/domain-full/exports/ema \
  --output /runs/domain-full/evaluation/ema
# Remove manual protection when the snapshot is no longer needed:
krea2-train checkpoint pin --checkpoint /runs/domain-full/snapshots/step-000050 --unpin
```

Averaging runs entirely on the CPU in Rust. It accepts 2–16 native snapshots or
resume checkpoints from the same run/software, sorts by optimizer step, and
rejects duplicate steps, incompatible tensor layouts, nonfinite values or
checksum mismatches. Export and average validate full model payloads; listing
checks metadata and length without scanning weights. Exporting an average retains
its provenance; averaged outputs cannot themselves be averaging inputs. Current
artifact commands require the native `artifact.json` manifest and canonical
tensor layout; they do not import arbitrary third-party checkpoint formats.

For snapshot EMA, the first selected model initializes the accumulator. Each
subsequent snapshot uses `beta = 2^(-step_gap / half_life_steps)` and
`ema = beta * ema + (1 - beta) * model`. Half-life is measured in optimizer
updates, independent of accumulation. This is EMA over saved snapshots, not EMA
over every training update, and cannot reconstruct unrecorded intermediate
weights. All accumulation is FP32, with one final conversion to each tensor's
original dtype. Tensor working buffers use at most 48 MiB, plus bounded headers
and metadata. No EMA buffers are allocated during training.

Outputs are synced and published atomically without replacing existing paths.
Retention runs after publication and skips pinned artifacts, artifacts from
other runs, and directories held open by evaluation/export/averaging/resume.
Failed writes preserve prior completed artifacts. A model is about 24 GiB.
Budget roughly **180 GiB of output capacity** for a 48 GiB resume checkpoint,
two 24 GiB snapshots, a 24 GiB average, and the next 48 GiB resume checkpoint
before pruning, plus margin. Hard links can reduce physical usage. Each write
checks its actual additional payload plus 2 GiB free-space margin; pinning more
models or saving more averages requires additional capacity.

CPU tests cover averaging oracles, chunk boundaries, corrupted sources, atomic
publication, pins and active readers. The small GPU fixture verifies that a
model-only save preserves training state and matches resumable model bytes.
Full-size throughput and RAW quality comparisons remain resource-dependent.
Compare the raw final snapshot, a uniform average and a snapshot EMA with the
same prompts and seeds, including both the new domain and general prompts to
check forgetting. Averaging is optional; numerical correctness alone does not
establish a quality improvement.

```sh
cargo bench --bench checkpoint
# Place streamed fixtures on the filesystem to measure; sources are warm-cached:
KREA2_CHECKPOINT_BENCH_DIR=/runs cargo bench --bench checkpoint -- checkpoint/stream
cargo bench --bench checkpoint -- --test
```

These Criterion benches measure FP32/BF16 chunk blending and complete mixed-dtype
streaming averages, including checksums, encoding, fsync and publication. Fixture
construction and cleanup are excluded. They do not establish cold full-model
disk throughput; keep timing reports outside the repository.
