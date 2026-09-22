# Native Loom NPU text fusion

The `npu` feature enables a native Loom/XDNA implementation on HRX 0.8.3.
[Measured results](npu-fusion-native-results.md) show a working but much slower
candidate; automatic selection stays on GPU.
It replaces the retired Chess/IRON execution path for
`txtfusion.layerwise_blocks.0.mlp.up`. Other operations remain on GPU.

```sh
cargo build --release --features npu
cargo run --release --features npu -- --fusion-backend npu \
  -p "a red ceramic cup on a wooden table" --out cup.png
```

No Chess license, IRON environment, Python compiler or vendor SDK is needed.
Loom compiles the checked-in `kernels/native/fusion*.loom` sources into the
canonical native image and GPU handoff kernels using HRX's verified bundle.
The AMD XDNA driver and accessible NPU are required for forced NPU execution.

`auto` and `gpu` use GPU without opening the NPU or compiling a native image.
`npu` is an explicit experiment; compilation and device errors propagate.
Without the Cargo feature, forced NPU returns a rebuild instruction.
Legacy qualification profiles are never read. `Pipeline::fusion_selection()`
reports the selection and reason.

## Execution and limits

Inputs and weights remain BF16. Native 8×8×512 outer products accumulate in
FP32; a GPU kernel reduces the K blocks, applies optional BF16 bias, and rounds
to BF16. This does not use BFP quantization. Reduction ordering differs from GPU,
so successful arithmetic checks do not establish full-model quality parity.

M, K and N tails are zero-padded. Width is processed in chunks of at most 256
output tiles to respect the shim DMA repeat limit. One native worker and one
prepared native run are reused across the projection: larger worker layouts
exhausted compiler stream routing, and a separate native run for every matrix
tile exhausted hardware contexts. Zero-offset shared staging buffers satisfy the
native image's binding contracts. These constraints and the required transfers
make this a correctness experiment, not an optimized matrix implementation.

A projection retains packed weights, activations, partial sums and staging.
One specialization is cached per model; changing the activation shape evicts it
before replacement. Execution is serialized across the cache, and fresh input
is packed on every call. GPU/NPU transfers declare access through HRX's
`with_gpu_access`; all handoffs and outputs complete before the operation returns.

Dimensions are bounded to M≤4096, K≤16384 and N≤32768, with a stricter 512 MiB
cap on padded data allocations. The cache inherits the model stream's residency
budget. The data cap excludes compiler, executable and driver metadata, CPU
weight download scratch, and the existing GPU model. Auto never allocates this
cache.

## Reproduce arithmetic and performance checks

Small deterministic cases check all outputs against a scalar f64 oracle,
including unaligned dimensions, split K, output chunk boundaries, bias and
changed-input replays:

```sh
cargo run --release --features npu --example fusion_npu
```

Capture a real stage input while measuring GPU generation, then benchmark that
immutable capture. Capture files are diagnostic inputs, not quality references.
Use a fresh capture directory for a different checkpoint or prompt.

```sh
KREA2_FUSION_CAPTURE=build/native-fusion/cases \
  cargo run --release --features npu --example bench_runtime -- \
  build/native-fusion/gpu.rgb gpu
cargo run --release --features npu --example fusion_npu -- \
  build/native-fusion/cases/156x2560x6912
cargo run --release --features npu --example bench_runtime -- \
  build/native-fusion/npu.rgb npu
```

The stage benchmark validates capture hashes, checks 1,024 scalar-f64 output
samples on original and changed inputs, compares full outputs with GPU, and
checks that warm execution creates no tracked allocations or imports. Each
process performs ten warmup pairs and 100 pairs with alternating backend order.
Run five fresh processes for a timing comparison. Timing includes input packing,
all staging/copies, coherency, NPU execution, GPU reduction/bias and completion;
compilation and correctness readback are outside the interval.

Full quality qualification requires the original immutable `build/quality`
fixture and `tests/fixtures/unquantized.json`. Run the existing ignored
`unquantized_bf16_reference_quality_does_not_regress` test with
`KREA2_QUALITY_FUSION_BACKEND=gpu`, then `npu`, without minting a new baseline.
The latency requirement remains ≥5% lower completed-stage median with no worse
p95, ≤5% full-generation slowdown, and no more than 0.1 dB regression in either
existing latent/image quality gate. No automatic profile loader is enabled.

The [historical HRX 0.7 experiment](npu-fusion-legacy.md) and its retained
`native/npu` sources are archival and are not used by this implementation.
