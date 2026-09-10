# NPU text-fusion pilot

`--fusion-backend auto|gpu|npu` selects the backend for
`txtfusion.layerwise_blocks.0.mlp.up`. Other operations use the existing GPU path.
The library exposes the same choice through `PipelineOptions`.

Auto is the default. Without a matching, passing local qualification it uses the
GPU without opening an NPU or compiling a kernel. `gpu` forces the existing
implementation. `npu` requires a verified compiled artifact and reports errors;
it bypasses the performance gate for explicit qualification. Device execution
errors propagate once offload starts.

Build with `--no-default-features` to omit NPU support. Such a build accepts Auto
and GPU, and rejects forced NPU selection.

## Operation and lifetime

The pilot keeps native BF16 inputs and weights, accumulates into F32, then applies
the original optional bias and BF16 rounding in a GPU epilogue. It does not use
BFP16 emulation. The checked-in IRON generator and tile sources are in
`native/npu`; their upstream attribution and license are beside them. Chess and
its license are supplied separately.

The generator uses a 4-column XDNA2 array with 64×64×32 tiles. M is padded to a
multiple of 512; K must be divisible by 64 and N by 128. Unsupported shapes use
GPU in Auto mode. Each specialization retains resident NPU weights, shared
input/F32 output, a GPU epilogue and one prepared HRX graph. Input and output
copies use `Runtime::with_gpu_access` on the pipeline's existing stream. Completed
stage timing includes both copies, cache maintenance, graph submission and waits.

The cache holds at most two shapes and 512 MiB of tracked buffer storage in total.
Eviction is least recently used and occurs only after execution has completed.
This limit excludes driver/program metadata, compiler memory and the existing
GPU model workspace. No background compilation or first-use benchmarking occurs.

## Compile and qualify

Use the installed MLIR-AIE/IRON environment and Chess compiler. For the local
DINO toolchain described in this workspace:

```sh
source ../dinov3-xdna2/env.sh
python3 ../hrx.rs/scripts/pin-npu-toolchain.py \
  --python ../dinov3-xdna2/.toolchain/bin/python \
  --aiecc ../dinov3-xdna2/.toolchain/bin/aiecc \
  --backend Chess --identity-root "$HOME/tools/rai/bin" \
  --output build/fusion-toolchain-chess.json
python3 scripts/qualify-fusion.py --toolchain build/fusion-toolchain-chess.json
```

The Chess intrinsic wrapper must match the installed compiler. The recorder
preserves the environment's wrapper precedence and hashes the installed compiler
files; it does not install or modify the toolchain. Keep the resulting local
manifest private because it contains local paths and license environment settings.

Qualification requires the existing immutable `build/quality` fixture and its
checked-in `tests/fixtures/unquantized.json` manifest. `--fixture`, `--manifest`
and `--checkpoint` select other existing inputs. It refuses baseline minting.
The command captures real fusion activations, compiles the specialization, checks
changed-input graph replays against a scalar f64 oracle, and measures five fresh
processes with 10 warmups and 100 alternating completed-stage samples per backend.
It then runs the unchanged reference quality gate and warm production
`Pipeline::generate` measurements in five fresh processes per backend.

Selection requires all of the following:

- At least 5% lower median completed-stage latency, using the median of the five
  process medians; the median process p95 must be no worse than GPU.
- Median full-generation time no more than 5% slower than GPU.
- The existing relative-RMS and image-PSNR gates, each allowing at most 0.1 dB
  regression against the pinned accepted baseline.
- Matching shape, checkpoint and weight hashes, implementation/dependency
  identity, compiled artifacts, runtime bundles, hardware, driver, firmware and
  power configuration.

Evidence and logs stay in `build/fusion-qualification/run-*`. Profiles are saved
atomically under `$XDG_CACHE_HOME/hrx/krea2-fusion` (or `~/.cache/hrx/krea2-fusion`).
Use `--profiles` when qualifying and `KREA2_FUSION_PROFILES` when running to select
another directory. This is a trusted local compiler-output store: hashes detect
changes, but do not make arbitrary native artifacts safe. Runtime/library
provisioning overrides prevent qualification. A slower candidate is recorded as
unqualified, so Auto continues using GPU.

Qualification is checked when a shape first enters a pipeline's cache. Restart
the pipeline after changing profiles or power configuration. Read
`Pipeline::fusion_selection()` for the selected backend or fallback reason; the
CLI prints it after generation unless `--quiet` is set.
