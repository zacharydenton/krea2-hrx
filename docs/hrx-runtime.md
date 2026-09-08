# HRX runtime on gfx1151

The inference path is C ABI → Rust host → public HRX C API → Loom HSACO → gfx1151.
`crates/hrx` uses HRX device, stream, buffer and executable APIs directly. It
packs Loom's scalar/pointer argument ABI and submits custom direct arguments.
Every dispatch/copy has an execution barrier on the shared ordered stream.
Host transfers synchronize that stream; tensors and temporary buffers stay on
the GPU between operations. There is no HIP compatibility layer.

`libkrea2.so` and `libkrea2_pipeline.so` are one cdylib under two names — the
second is a symlink to the first — so loading the block and pipeline APIs in one
process cannot create duplicate runtime ownership: the dynamic linker sees one
soname. Sessions retain
separate pools and buffers. Each weight bundle is validated, mapped read-only,
uploaded once, and exposed as tensor views into one resident GPU allocation.
This avoids a separate HRX allocation and synchronization for every tensor.
HRX initialization already owned by another caller
is respected; the adapter only shuts down initialization that it owns.

All GPU arithmetic is Loom: BF16/fp16 WMMA matrix multiplication, convolutions
through im2col, normalization, activations, rotary embeddings, dense text/VAE
attention, layout transforms, scheduler updates, Sage smoothing/packing/correction,
and the existing INT4 transformer and attention kernels. BF16 auxiliary weights
keep their existing precision. Sage packs four channels per lane with wave
shuffles, processing eight token/head rows per workgroup without LDS barriers.
The V transpose adapts the tiled Loom implementation in `minimax-h3-loom`.

Auxiliary WMMA kernels load and store four contiguous elements at a time when
the shape permits it. Irregular tails retain scalar guards. Operand and result
tiles share LDS after the final workgroup barrier: the 64×64 kernel needs 10 KiB
instead of 18 KiB. Image-sized BF16 projections use 128×64 tiles, and long
reductions with enough output columns use 128×128 tiles to halve repeated input
reads. Selection avoids adding padded rows or columns compared with 64×64 tiles.
The FP32 accumulation order and BF16 rounding boundaries are unchanged.

The host tokenizer uses complete Unicode 16.0 category and canonical-normalization
tables, including algorithmic Hangul composition. It implements the exact Qwen
split expression before byte-level BPE. The tables and embedded Loom source are
generated during development, checked into the repository and compiled into the
library. Neither Python nor a regex/Unicode shared library is needed at inference.
Cache hashes use the included SHA-256 implementation.

## Building and deploying

`scripts/build.sh` needs only cargo; nothing here is compiled by a C or C++
compiler. `libhrx.so` comes from the local HRX checkout. `scripts/runtime.sh`
copies HRX and the compatible HSA provider into `build/runtime` and publishes
them by rename, so a running process keeps its mappings. The loader locates the
provider beside `libkrea2.so`; a standalone process needs no `LD_LIBRARY_PATH`
setting. The C headers under `build/include` are generated from the Rust sources
by cbindgen and are build artifacts, not inputs. Deployment includes both Krea libraries, the `runtime`
directory, ComfyUI's model files (read as they are; the tokenizer and kernel
sources are embedded in the library), and populated caches or `loom-compile`.
The GPU still needs Linux's amdgpu/KFD driver and normal device permissions.
HIP, BLAS and Python are absent from the native dependency graph.

The installed `/opt/rocm` HSA provider cannot satisfy this HRX build's agent
queries. The build uses the compatible provider in `~/.local/rocm-hrx` instead.
Its registration library is matched to the installed ROCm ABI. Tests that load
Torch and HRX together must choose this same provider at process startup;
`scripts/env.sh` does that. This environment setting is for mixed-runtime tests,
not a requirement of the standalone C ABI.

Auxiliary kernels specialize on shape and launch geometry. The geometry must be
part of the compiler metadata: overriding a one-dimensional launch at dispatch
cannot undo a compiler-folded Y index. The tests cover multiple batch heads,
multiple row tiles, wide/tall matrices and partial tails to catch that failure.
Sources and configurations identify persistent caches; artifacts are verified
with SHA-256 before loading. In-memory hits avoid rehashing kernel source on each
operation. A populated auxiliary cache is verified and used without taking the
lock, so a read-only cache directory works; only compilation needs write access.
A failed compilation reports the compiler's stderr tail in the error string.

## Validation

`tests/test_hrx_runtime.cpp` launches a real Loom kernel, validates an interior
buffer span, and inspects `/proc/self/maps`. It requires HRX and HSA and rejects
HIP, BLAS, Torch, Python, ICU, PCRE2 and OpenSSL mappings. Constructor-failure
checks wrap actual HRX buffer allocation/release. The quick suite includes both
checks plus standalone auxiliary arithmetic, main block kernels and Sage oracles.

The tokenizer is checked where each implementation lives: `cargo test` covers the
Rust one, including that the file's NFC normalizer is applied and that Krea's
template costs the 34 tokens the encoder later strips, and
`tests/test_native_pipeline.py` compares the shipping one with HuggingFace's
`AutoTokenizer` through `krea2_tokenize`. The native model suites compare
components, the full transformer and tiled VAE to the installed references and
verify the real CUDA scheduler's BF16 rounding.

## Local performance and image comparison

Prompt: `a red fox in the snow`, 1024×1024, eight steps, seed zero. No other GPU
inference process was visible during the paired native benchmarks. CPU build work
ran during part of the baseline sequence, so these are local measurements rather
than controlled hardware characterization.

| Build | Model load | First generation | Warm generation |
| --- | ---: | ---: | ---: |
| Saved HIP build | 1.99 s | 19.89 s | 19.33 s |
| HRX, per-tensor weight allocations | 6.31 s | 19.52 s | 18.41 s average of two |
| HRX, one allocation per bundle | 1.43 s | 19.38 s | 18.49 s |

The final weight-layout change preserves the 1024² RGB SHA-256 exactly:
`65507120ecf9fca0ccfb96d3a1ed7d9fc95e087abb4f7a0aed2be75cb21e5f09`.
The initial HRX process also paid new-shape preparation and took 24.20 s for its
first image. These native benchmarks use the native RNG; their noise differs from
the shared-noise UI comparison below.

UI job `b090e5212a974313` uses identical float32 initial noise in both backends.
ComfyUI INT8 ConvRot generation took 53.73 s; native HRX took 19.36 s. This job
predates the final allocation consolidation, which changes loading and storage
layout but preserves image bytes. The pair's image PSNR is 18.96 dB and mean RGB
absolute difference is 14.88/255. This is one prompt compared to ComfyUI INT8,
not a BF16 quality sweep. Both full-resolution PNGs are available in the UI;
composition is similar and detail differs. History and measurements are stored
under `build/comparison-ui/b090e5212a974313`.

## Auxiliary GEMM optimization

`tools/bench_native_gemm.py` compiles the current kernels and a selected Git
revision, checks bit-identical results on signed BF16 inputs, then alternates
their execution order on the same resident GPU buffers. It reports paired
speedup and latency percentiles; compilation, allocation and transfers are
outside the timed region. These are kernel measurements, not image latency.

```sh
scripts/build.sh
env -u LD_LIBRARY_PATH .venv/bin/python tools/bench_native_gemm.py --baseline 2870077
```

The September 6 run used 80 alternating rounds per shape. Another H3 video job
shared the GPU, so the observed improvements are estimates under contention.
The [raw results](benchmarks/aux-gemm-2026-09-06.json) include latency percentiles.

| GEMM M×N×K | Median paired speedup | Output |
| --- | ---: | --- |
| 64×6144×2560 | 1.37× | bit identical |
| 4096×6144×64 | 3.37× | bit identical |
| 65536×256×2304 | 1.87× | bit identical |
| 16384×512×4608 | 1.49× | bit identical |

The end-to-end timing sweep was stopped after the saved baseline took 105 seconds
for a previously ~19-second image under the competing workload. No new full-image
speedup is claimed from that sweep. The final build passes
`scripts/test.sh --quick --native`, including the independent pipeline references,
all three GEMM tile shapes with irregular tails and batched heads, real HRX
dependency checks, and all 5,050 CUDA scheduler steps.
The final 1024², eight-step, seed-zero fox image is byte-identical to `2870077`:
SHA-256 `65507120ecf9fca0ccfb96d3a1ed7d9fc95e087abb4f7a0aed2be75cb21e5f09`.

The optimization retains BF16 auxiliary weights and existing quantization in
the main transformer. Larger INT4 workgroup tiles adapted from the sibling
GEMM/H3 work were also tested: their gains varied by projection and were not
adopted. Removing raster padding alone did not produce a consistent gain.
Carrying the next Sage key scale and correction in registers preserved output
but slowed the 4K-token attention kernel from 13.97 to 15.55 ms median; carrying
only the scale also lost. The existing attention schedules remain selected.
These local experiments do not establish a state-of-the-art ranking.

A later down-projection sweep measured a 1.06×–1.09× kernel speedup, but an
integrated warm generation changed its checksum. The subsequent investigation
reproduced the fault in production VAE decoding alone and isolated a shared-memory
race in auxiliary softmax. Maximum and sum reductions now use separate LDS
regions. The wide down kernel still awaits clean whole-image timing before
selection; see [the experiment report](down-gemm-experiment.md).
