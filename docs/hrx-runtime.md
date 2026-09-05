# HRX runtime on gfx1151

The inference path is C ABI → C++ host → public HRX C API → Loom HSACO → gfx1151.
`host/gpu.cpp` uses HRX device, stream, buffer and executable APIs directly. It
packs Loom's scalar/pointer argument ABI and submits custom direct arguments.
Every dispatch/copy has an execution barrier on the shared ordered stream.
Host transfers synchronize that stream; tensors and temporary buffers stay on
the GPU between operations. There is no HIP compatibility layer.

`libkrea2.so` owns the shared runtime, block implementation and auxiliary-kernel
cache. `libkrea2_pipeline.so` links that core, so loading the block and pipeline
APIs in one process does not create duplicate runtime ownership. Sessions retain
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

The host tokenizer uses complete Unicode 16.0 category and canonical-normalization
tables, including algorithmic Hangul composition. It implements the exact Qwen
split expression before byte-level BPE. The tables and embedded Loom source are
generated during development, checked into the repository and compiled into the
library. Neither Python nor a regex/Unicode shared library is needed at inference.
Cache hashes use the included SHA-256 implementation.

## Building and deploying

`scripts/build_native.sh` uses a normal C++ compiler. HRX headers and `libhrx.so`
come from the local HRX checkout. The build copies HRX and the compatible HSA
provider into `build/runtime` and publishes libraries by rename. The C++ loader
locates the provider beside `libkrea2.so`; a standalone process needs no
`LD_LIBRARY_PATH` setting. Deployment includes both Krea libraries, the `runtime`
directory, the exported model bundle, and populated caches or `loom-compile`.
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
operation. The auxiliary cache currently requires a writable lock file.

## Validation

`tests/test_hrx_runtime.cpp` launches a real Loom kernel, validates an interior
buffer span, and inspects `/proc/self/maps`. It requires HRX and HSA and rejects
HIP, BLAS, Torch, Python, ICU, PCRE2 and OpenSSL mappings. Constructor-failure
checks wrap actual HRX buffer allocation/release. The quick suite includes both
checks plus standalone auxiliary arithmetic, main block kernels and Sage oracles.

`tests/test_unicode.py` checks SHA-256 block/padding boundaries, Unicode NFC and
mixed-script tokenizer inputs against independent Python implementations. The
native model suites compare components, the full transformer and tiled VAE to
the installed references and verify the real CUDA scheduler's BF16 rounding.

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
