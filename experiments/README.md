# Experimental kernels

These kernels are not embedded in the library or selected by production builds.

- `attention_query32.loom`: an accumulator-repack experiment rejected by the
  pinned compiler with `AMDGPU/041`. `tests/quantized.rs` checks that diagnostic.
  It is distinct from the production kernel that shares K/V staging between
  two groups of sixteen queries.
- `sage/`: retired smoothed int8/int4 QK attention and its preparation kernels.
- `npu-fusion/`: retired XDNA2 text-fusion projection with GPU packing and reduction.

Retired host code and investigation history are available in Git. Keep new
experimental harnesses and results outside this checkout; repository benchmarks
belong in Criterion suites under `benches/`.
