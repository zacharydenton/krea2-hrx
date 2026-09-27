# Experimental kernels

Kernels kept for investigation. They are not embedded in the library or
selected by any build.

- `attention_query32.loom`: fp16 attention over 32 query rows per workgroup. It
  was 1.76–1.78× faster at 4115 tokens but failed the image-quality gate, and
  the pinned HRX 0.8 compiler now rejects its accumulator repack
  (`AMDGPU/041`). `tests/quantized.rs` records that rejection; see
  [the evaluation](../docs/attention-2x.md).
