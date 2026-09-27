# Experimental kernels

Kernels kept for investigation. They are not embedded in the library or
selected by any build.

- `attention_query32.loom`: fp16 attention over 32 query rows per workgroup. It
  was 1.76–1.78× faster at 4115 tokens but failed the image-quality gate, and
  the pinned HRX 0.8 compiler now rejects its accumulator repack
  (`AMDGPU/041`). `tests/quantized.rs` records that rejection; see
  [the evaluation](../docs/attention-2x.md).
- `sage/`: SageAttention-style smoothed int8 and int4 QK attention (the
  `attention_sage_*` kernels) and their preparation: key and query means, the
  per-token quantizers and the `gemm_f16_f32_nt` mean-correction GEMM. On
  gfx1151 int8 WMMA runs at the fp16 rate, so int8 QK bought nothing; int4 QK
  made attention 1.32x faster (the ceiling when only QK is quantized) but the
  preparation took back about 40% of that, leaving a step about 3% faster for
  2-3 dB less latent PSNR. The host side (`src/session/sage.rs`, the `--attn`
  option and `KREA2_ATTN_QK`) was removed; it is recoverable from Git.
- `npu-fusion/`: the XDNA2 NPU backend for the first text-fusion up
  projection (`fusion.xdna.loom` with its GPU pack and reduce kernels) and its
  write-ups. Automatic selection always chose the GPU: the NPU path never passed
  its latency and quality gates. The host side (`src/fusion`, the `npu`
  feature, `--fusion-backend` and `PipelineOptions`) was removed on 2026-09-28,
  along with the older Chess/IRON generator under `native/npu`.
