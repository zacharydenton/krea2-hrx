# krea2-loom notes

Krea 2 Turbo's 28 transformer blocks (12.9B single-stream DiT, width 6144, 48/12 heads
of 128, SwiGLU 16384) in Loom on the Radeon 8060S, W4A4 ConvRot. Text fusion, the
embeddings, the final layer, the text encoder (Qwen3-VL-4B), the scheduler and the VAE
stay in torch through diffusers' official Krea2Pipeline; only `blocks_forward` moves.

## Decisions

- **The oracle is a transcription, validated structurally.** reference/krea2_ref.py is
  diffusers' Krea2Transformer2DModel rewritten onto the ComfyUI checkpoint key names
  (the format of every Krea 2 file on this box). tests/test_ref_vs_diffusers.py builds
  both at toy size with the same random weights and they agree to the last bit, so the
  transcription is trusted at full size without a diffusers-format download. Its
  `quant="w4a4"` mode is the exact arithmetic the kernels implement (group-256 Hadamard,
  int4 per row / per token, absmax/7), so kernel tests compare against it exactly.
- **Text tokens are compacted, not masked.** The pipeline pads prompts to a fixed 512
  tokens with a key mask; masked keys contribute nothing and their rows are dropped, so
  keeping only the valid text tokens is exact and removes both the mask from attention
  and ~450 tokens per forward. The fixture from a real prompt has 19 text + 4096 image
  tokens.
- **Fused projections.** q|k|v|gate is one GEMM (N = 15360) and gate|up one GEMM
  (N = 32768); attention reads q, k and the output gate at column offsets of the fused
  buffer with a 15360 stride, and V is transposed once per block for the attention's
  operand layout. Ten launches per block: prepare, qkvg GEMM, QK-norm+RoPE, V transpose,
  attention, gated prepare, wo GEMM (+ pregate residual), prepare, gate|up GEMM, SwiGLU
  prepare, down GEMM (+ postgate residual).
- **Every GEMM input is made by a "prepare" kernel**, one workgroup per token: form the
  row (norm+modulate | sigmoid gate | SwiGLU), Hadamard it in LDS as four radix-4
  stages, absmax, int4 pack, f32 scale. Exact against the reference quantiser (0%
  of codes differ; the 16384-wide SwiGLU row lives in LDS as f16 and differs by one
  code on 0.07% of ties).
- **Attention operands are assembled by hand.** The transposed-view K fragment lowered
  to 128 two-byte global loads per key tile (1.4 TFLOP/s); K's natural row layout is
  already the rhs lane layout (two 16-byte loads), V gets a one-time transpose, and Q
  reloads per tile because holding all eight fragments spilled at 256 VGPRs.

## Things that bit

- diffusers instantiates the transformer in f32 on the CPU before any cast: 52 GB of
  host RAM on top of the 26 GB checkpoint, an OOM kill. Build on the meta device, attach
  the bf16 tensors with assign=True, load the safetensors straight to the GPU.
- The Qwen-Image VAE decode of a 1024^2 image without tiling ran for 15+ minutes at
  100% GPU on this stack. `enable_tiling()`; compare latents, decode once.
- `pkill -f <pattern>` matches the shell that issued it when the pattern is in its
  own command line. Kill by pid.
- The prover rejects `index.rem` by a launch argument and lane-started loops
  (`scf.for %i = [%lane to ...]`) inside a multi-wave workgroup; state the bound with an
  assume and loop over a uniform count with the lane added inside. A divergent
  `scf.if` before a loop is rejected too; put it after the last loop.
