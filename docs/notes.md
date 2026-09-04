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

## First end-to-end numbers (fixture: 4115 tokens, timestep 1.0)

The native 28 blocks against the reference, cosine of the update the blocks make to
the residual stream (the stream's own cosine hides everything behind the identity):

| blocks | Loom vs reference W4A4 | Loom vs bf16 | reference W4A4 vs bf16 |
| ---: | ---: | ---: | ---: |
| 1 | 0.99943 | 0.99899 | 0.99908 |
| 2 | 0.99904 | 0.99862 | 0.99887 |
| 4 | 0.99872 | 0.99833 | 0.99882 |
| 8 | 0.99720 | 0.99684 | 0.99781 |
| 16 | 0.97783 | 0.97912 | 0.98124 |
| 28 | 0.96841 | 0.96815 | 0.97169 |

The Loom blocks track the same-arithmetic reference at the rate that arithmetic
tracks bf16, so the decay is the quantisation error compounding through the depth,
not a kernel fault. Whether 0.97 after 28 blocks is acceptable is decided at the
image level (latent PSNR and the pictures), below.

Stage profile, 28 blocks, before the attention work:

| stage | ms | share |
| --- | ---: | ---: |
| attention | 10839 | 86.6% |
| gemm gate\|up | 658 | 5.3% |
| gemm down + residual | 388 | 3.1% |
| gemm qkv\|gate | 304 | 2.4% |
| gemm wo + residual | 127 | 1.0% |
| prepare (three kinds) | 175 | 1.4% |
| qk norm + rope, v transpose | 24 | 0.2% |

The four GEMMs run at 31-49 TOPS at M = 4115 (a tail tile row and the epilogue's
token scale); the whole block minus attention is 60 ms. Attention at 390-560 ms per
block is the entire problem: the 16-key online-softmax kernel pays two cross-lane
reductions, an LDS round trip and two barriers per 16 keys.

## The quantisation is acceptable

Same seed, same prompt, 8 Turbo steps: the reference W4A4 (per-row weights, per-token
activations, group-256 Hadamard) against bf16 gives latent PSNR 9.96 dB and image
PSNR 18.9 dB, and the two pictures (build/bf16_seed0.png, build/w4a4_seed0.png) are
the same fox in the same pose and light, differing only in fur and snow detail, with
no artifacts. For a distilled 8-step model that is sample-level variation, not
degradation. No finer scale granularity is needed.

## Attention: what the 32-key variant showed

Processing 32 keys per iteration (half the reductions and barriers) lost, 0.885x
(155 vs 137 ms at 4115 tokens), so the per-tile softmax overhead is not the cost.
The loads are: every lane fetches 16-byte pieces of rows 3 KB apart, and the four
query heads sharing a key-value head each stream the whole K and V. 137 ms is 3
TFLOP/s. The next kernel stages each 16-key K and V tile once into LDS for a
workgroup made of the four query heads of one key-value head, with coalesced loads.

## Attention staged in LDS: 3.19x

kernels/attention_gqa_lds_f16_wmma.loom: one workgroup of four waves per (16 query
rows, key-value head), each wave one of the four query heads that share that K/V.
Per 16-key tile the workgroup loads the K and V tiles once, coalesced (each lane 32
contiguous bytes), into LDS; K's rhs fragments are assembled from the tile rows (lane
= key, 16 contiguous channels), V's are dense fragment loads from the same tile rows
(the tile's [key][channel] layout is the rhs' logical [k][n]; a transposed view here
was the one bug). The result publishes one fragment at a time through 1 KB per wave,
so the LDS is 14.8 KB and four workgroups fit per CU; 216 VGPRs, no spills. The V
transpose kernel is gone: the QK-norm/RoPE kernel now writes q, k and v contiguous.

| kernel, 4115 tokens | ms | TFLOP/s |
| --- | ---: | ---: |
| 16-key, operands from global | 136.9 | 3.0 |
| 32-key, operands from global | 155.2 | 2.7 |
| LDS-staged, four heads per tile | 43.0 | 9.7 |

Stage profile, 28 blocks, after: attention 1261 ms (42.8%), gemm gate|up 672
(22.8%, 69 TOPS), down 391 (13.3%, 59 TOPS), qkv|gate 304 (10.3%, 72 TOPS), wo 126
(4.3%, 69 TOPS), the prepares 174, QK-norm+RoPE 20. A forward of 28 blocks: 7.95 ->
2.96 s. Correctness unchanged (update cosine 0.96841 vs the W4A4 reference).

## End to end in Loom (first run, before the staged attention)

Same seed as the baseline: latent PSNR 10.09 dB / image 19.0 dB against bf16 (the
W4A4 reference is 9.96 / 18.9), and 12.95 dB / 22.1 dB against the W4A4 reference,
so the native pipeline lands in the same quality class as the arithmetic it
implements. 7.85 s per forward then; the picture is build/loom_seed0.png.

## Prefetching the next K/V tile into registers -- lost

Carrying the next tile's two chunks per lane across the loop so their loads overlap
the WMMAs: 248 VGPRs, 0.929x (43.4 vs 40.3 ms). Occupancy beats the overlap here, as
the 4-wave GEMM did. Kept under experiments/.

## End to end with the staged attention

29.2 s for the 1024^2 image at 8 Turbo steps (3.65 s/step including the text encoder,
the torch-side embeddings and the tiled decode; 2.89 s per forward of the 28 blocks),
against 65.3 s for the official bf16 pipeline on the same box: 2.24x, with the PSNRs
unchanged to the last digit (19.04 dB against bf16, 22.13 dB against the W4A4
reference). A published int4 ConvRot checkpoint of this model that keeps 96 of 224
block linears in int8 reports 18 s per image on an NVIDIA L4 at a minimum 17.7 dB.

## GEMM raster group per token count -- won (3-4% on the forward)

At 4115 tokens the 33 tile rows were padded to 36 for the group-of-4 raster and the three
ghost rows ran the whole k-loop on zeros. The group is now a config (`m_group`, chosen by
the builder and the host by the same rule: of 4, 3, 2 the one that pads least, ties to
the larger), so 33 rows raster in groups of 3 with no ghosts. Interleaved A/B, 28 blocks:

| | group 4 | group 3 |
| --- | ---: | ---: |
| gemm gate/up | 670 / 655 ms | 603 / 621 ms |
| gemm qkv/gate | 304 / 302 | 280 / 282 |
| gemm wo | 126 / 126 | 117 / 117 |
| gemm down (K=16384) | 403 / 403 | 388 / 429 |
| forward | 2936 / 2945 | 2809 / 2872 |

The down GEMM's spread (its 1 MB W tiles are the L2-sensitive case) is the box's noise;
the other three gain 7-8%, the whole forward 3-4%. `KREA2_M_GROUP` overrides both sides
for A/Bs.

## Attention round two: what actually stalled the staged kernel

Instruction histogram of the 16-key loop (per tile per wave, 483 instructions around 16
WMMAs) and four generator toggles, each A/B'd at 4115 tokens (noise < 1%):

| change | kernel ms | TFLOP/s | VGPRs | verdict |
| --- | ---: | ---: | ---: | --- |
| shipped this morning (subgroup reduces, V rhs from the tile view) | 40.4 | 10.3 | 226 | baseline |
| xor-butterfly row max over the 16-lane half, per-lane partial sums | 41.3 | 10.1 | 248 | neutral: kept for the register drop below |
| V staged transposed [channel][key] in LDS, rhs from two wide loads | 37.0 | 11.3 | 208 | won 1.09x: the V `fragment.load<rhs>` from LDS was 128 two-byte loads per tile |
| two accumulator chains for the scores; no weight mask | 37.4 | 11.1 | 208 | neutral (mask removal kept) |
| Q: 4 of 8 fragments hoisted into registers | 28.5 | 14.6 | 240 | won 1.31x: the per-tile Q re-fetch from global was the stall |
| Q: all 8 hoisted (spills 76 B) | 34.3 | 12.1 | 256 | 1.08x, spills |
| Q: 4 hoisted + other 4 staged once per wave in LDS, result stage aliased onto the K/V tiles | 20.9 | 20.0 | 240 | **won 1.39x more; shipped** |
| Q: 0 hoisted, 8 from LDS | 22.1 | 18.8 | 208 | 0.94x of the above |
| Q: 5 or 6 hoisted | 21.2 / 40.7 | | 248 / 256 | worse; 6 spills |

Cumulative: 40.4 -> 20.9 ms on the kernel (1.93x), 28-block forward 2949 -> 2313 ms;
attention is 28% of the forward, the four GEMMs 63%. LDS 21760 B per workgroup of four
waves. The lesson repeats dinov3's: a `fragment.load` from a view whose contiguous axis
is not the fragment's k axis lowers to per-element loads, on LDS as on global; assemble
the operand from wide loads yourself.

## Prepare kernels: 8 elements per lane -- won (177 -> 98 ms over 28 blocks)

The three prepare kernels loaded f16 two bytes per lane and stored int4 one byte per
lane, and reached 60-115 GB/s. They now move 8 elements per lane per step: 16-byte global
loads, vector arithmetic (`vector.extf/expf/absf/maxnumf/roundevenf/fptosi/andi`), 32-byte
LDS row stores, and the 8 nibbles packed into one 4-byte store. The Hadamard stages are
unchanged. Codes match the reference exactly as before (ties aside); the block-level
cosines are identical through 4 blocks and drift by 0.003 at 28, the tie-breaking noise
of the quantiser compounding.

| stage | before | after |
| --- | ---: | ---: |
| prepare swiglu | 89.4 ms | 48.0 ms |
| prepare norm | 58.4 | 32.9 |
| prepare gated | 28.8 | 17.3 |
| forward | 2313 | 2129 |

The gate stride config only needs 8-element alignment (the qkv|gate stride is 15360).

## 32-key attention tile -- lost (0.957x)

Two score fragments per tile, one butterfly and one rescale per 32 keys, two probability
fragments through a 1 KB scratch, sixteen matrix multiplies each way. With four Q
fragments hoisted it spills (256 VGPRs, 140 B); with none hoisted it fits at 248 VGPRs
but needs 40 KB of LDS (three workgroups per WGP instead of five). Interleaved A/B at
4115 tokens: 24.65 vs 23.60 ms, 0.957x, and its output deviated from the reference
(cosine 0.9987 against 0.99999996), which was not chased since it loses regardless. The
generator keeps the `ATTN_TILE=32` path as the record of what was tried; the shipped
16-key output is byte-identical. The lesson: at 240 registers the kernel is occupancy
bound, and any lever that adds LDS or registers per wave has to pay for itself twice.
