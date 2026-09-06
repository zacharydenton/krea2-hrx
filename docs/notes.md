# krea2-loom notes

Development history for Krea 2 Turbo on the Radeon 8060S: 28 transformer blocks,
12.9B single-stream DiT, width 6144, 48/12 heads of 128, SwiGLU 16384, W4A4 ConvRot.
The early measurements below used Torch outside the Loom blocks. The current
native pipeline also implements text encoding, embeddings, scheduler and VAE
through the C ABI; see the native sections below and the README.

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
the other three gain 7-8%, the whole forward 3-4%. `KREA2_M_GROUP` overrode both sides
for these A/Bs; the override was removed once the builders chose the group automatically.

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

## Where a step goes outside the blocks

Per-call timers (`KREA2_TIMING=1 tools/pipeline.py --backend loom --images 2`), second
image of the process: text_in + image_in 13 ms, time embedding + rope tables 13 ms, block
modulation 1 ms, the Loom forward 2.32 s of which the host copies are under 10 ms, final
layer 5 ms. The whole image is 21.9 s: 8 x 2.35 s of blocks, about a second of text
encoding, about two of tiled decode. The first image of a process took 189 s: 6 s of
session build and, once per machine, MIOpen's kernel search for the VAE's convolutions.
The suspected per-step glue was that first call's session build averaged over the steps.

## Lazy accumulator rescaling and exp2 -- lost to registers

FlashAttention-3's trick: leave the running max where it is unless a row's max grows
by more than 2^8, so the 64 accumulator multiplies and 8 exponentials per tile run on a
handful of tiles instead of all 258. In Loom the skip is an `scf.if` yielding the eight
accumulators, and a branch that yields the accumulators keeps both versions live:

| variant | VGPRs | spill | vs shipped |
| --- | ---: | ---: | ---: |
| conditional, 4 Q fragments hoisted | 256 | 112 B | 0.595x |
| conditional, 2 hoisted | 256 | 12 B | 1.030x |
| conditional, 0 hoisted | 248 | 0 | 0.903x |
| exp2 with log2(e) folded into the scale, no conditional | 240 | 0 | 0.988x (noise) |

The condition itself needs no vote: the 16 lanes of a row group see the same maxima
after the butterfly, so a per-lane compare is consistent (the vote's uniform scalar is
rejected as a branch mask anyway). Both levers are dropped; the kernel stays at 240
VGPRs with four Q fragments resident, which every variant so far has confirmed is the
binding constraint.

## SwiGLU in the gate|up GEMM epilogue -- neutral (0.995x), kept

The gate and up weight rows are interleaved in 16-row groups by `tools/export_weights.py`
so each wave's fragments fj = 0,1 (and 2,3) are the gate and up of the same 16 outputs;
`kernels/gemm_i4_swiglu.loom` stages both through a 2 KB per-wave result stage and writes
f16(silu(g) * u) to a [T][16384] output, and `prepare_plain_i4` takes the row as is. The
one-block cosine against the reference is unchanged (0.99943). Interleaved kernel A/B
(`tools/ab_gu.py`, 4115 tokens, GEMM + prepare, the box shared with another session's
GPU job): 29.77 ms unfused (27.97 + 1.81) against 29.91 fused (28.68 + 1.24): the epilogue's
sigmoid per output costs about what the halved 270 MB write saves, the prepare halves.
Kept: exact, 270 MB per block less traffic, and silu on the unrounded f32 products.

Measurement note for this evening: another session's GPU job inflated every stage of the
block profile by 1.3-1.7x (attention, unchanged, read 705-847 ms against 562 on a calm
box); the interleaved kernel A/Bs are the only numbers that survived that.

## Review corrections and validation

Kernel bundles now include source/configuration fingerprints and immutable launch
metadata. The native host uses that metadata instead of rereading the raster-group
environment variable. ABI 2 also exposes a contiguous block range for verification.
Constructor failures release allocations and modules, and the host build includes
the test runner.

Plain preparation divides each radix-4 stage by four before storing fp16 values and
restores the orthonormal factor in the quantization scale. This bounds intermediates
by the input magnitude. Constant 5000, maximum-magnitude signed Hadamard basis rows,
negative 65504 and zero all pass the GPU regression without nonfinite scales. The
32-key attention generator's first reduction now includes both score fragments;
its 100-token test improves from cosine 0.99776765 to 0.99999996.

The Python adapter preserves input tensors, processes batch masks individually,
recreates sessions when token counts change, preserves fp32 norm weights in the
baseline loader, and enables VAE tiling. Quality scripts stop on failures and remove
stale completion markers before starting. Historical image PSNR figures above used
the earlier baseline loader and need remeasurement.

The old full-depth 0.99 comparison conflated kernel accuracy with divergence of
independent quantized trajectories. Merely switching the reference to fp16 was
insufficient. A Torch oracle with explicit native fusion/storage boundaries improved
one-block agreement to 0.999947, but independent 28-block trajectories still reached
only 0.96886. The test now checks every block on the same native input with the
unchanged 0.99 threshold, and requires bit-exact agreement between composing those
native blocks and running the complete stack. All 28 blocks pass (minimum cosine
0.996997); composition is exact at both requested depths, 1 and 28. The full native
trajectory's update cosine versus the bf16 model is 0.96763, above its existing 0.9
quality threshold. Image-level quality remains a separate measurement.

The full suite and final quick rerun pass. One-step 1024x1024 image-generation
smoke tests also complete for both the Loom backend and the corrected bf16 baseline,
including tiled VAE decoding. These smoke tests do not remeasure eight-step PSNR.

## Standalone prompt-to-image C API

`host/krea2_pipeline.h` adds a separate ABI 1 for complete Turbo inference. Its
implementation is native C++ with HIP/hipBLAS for the Qwen3-VL text encoder,
text-fusion stack, embeddings, final layer and Qwen-Image VAE. The existing 28
W4A4 blocks still use Loom and the block ABI 2. The native byte-level BPE tokenizer
uses PCRE2 and ICU NFC; the scheduler runs in C++. Python and Torch remain export
and reference-test dependencies only. This does not make every auxiliary kernel a
Loom kernel.

The text encoder computes the twelve required hidden-state taps through layer 35,
compacts padding, and uses the same prefix/suffix and 512-token truncation as the
reference. First-frame causal VAE convolutions reduce to their final temporal tap;
temporal upsamplers skip their temporal convolution for this single-image path.
Spatial decoding uses 32-latent tiles with stride 24 only when an axis exceeds 32,
and reproduces the reference's sequential bf16 overlap blending.

The new component tests compare against the installed model implementations:

| Comparison | Measured result |
| --- | ---: |
| Tokenizer, including Unicode normalization, specials and long text | exact IDs |
| Qwen text encoder taps | cosine 0.99985760 |
| Text fusion on identical tapped states | cosine 0.99997914 |
| Time embedding / projection | cosine approximately 1.0 |
| Final layer on identical inputs | cosine 0.99999636 |
| Full transformer, independent quantized trajectory | cosine 0.98455369 |
| VAE RGB MAE, 64×64 | 1.0571 / 255 |
| VAE RGB MAE, 256×256, untiled boundary | 0.9343 / 255 |
| VAE RGB MAE, 320×272, tiled with short edges | 0.9552 / 255 |
| Historical two-step generation versus components with CPU sigmas | exact RGB bytes (not the real pipeline's rounding) |

The full-transformer test uses a 0.95 trajectory threshold, while the identical-input
outer-layer tests require 0.999. These are distinct from the existing per-block
0.99 update-cosine checks. Eight-step image PSNR versus the corrected bf16 baseline
has not been remeasured.

The original scheduler test incorrectly used CPU sigmas and the description above
misidentified it as a GPU scheduler comparison. That exactness claim is withdrawn.
With CUDA sigmas, Torch rounds the delta to bf16 before multiplying the bf16
velocity; the product rounds again before addition to the fp32 sample. CPU scalar
promotion preserves the fp32 delta instead. The native implementation originally
missed the first rounding boundary. Native RNG seeds intentionally differ from
Torch; the API accepts shared initial latents for comparisons.

`tools/export_native.py` stages and publishes a self-contained bundle, refusing to
overwrite existing output. A development-only flag links existing block weights.
The native compiler wrapper uses argument vectors, a process lock, atomic cache
publication and SHA-256 source/configuration fingerprints plus artifact checksums.
Cached inference needs no compiler and can read a bundle without write access.
Compiler changes require explicitly clearing the native cache.

The C example was compiled with `cc`, then generated a 256×256 eight-step image
with `PATH=/nonexistent`. An exec trace contained the C program and the nine native
Loom compiler invocations, with no Python process. The shared library dependency
list contains HIP/hipBLAS, PCRE2, ICU, OpenSSL and standard native libraries, without
Torch or Python.

The standalone CLI also completed an eight-step 1024×1024 image in 45.55 seconds,
including model loading and first-use compilation under process tracing. This is
an end-to-end smoke measurement, not a steady-state performance benchmark. Repeating
the C example against a read-only bundle with an invalid compiler path produced
identical image bytes; its trace contained only the C executable. The generated
previews are `build/native-c-fox.png` and `build/native-fox-1024.png`.

Final validation: `scripts/test.sh --quick --native` passes, including all native
component, scheduler, input-validation, host cleanup and kernel regressions.

## Native allocation and rotary-cache optimization

A per-session temporary-buffer pool removes repeated `hipMalloc`/`hipFree` calls
from the text encoder, outer transformer operations and VAE tiles. It reuses a
buffer only after the last tensor view releases it, and retains at most 512 MiB
of unused GPU storage. Live tensors and model weights are additional. Weights are
loaded outside the pool, and session destruction releases the cached storage.
All operations use the default HIP stream; serialized session calls preserve
ordering even when invoked from different host threads. Error paths restore the
calling thread's previous allocator scope.

Rotary-position arrays are computed once per image geometry and text-token count.
The key includes width and height separately, so equal-area rectangular grids
cannot accidentally reuse each other's phases. Neither change alters the C ABI,
quantization, scheduler arithmetic or image bytes.

`KREA2_NATIVE_PROFILE=1` enables synchronized per-stage wall-clock timing. The first
before/after profiles measured VAE decoding at 10.48/3.81 seconds, but also showed
variation in the unchanged Loom blocks and session loading. These profiles are
useful for finding work, not for estimating the overall speedup.

A separate comparison with profiling disabled used `tools/bench_native.py`, the
same prompt ("a red fox in the snow"), native seed 0, eight steps, 1024×1024,
and three sequential images per library in a resident session:

| Native library | First generation | Warm generation 1 | Warm generation 2 | Warm mean |
| --- | ---: | ---: | ---: | ---: |
| Before buffer/rotary reuse | 32.35 s | 31.90 s | 29.88 s | 30.89 s |
| After buffer/rotary reuse | 32.62 s | 27.17 s | 28.27 s | 27.72 s |

This is a measured 10.3% reduction in warm generation latency (1.11× throughput),
from two warm samples per version; hardware timing varies. Model loading is
excluded and the first generation includes block-session preparation. All six RGB
buffers have SHA-256 `36293aa77377f808462b5b56112424d254cf89d434db47b8b938911c3e724ff7`.
The separate profiled PPM files also compare byte-for-byte equal.

The remaining dominant cost is the Loom block stack. GPU-side conversion of its
residual stream could remove the remaining host round trips; further block GEMM
and attention tuning would target the larger part of runtime.

The allocation/rotary optimization passes the native component suite, including
new equal-area rectangle and cross-thread session regressions. The follow-up
[SA2 implementation and FA3-inspired prefetch](native-attention.md) adds opt-in
native INT4 QK attention, 64-token Q smoothing, sequence-wide K smoothing and
fp16 PV. The ping-pong variant prefetches through alternating LDS slots. Both
variants keep the C ABI and use no Torch runtime; fp16 attention remains the
default. Correctness tests cover partial tiles, zero-range quantization and
byte-identical outputs between the two variants. Timing measurements encountered
other GPU workloads and do not establish a speedup.

## H3-derived tuning and native device bridge

`sa2-fast` transposes V once, uses eight waves and alternating LDS slots below
8,192 tokens, and switches to four-wave prefetch above that boundary. Q mean and
quantization now run separately; K quantization also produces the centered K
copy. Native residual conversions stay on the GPU, and native GEMM raster groups
now minimize padding as the Python builder already did. These changes preserve
the existing SA2 arithmetic and the saved native RGB checksums.

Two warm 1024×1024 eight-step `sa2-fast` generations measured 21.78 and 21.53 s.
Shared GPU work and memory pressure prevent an isolated speedup claim. The full
Torch-reference suite was stopped after the transformer/rectangle/VAE checks
because RAM and swap were exhausted; native-only image comparisons completed.
See [the schedule sweep, timing conditions and validation](native-attention.md).


## Fixed gfx1151 runtime

The tuned INT4 attention path is now unconditional. Runtime attention selectors,
FP16 fallback, the two original SA2 variants and the GEMM group override were
removed. Both builders choose eight waves below 8,192 tokens and four-wave
prefetch above that boundary, always transpose V, and write version-2 launch
metadata. Old launch metadata is rejected. Only the two tuned attention kernels
are exported; the FP16 comparison source lives in `experiments/`. Native HIP
builds explicitly target gfx1151. The C API remains callable without Torch/Python.

The quick suite passes, including generator checks, cache publication/corruption,
constructor cleanup, preparation, RoPE, and both tuned kernels against the
quantization oracle. A real first-block fixture check with the updated smoothed
INT4 storage oracle gives update cosine 0.999946, exact native composition and
0.99897 update cosine against the original bf16 model. The block test now loads
only the checkpoint blocks it exercises.

A default C-API 1024×1024 eight-step generation, red-fox prompt and seed 0,
produced the saved tuned RGB checksum
`31a19cff459feb1f3ac36b496b50ad7b8a40e9bf4810d7b61591ce90f7ce26da`.
Both the first and warm calls matched that checksum. They took 208.95 and
88.24 seconds with concurrent video-model and game workloads; these runs verify
dispatch, session reuse and output preservation, not isolated throughput.
The full Torch-reference pipeline suite was not rerun in this pass.


## Native runtime review fixes

The scheduler now rounds both the delta and its velocity product to bf16, then
adds to the fp32 sample and rounds the result to bf16. The shifted-sigma calculation
also preserves NumPy's float32 boundaries. Using the earlier double-precision
shift crossed a bf16 delta rounding boundary at a few steps, even though the
bf16 transformer timesteps agreed. The update runs as a HIP kernel on resident
latents and velocities, removing the remaining scheduler downloads and upload.

One immutable device weight allocation now belongs to each native pipeline and
is shared with its shape-specific block sessions. Changing token count rebuilds
scratch buffers and kernels without reading or uploading the weight blob again.
The initial upload uses a read-only mmap instead of a full host vector. This does
not introduce a global weight cache or retain scratch buffers for every resolution.
Block modulation tables are packed on-device at model creation; each forward
assembles the float32 modulation buffer with the original bf16 addition rounding
in one GPU kernel, with no per-table host downloads.

The native raster-group rule and GPU residual bridge were already present when
these findings were applied: both builders choose group 3 at 4,115 tokens, and
residual conversion stays on the GPU. The direct `<map>` include was also already
present; `<algorithm>` is now included by the native compiler. Missing JSON files
now report their paths. Tracked completion markers were removed, their old names
are ignored, and workflow markers and logs now live under `build/`.

Validation: the quick suite passes, including actual HIP constructor cleanup and
workflow failure handling. The public block API also passes the real first-block
fixture at 0.999946 update cosine, with exact native composition. The focused native regression suite checks 1,292,800
output elements over all 5,050 steps across inference counts 1 through 100 against
Diffusers with CUDA sigmas, with exact bf16 results and matching bf16 transformer
timesteps. Its fixture also detects the old unrounded-delta bug. All 28 device
modulation tables match independent bf16 Torch additions exactly. A pipeline
runs 64×64 → 64×128 → 64×64 with its private weight and manifest links removed
after the first call; returning to the first shape reproduces the output exactly.
A 64×64 two-step image matches independently scheduled native component calls
using the CUDA Diffusers scheduler exactly. These checks load no full Torch model.
The full Torch-model comparison suite and image-quality sweep were not rerun;
historical image hashes and timings predate the scheduler correction.

### Native HRX / all-Loom auxiliary port (2026-09-05)

The native C ABI now calls HRX directly. The block library owns the shared HRX
runtime, and the full pipeline links it. All GPU operations are Loom, including
BF16 matrix/conv operations, text and VAE attention, Sage preparation/correction,
layout conversions and the BF16 scheduler. HIP, hipBLAS, ICU, PCRE2 and OpenSSL
were removed from the production build. Tokenization uses complete Unicode 16.0
NFC/category tables; hashing uses the included SHA-256 implementation. Build
scripts use C++17 and package HRX plus its compatible HSA provider. See
`docs/hrx-runtime.md` for deployment and the mixed Torch/HRX test environment.

The initial auxiliary launch metadata incorrectly fixed Y to one. Multi-head
Sage and batched GEMM comparisons exposed the resulting constant folding. Both
launch dimensions are now specialized in compiler metadata and tested with tall,
wide, batched and partial-tile matrices. Sage quantization uses four channels per
lane and wave shuffles, eliminating the first port's LDS reduction barriers.
In-memory kernel hits avoid rehashing source on every operation.

Validation:

- The quick suite passes: generated-source consistency, native ownership cleanup,
  Python integration, prepare/rotary kernels, auxiliary arithmetic and both tuned
  attention kernels. Sage cosine against the smoothed INT4 oracle is at least
  0.99999994 across 16, 48, 65, 100 and 4,115 tokens; both variants are byte-identical.
- A C++ process dispatches Loom through HRX and inspects loaded mappings. HRX/HSA
  are present; HIP, BLAS, Torch, Python, ICU, PCRE2 and OpenSSL are absent.
- SHA-256 padding/block boundaries match hashlib. NFC checks cover 17,085
  decomposable codepoints; 310 mixed-script/special-token prompts match tokenizers.
  Invalid, overlong, surrogate and truncated UTF-8 is rejected.
- CUDA scheduler results match exactly over all 5,050 steps at step counts 1–100.
  All 28 modulation tables match BF16 Torch addition. Resolution changes work
  after removing private weight links, and returning to a shape is exact.
- Full model checks: text encoder cosine 0.99984735; text fusion 0.99997938;
  time embedding 1.00000012; modulation 1.00000000; final layer 0.99999636.
  The complete transformer reaches 0.99025321 against the independent outer-model
  implementation using the same Loom blocks. This is not a full BF16 baseline.
- VAE RGB mean errors are 1.2226/255 (64²), 0.9047/255 (256²), and 0.9324/255
  (320×272). Saved HIP results are 1.2350, 0.9084 and 0.9347 respectively; the
  HRX/HIP difference is 0.155–0.170/255. Independent scheduling reproduces native
  two-step RGB exactly. Input validation and calls from multiple threads pass.

The model test now releases the text encoder after collecting its states and
loads the VAE only for decoding. The earlier setup held an unused second 13B
transformer and caused severe swapping. It also produced a different Torch VAE
result: the same input gave 3.736/255 error for HRX and 3.747/255 for the saved HIP
build, versus about 1.2/255 with a freshly loaded VAE. The port did not cause that
difference. VAE weights stayed unchanged in a separate native encode/transformer
stress check, and native decoding was identical before and after it. The revised
full suite passes its original thresholds; they were not relaxed.

Historical throughput measurements above predate this port. The GPU is shared
with unrelated inference work; timings from concurrent correctness tests are not
isolated throughput measurements.

A final startup pass validates and maps each auxiliary weight file and uploads it
into one resident allocation, with shared tensor views. This reduced observed
model loading from 6.31 s to 1.43 s while retaining the exact 1024² RGB checksum.
Final warm generation was 18.49 s versus 19.33 s for the saved HIP build under the
same local conditions. The fresh shared-noise UI pair took 19.36 s for native
HRX generation and 53.73 s for ComfyUI INT8 ConvRot. Detailed conditions, hashes
and the single-prompt quality comparison are in `docs/hrx-runtime.md`.

## GEMM levers from minimax-h3-loom (2026-09-06)

The sibling H3 repo's falsifier-first campaign found four things worth trying here.
Every number below is a paired, interleaved A/B on an idle GPU (`tools/bench_i4_gemm.py`
refuses to time while `gpu_busy_percent` is nonzero; another session's benchmark job
was running on the box, so runs were queued for idle windows), 80 rounds, outputs
compared bit for bit before and after timing. Records: `docs/benchmarks/gemm-*.jsonl`.

**Unconditional staging loads (H3: +5-7%, 228 -> 192 VGPRs).** Replacing the
predicated `scf.if` prefetch loads with plain loads from clamped rows cut the plain
kernel from 156 to 141 VGPRs and 815 to 623 instructions, but Loom then sinks the four
loads into the middle of the WMMA block instead of issuing them after the barrier, and
the timing split by shape: wo (K 6144, N 6144) 1.047x / 1.042x, plain 1.002x / 0.998x,
swiglu 0.996x / 1.027x, down (K 16384) 0.972x / 0.955x. Issuing the loads before the
LDS stores, ahead of the barrier, was worse everywhere (0.88-0.92x). Kept for the
plain and swiglu kernels (simpler, neutral), reverted for the shared resid kernel
because the down projection is the larger stage. `tools/gen_gemm.py` keeps all three
forms selectable (`loads=`) for the 256-row tile.

**Operand row pitch (H3: 31 -> 41 TOPS at 1024-byte-multiple rows).** Same kernel at
K vs K + 128, TOPS by the real K: down projection (8192-byte rows) 60.9 -> 77.5 TOPS,
time ratio 1.26x at 4115 tokens and 1.15x at 8192, reproduced twice with a same-K
control at 0.997x. The 3072-byte rows of K = 6144 showed nothing: plain 0.995x /
0.989x raw time for 2% more bytes, swiglu 0.987x, wo 1.050x (control 0.998x). So
`gemm_pitch(k)` pads only rows that are a multiple of 8192 bytes: the GEMMs and the
prepare kernels take `k_stride` / `out_stride` configs, the weights are re-pitched on
upload (`krea2_weights`, 16 MB host staging chunks, pad bytes zero), the prepared
operand buffer is sized by the padded pitch, and `GEMM_KPAD=128` / the prepare test's
sentinel pad prove neither side touches the padding. Launch metadata is version 3.

**256x128 tile on all four GEMMs (H3: 15-17%).** `tools/gen_gemm.py` is H3's
generator with Krea's epilogues (f16 residual stream with a per-column gate, no
saturation) so the outputs stay bit-exact with the 128-row kernels. With H3's padded
raster grid the tile was a wash at 4115 tokens (plain 1.002x, wo 0.982x, down 1.052x,
swiglu 1.019x: 17 tile rows padded to 18) and clearly ahead at 8192 (1.08x, 1.07x,
1.06x, 1.14x). With the down experiment's in-kernel tail shortening (no ghost
workgroups, `m_group` 4) it wins at 4115 too: plain 1.058x, wo 1.035x, down 1.075x,
swiglu 1.079x; 8192: down 1.071x, swiglu 1.108x; 16896: plain 1.096x; 4353: down
1.038x; 2064: down 1.066x; 1040: plain 0.971x. `gemm_rows(tokens)` therefore takes the
256-row tile from 2048 tokens when its tile rows are within 8% of the 128-row grid's
padded rows. `tests/test_gemm_i4.py` checks both tiles against a float64 oracle and
each other at seven token counts, with and without pad columns.

**Head-major attention operands (H3: +25% on int8 attention).** The Sage preparation
now writes codes `[heads][capacity][64 B]` and scales `[heads][capacity]` so a key tile
is one contiguous block; the attention kernels index the same data head-major. Outputs
are byte-identical (golden outputs at 16..8192 tokens, both wave forms, and the
whole-image hash). Paired against the previous checkout (`tools/bench_sage_attention.py
--against`, preprocessing + kernel, three rounds): 4115 tokens 20.05 -> 19.65 ms
(1.02x), 8192 79.4 -> 73.6 ms (1.08x), 16384 297.5 -> 254.9 ms (1.17x); the kernel
itself 14.7 -> 14.2, 66.7 -> 60.9, 265.6 -> 223.6 ms. Record:
`docs/benchmarks/sage-head-major-2026-09-06.jsonl`.

**Int8-QK attention twin (H3: int4 QK ghosted, int8 matched f16).**
`tools/gen_sage_attention.py` derives `attention_sage_i8_fast{,_prefetch}` from the
finished int4 text (int8 operand schema, 16x36 i32 K tiles, doubled packed columns,
LDS offsets moved 1 KB per K slot; every substitution asserts) and
`tools/gen_native_ops.py` emits `sage_quant_{q,k}_i8` (absmax/127, four codes per i32).
`KREA2_ATTN_QK=8` selects it in both builders; the launch metadata carries the width and
the session picks the kernel and preparation. Both twins match their own oracle to
cosine 1.0 on both wave forms, but at 4115 tokens the int8 kernel takes 29 ms against
the int4 kernel's 14 ms (256 VGPRs with small spills, half the WMMA rate). The bf16
quality sweep (seed 0, 8 steps, 1024^2, `tools/pipeline.py --backend loom` then
`tools/decode_latents.py`) says the attention codes are not where Krea's quality goes:
image PSNR against the bf16 run is 18.91 dB with int4 QK and 19.22 dB with int8 QK
(latent 9.75 vs 10.06 dB); the two Loom runs sit at 24.98 dB from each other, while
both sit ~19 dB from bf16, which is the W4A4 GEMMs and the eight-step trajectory. The
fixture block stack agrees: update cosine vs bf16 0.99899 / 0.96856 (1 / 28 blocks) with
int8 QK against 0.99898 / 0.96576 with int4.
The int4 kernel stays the default; the int8 twin is the switch to flip if a prompt
shows attention ghosting.

**Whole image.** Paired against the previous checkout (its own build and kernel
sources, `tools/bench_native.py --runs 2`, alternating order, two rounds, idle box):
the warm 1024^2 eight-step image went from 18.19 / 18.15 s to 16.63 / 16.62 s (1.09x),
first runs 19.02 / 19.00 -> 17.46 / 17.45 s, every run the same RGB hash
`65507120ec…`. Record: `docs/benchmarks/image-h3-levers-2026-09-06.txt`.
