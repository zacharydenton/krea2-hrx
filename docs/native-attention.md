# Attention on gfx1151

## The kernel

Attention is fp16 QK and PV with an fp32 online softmax. The kernel is `kernels/attention_gqa_lds_f16_wmma.loom`. Experimental
query32 attention lives in `experiments/` and is not embedded; see
[its evaluation](attention-2x.md). The pinned HRX 0.8 compiler rejects its
accumulator-to-RHS repack with `AMDGPU/041` / `layout_strategy`. Production
CPU-oracle tests cover query16; a separate diagnostic test records the query32
limitation.

The default kernel consumes V fragments in pairs to avoid VGPR spills. See the
[register-pressure investigation](attention-spills.md) for disassembly findings,
the Rust comparison tool, and measured kernel latency.

Every mode reads V transposed, `[kv_heads × 128][capacity]`, written by
`sage_transpose` after RoPE. The fp16 kernel stages each 16-key V tile with one
32-byte load per lane from a V^T row. Reading natural `[tokens][kv_heads × 128]` V
made each lane fetch 32 bytes from a different token row. That kernel held 43%
of the register-only fp16 WMMA peak at 1–2k tokens, but fell to 37% at 4115
and 19–22% past 6k, once a sequence's K and V left the cache. Byte-identical
output, same-run device-clock medians on a shared GPU (`examples/attention.rs`):

| Tokens | Natural V | V^T | Speedup incl. transpose |
| ---: | ---: | ---: | ---: |
| 1040 | 1.113 ms | 1.121 ms | 0.97× |
| 1555 | 2.575 ms | 2.568 ms | 0.99× |
| 4115 | 22.57 ms | 17.08 ms | 1.31× |
| 6163 | 77.68 ms | 40.07 ms | 1.92× |
| 9000 | 199.8 ms | 100.3 ms | 1.98× |

The transpose takes 0.02 ms at 1040 tokens and 0.13 ms at 4115. XOR-swizzling
the K tile instead of padding it raised occupancy but lost 4–7% below 2k
tokens. Packing the V and Q tiles to reach the next LDS occupancy tier lost
6–21%: the swizzles either conflicted or needed split loads and concatenation.

Each iteration now prefetches the next tile's K and V into registers, so the
global loads overlap the current tile's work. This is byte-identical and
1.05–1.07× faster up to 4115 tokens, and neutral at 9000.

Removing each phase from that kernel in turn, at 4115 tokens, shows what
limits it:

| Removed | Speedup |
| --- | ---: |
| QK WMMAs | 1.40× |
| PV WMMAs | 1.31× |
| softmax | 1.12× |
| K/V staging (before prefetch) | 1.15× |
| K, Q or V fragment loads | ≤ 1.04× |
| barriers, fences, rescale | none |

On RDNA3 the WMMAs execute on the same VALU as everything else. Each 16-key
tile issues 16 WMMAs beside about 256 other VALU instructions. Of those, 72
are back-edge copies: the compiler writes each `accumulator × scale` rescale
into a fresh aligned register tuple and moves all 64 values back before the
branch (`move_causes`: `branch_edge`, 171 units). The copies disappear when the
rescale is removed. Spelling the multiply plainly or starting the accumulators
from distinct values does not help. Carrying one `vector<64xf32>` is rejected
because WMMA results cannot be concatenated (`concat.register_storage`), so
the fix belongs in Loom's allocator. hrx-rs compiler patch 0011 makes it:
the back edge keeps 21 moves, and the kernel runs 1.007-1.073x faster from
1040 to 9000 tokens (1.049x at 4115), byte-identical. The native bundle of
hrx-rs 0.8.10 includes it. `unroll(2)` gains 4.7% at an even tile
count but loses 11–16% at an odd one. Splitting the QK chain gains 1.5% at
most.

## Quality tradeoff

On the fixed 1024×1024, seed-zero, eight-step W8A8 fixture, against the bf16
transformer using identical noise, conditioning, scheduler and VAE:

| Attention | Latent PSNR | Image PSNR |
| --- | ---: | ---: |
| fp16 | 24.57 dB | 33.67 dB |
| Smoothed int8 QK (retired) | 22.6 dB | 31.8 dB |
| Smoothed int4 QK (retired) | 21.62 dB | 31.57 dB |

These are fixture results, not a quality sweep. `scripts/parity.sh` checks an
accepted reference trajectory; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Retired: smoothed int8/int4 QK

Until 2026-09-28, `--attn i8` and `--attn i4` (or `KREA2_ATTN_QK`) selected
SageAttention2-style smoothed QK: per-token int8 or int4 Q and K centered on
their means, a mean-correction GEMM, fp16 PV. On gfx1151 they stopped paying
once fp16 attention reached 48% of its peak. At 1024x1024, one step on the GPU
clock:

| Attention | Attention kernels | Preparation | Step |
| --- | ---: | ---: | ---: |
| fp16 (with V transpose) | 438 ms | — | 3.00 s |
| Smoothed int8 QK | 429 ms | ~40 ms | ≈ fp16 |
| Smoothed int4 QK | 324 ms | 40 ms | 2.92 s |

Int8 WMMA runs at the fp16 rate on this part (57 against 54 TOPS), so int8 QK
bought nothing. Int4 WMMA is twice as fast, but smoothing quantizes only QK,
half the attention's products: the ideal is 1.33x, and the int4 kernel reached
1.32x. Its preparation (key and query means, quantization, the correction GEMM)
returned 40% of that, leaving a step 3% faster for 2–3 dB less latent PSNR.
Attention is 14% of a step, the GEMMs about 80%. The kernels and their
preparation are kept in `experiments/sage/`; the host side is recoverable from
Git before its removal.

## Validation

- `tests/quantized.rs`: production fp16 attention and the int8 activation
  preparation against independent softmax and Hadamard references, including
  padded and partial tiles.
- `tests/unquantized_parity.rs`, via `scripts/parity.sh`:
  complete latent and image trajectories against the unquantized BF16 model.

The Python attention benchmarks were retired with the rest of that layer; the
measurements below stand and the tooling is recoverable from Git before e33b171.
Benchmark results must include preparation/transposition costs and must not be
presented as whole-image speedups. Historical records are in `docs/benchmarks/`.
