# Attention on gfx1151

## Supported modes

| Mode | Selection | Arithmetic |
| --- | --- | --- |
| fp16 (default) | `--attn f16` or `KREA2_ATTN_QK=16` | fp16 QK/PV, fp32 online softmax |
| Smoothed int8 QK | `--attn i8` or `KREA2_ATTN_QK=8` | Per-token int8 QK, fp16 PV, fp32 softmax and correction |
| Smoothed int4 QK | `--attn i4` or `KREA2_ATTN_QK=4` | Per-token int4 QK, fp16 PV, fp32 softmax and correction |

Selection is recorded in the compiled bundle metadata. The default kernel is
`kernels/attention_gqa_lds_f16_wmma.loom`. Experimental query32 attention is
not selected by either production builder; see [its evaluation](attention-2x.md).

## Quality tradeoff

On the fixed 1024×1024, seed-zero, eight-step W8A8 fixture, against the bf16
transformer using identical noise, conditioning, scheduler and VAE:

| Attention | Latent PSNR | Image PSNR |
| --- | ---: | ---: |
| fp16 | 24.57 dB | 33.67 dB |
| Smoothed int8 QK | 22.6 dB | 31.8 dB |
| Smoothed int4 QK | 21.62 dB | 31.57 dB |

These are fixture results, not a quality sweep. The fp16 default preserves the
best measured trajectory agreement. `scripts/parity.sh` checks
an accepted reference trajectory; see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Smoothed attention

The int4/int8 paths adapt Q/K smoothing from
[SageAttention2](https://arxiv.org/abs/2411.10958) to gfx11 WMMA. They use
per-token scales and fp16 PV, so they are not an exact implementation of the
upstream arithmetic.

`crates/session/src/sage.rs` prepares each block:

1. Compute K's mean over the sequence, per head/channel.
2. Center Q in 64-token groups, excluding padding from the means.
3. Quantize centered Q and K with symmetric absmax scales: codes −7…7 for
   int4, or −127…127 for int8.
4. Compute the query-mean × centered-key correction with fp16 inputs and fp32
   output. Four query heads share each KV head.

The attention kernel dequantizes QK, adds the correction, scales by
`1/sqrt(128)`, and performs online softmax and PV. The omitted K-mean term is
constant across each score row and cancels in softmax. Correction input rounding
and quantization remain approximations.

Codes and scales are head-major. V is transposed before attention. Below 8192
tokens, eight waves process two query tiles across four Q heads with alternating
LDS slots. At longer sequences, four waves use explicit prefetch. The threshold
is an empirical choice on gfx1151.

## Validation

- `crates/loom/tests/quantized.rs`: production fp16 attention and quantized
  preparation against independent softmax and Hadamard references, including
  padded and partial tiles.
- `crates/pipeline/tests/unquantized_parity.rs`, via `scripts/parity.sh`:
  complete latent and image trajectories against the unquantized BF16 model.

The Python attention benchmarks were retired with the rest of that layer; the
measurements below stand and the tooling is recoverable from Git before e33b171.
Benchmark results must include preparation/transposition costs and must not be
presented as whole-image speedups. Historical records are in `docs/benchmarks/`.
