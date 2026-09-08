# Query32 attention experiment

**Status: experimental, not selected by production builders.**

`kernels/attention_query32.loom` processes more query rows per workgroup to reuse
K/V loads. At 4115 tokens, paired kernel measurements reported 1.76–1.78×
speedup including V transposition. The change nevertheless failed the full
image-quality gate.

On the same 1024×1024, seed-zero, eight-step fixture:

| Attention | Latent PSNR | Image PSNR |
| --- | ---: | ---: |
| Default fp16 | 24.57 dB | 33.67 dB |
| Query32 | 17.81 dB | 26.28 dB |

Block update cosine was approximately 0.9991 for both kernels, which did not
predict the 7.39 dB image loss. Synthetic oracle and same-input block checks
were therefore insufficient to promote this kernel. The numerical cause
remains unresolved.

Both builders select one fp16 query tile at every supported sequence length.
Bundle metadata records the query-tile count; sessions reject incompatible
bundles. The experimental generator and kernel remain available for investigation.

## Requirements for reconsideration

A candidate must pass synthetic and real-input attention checks, repeated
resident dispatch, builder/metadata parity, and the complete trajectory gate:

```sh
source scripts/env.sh
scripts/test.sh --quick --quality
```

Set `KREA2_QUALITY_BASELINE` to an accepted run. For retained candidate output:

```sh
.venv/bin/python tools/quality_vs_bf16.py regression \
  --baseline /path/to/accepted-run --work build/quality-query32
```

The gate requires matching reference inputs and permits at most 0.1 dB loss
in either latent or image PSNR. Additional prompts and seeds are needed to
establish broader quality. Preserve numerical failures in reports even when
aggregate metrics or kernel timings improve.

See [supported attention modes](native-attention.md). Detailed trial history
is available in Git history; benchmark artifacts remain in `docs/benchmarks/`.
