# Attention on gfx1151

Attention uses fp16 QK and PV products with an fp32 online softmax.
`attention_gqa_lds_f16_wmma.loom` handles sixteen query rows per workgroup.
At 8192 tokens and above, `attention_gqa_lds_f16_wmma_q32.loom` shares K/V
staging between two groups of sixteen queries. Both preserve the same
per-query arithmetic. Source selection, launch dimensions and buffer capacity
are defined together in `src/kernels/shape.rs`.

V is transposed to `[kv_heads × 128][capacity]` by `sage_transpose` after
RoPE so lanes can load contiguous fragments. Each iteration prefetches the
next K/V tile. V fragments are consumed in pairs to limit register pressure.
Padding and partial tiles must remain masked consistently with the logical
sequence length.

The older `experiments/attention_query32.loom` uses an accumulator-to-RHS
repack and is distinct from the production query-group kernel. The pinned
compiler rejects that experimental layout with `AMDGPU/041`.
Retired smoothed int8/int4 QK kernels remain in `experiments/sage/`.

`tests/quantized.rs` checks both production kernels against independent CPU
softmax, compares their outputs, exercises ragged lengths and checks for spills
across the selection boundary. `scripts/parity.sh` checks complete latent and
image trajectories against the frozen unquantized BF16 reference.

Use the [Criterion benches](../CONTRIBUTING.md#benchmarks) for timing.
The isolated attention bench starts with transposed V; full generation includes
transposition and all other preparation.
