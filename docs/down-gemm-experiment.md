# Wide down-projection experiment

The original hand-written INT4 down-projection candidate is retired. Its
256×128 tile and raster-tail shortening were incorporated into the generated
GEMM family in `tools/gen_gemm.py`; production selection now follows
`crates/loom/src/shape.rs`.

Early resident-kernel trials reported 1.06–1.09× improvements under contention.
A later contended run reversed the result, so those samples are not evidence
of a reliable whole-image speedup. The
[raw measurements](benchmarks/down-gemm-2026-09-06.json) remain available;
commands for the removed candidate are in Git history.

## Repeated-image failure

An integrated trial produced different RGB on repeated identical requests.
The failure was subsequently reproduced in the unchanged VAE decoder, without
the candidate down kernel. Attention scores were identical, but softmax
probabilities differed.

The two softmax reductions reused one LDS region. A wave could overwrite a
partial maximum with a partial sum while another wave still had an outstanding
read. Maximum and sum reductions now use separate regions that are not
rewritten after their final broadcasts. Reduction arithmetic is unchanged.

`crates/ops/tests/softmax_repeat.rs` uses constant logits with exactly known
probabilities to exercise repeated resident dispatch. The
[recorded investigation](benchmarks/softmax-race-2026-09-06.json) includes
failing and corrected runs.

`tools/bench_native.py` also checks repeated RGB and retains differing images,
settings and error statistics on failure. Kernel microbenchmarks and isolated
component calls do not replace this repeated-generation check.
