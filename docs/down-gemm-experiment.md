# Wide down-projection experiment

The original hand-written INT4 down-projection candidate is retired. Its
256×128 tile and raster-tail shortening were incorporated into the GEMM family
now checked in at `kernels/gemm_*.loom`, which is authoritative since
its generator was retired; production selection follows `src/kernels/shape.rs`.

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

`tests/softmax_repeat.rs` uses constant logits with exactly known
probabilities to exercise repeated resident dispatch. The
[recorded investigation](benchmarks/softmax-race-2026-09-06.json) includes
failing and corrected runs.

The repeated-generation check that caught this -- repeated RGB hashes, with
differing images and error statistics retained on failure -- lived in
`tools/bench_native.py`, retired in e33b171 and recoverable from Git. Kernel
microbenchmarks and isolated component calls do not replace it, and nothing in
the Rust suite covers it today.
