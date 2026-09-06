# Wide INT4 down projection: not selected

The gfx1151 down projection (K=16384, N=6144) benefits from the MiniMax H3 /
loom-gemm 256×128 workgroup tile in resident-buffer measurements. Each wave
computes 64×64 outputs. Each W operand feeds twice as many accumulators as in
the current kernel; total LDS operand reads per WMMA fall from 0.75 to 0.5.
The 30 KiB operand stage retains register prefetching. Four-tile raster groups
shorten at the end, avoiding empty workgroups.

The candidate remains in `experiments/gemm_down_i4.loom`. It is **not selected
by either production builder**: an integrated trial produced a different image
on its warm repeat. The isolated kernel's speedup is insufficient evidence to
change the default while that observation remains unexplained.

## Kernel measurements

```sh
scripts/build_host.sh
env -u LD_LIBRARY_PATH .venv/bin/python tools/bench_down_gemm.py --baseline db476cf
```

This compiles both kernels, alternates their order on the same resident inputs,
and checks every output bit before and after 80 residual updates. Inputs include
signed gates and nonzero FP16 residuals. Compilation, allocation and transfers
are excluded; HRX submission and synchronization are included.

| Tokens | Baseline median | Wide median | Median paired speedup |
| ---: | ---: | ---: | ---: |
| 4096 | 12.61 ms | 11.90 ms | 1.061× |
| 4115 | 13.75 ms | 12.63 ms | 1.091× |
| 8192 | 25.20 ms | 23.70 ms | 1.058× |
| 16896 | 49.53 ms | 46.48 ms | 1.065× |

[Raw results](benchmarks/down-gemm-2026-09-06.json) include p10/p90 and minima.
The GPU was shared with H3 jobs. One ComfyUI run was killed by the system OOM
killer during the session, and other inference continued afterward. These are
not isolated timings. A 4115-token stage profile attributed 19.1% of block time
to this projection, so the measured improvement predicts only about 1–2% less
block time, not a 6–9% whole-image improvement.

`tests/test_gemm_down.py` checks complete output equivalence and independent
sampled CPU INT4 dots. Cases cover complete raster groups and tails containing
one, two and three tiles, signed gates, zero scales and nonzero residuals.

## Integrated trial and repeat-image failure

The trial automatically selected the wide kernel at 4096 or more tokens. Its
full `scripts/test.sh --quick --native` suite passed, including all 5,050 CUDA
scheduler steps and independent model/VAE references. Those component checks
did not cover two consecutive eight-step generations at 1024².

The same process then generated the seed-zero prompt `a red fox in the snow`
twice at 1024² and eight steps:

| Call | Time under contention | RGB SHA-256 |
| --- | ---: | --- |
| First | 81.36 s | `65507120ecf9fca0ccfb96d3a1ed7d9fc95e087abb4f7a0aed2be75cb21e5f09` |
| Warm repeat | 71.22 s | `83a6f66b2c786cf868faf5b02a45c05978713a89f31bae06b0381fdb663a65a2` |

The first hash matches `db476cf`. A separate trial again produced that first
hash. Repeating individual encoder, transformer and 1024² VAE calls was exact,
including a transformer call after changing its timestep and after decoding.
No root cause or pixel-error magnitude has been established for the differing
warm image. The saved baseline reproduced the expected hash on both its first
and warm calls in a separate process; their timings were not comparable to the
candidate's contended run.

The saved baseline also retained the expected checksum on a second pair under
heavy contention (390.11 s first, 705.76 s warm). The restored production
libraries are byte-for-byte identical to the saved `db476cf` libraries. The
additional 4107-token kernel case matches the native prompt's actual sequence
length; it is included in the kernel regression test.

The production integration was withdrawn. The candidate binaries and integration
patch are retained locally under `build/down-candidate/` for further diagnosis.
`tools/bench_native.py` now fails if an identical prompt/seed repeat changes the
RGB checksum, rather than merely printing the different hashes. No whole-image
speedup or state-of-the-art ranking is claimed from this experiment.

## Other rejected candidates

- Compact rotated LDS and alternating INT4 buffers preserved kernel outputs but
  generally lost 6–13%. Padded alternating buffers were neutral or slower.
- Explicitly unrolling the existing GEMM output loops was effectively neutral.
- Transposing attention scores to KQᵀ enabled reductions within each lane and
  direct probability operands for PV. Benchmark outputs remained bit identical
  at 65, 4115 and 8192 tokens, but scale/correction and output-rescale gathers
  emitted costly LDS lane permutations. Register use rose from 240 to 256;
  the first 4115-token comparison regressed from 15.05 to 17.44 ms median.
  Static-gather and direct-load variants also failed to win.
