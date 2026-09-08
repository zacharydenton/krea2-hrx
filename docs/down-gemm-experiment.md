# Wide INT4 down projection: not selected

The gfx1151 down projection (K=16384, N=6144) benefits from the MiniMax H3 /
loom-gemm 256×128 workgroup tile in resident-buffer measurements. Each wave
computes 64×64 outputs. Each W operand feeds twice as many accumulators as in
the current kernel; total LDS operand reads per WMMA fall from 0.75 to 0.5.
The 30 KiB operand stage retains register prefetching. Four-tile raster groups
shorten at the end, avoiding empty workgroups.

The candidate was **never selected by either production builder**: an integrated
trial's repeat-image failure prompted withdrawal, and the follow-up investigation
below reproduced that failure class in the production VAE without the wide kernel
and fixed a shared-memory race in auxiliary softmax. Its idea arrived by another
route when `tools/gen_gemm.py` grew the 256x128 tile for every projection, so the
hand-written kernel, its test and its benchmark were removed; the measurements
below are kept for the record and can be read back out of the history.

## Kernel measurements

```sh
scripts/build.sh
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

A later paired run on 2026-09-06 (30 alternating rounds, `--baseline HEAD`),
while another process kept the GPU at 100% busy, reversed the result: 0.952×,
0.878×, 0.970× and 0.956× at 4096, 4115, 8192 and 16896 tokens, with the
baseline itself running about twice as slow as in the table above. The two
contended measurements disagree, so the kernel stays unselected until an idle
box can decide it. `tests/gemm_bench.cpp` now accepts a same-tile baseline
symbol so operand-path changes can be isolated from tile selection.

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
The original differing image's pixels were not retained. The saved baseline
reproduced the expected hash on both its first
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

## Follow-up: auxiliary softmax race

Replaying the saved candidate produced four exact full images. A diagnostic
build also captured identical initial latents, encoder taps, fused text, all
eight velocities and all eight updated latents across three generations.
Another diagnostic compared both down kernels on the same real inputs in every
block: all 448 projections across two full generations matched bit for bit.
These diagnostics add synchronization, so they alone could not exclude a race.

Concurrent candidate/baseline generation reproduced an image mismatch. The
candidate's first RGB hash was
`189999e53a084add31ca3c6ce8d7de75905a2833607607dd75f9e3b9056391b6`;
its next call and the baseline's first call produced the expected `65507120…`
hash. The saved image pair differed in 676 pixels / 723 channels, each by just
one 8-bit level. Every difference lay inside x=195..336, y=961..1023, suggesting
a VAE edge tile. Extra diagnostic processes were stopped when another workload
grew and available memory fell below 2 GiB; unrelated jobs were left alone.

Replaying the known final latent through the **unchanged production decoder**
then failed on call nine. No transformer or wide INT4 kernel ran in that test.
Capturing each tile on another replay showed identical latent input and one
changed decoded tile. Tracing that tile's operations isolated attention;
tracing attention's intermediates showed identical Q, K, V and QK scores but
different softmax probabilities. One softmax row had all 256 probabilities
changed. Small final RGB differences do not describe the size of this internal
error.

Both auxiliary softmax variants reused one 1 KiB LDS buffer for the maximum and
sum reductions. The old 256-token binary contains this sequence at 0x1290:

```asm
ds_load_b32 v1, v1   ; broadcast maximum
s_barrier           ; no lgkmcnt wait before the barrier
```

The wait occurs later, before consuming the loaded value. A different wave
can cross the barrier and overwrite the same LDS location with its partial sum
before an outstanding maximum read completes. Separate 1 KiB regions for the
two reductions remove that overlap. No region is overwritten after its final
broadcast; the two post-broadcast barriers are therefore unnecessary and have
been removed. Reduction order, exponentiation and BF16 rounding are unchanged.
The changed embedded source automatically selects a new kernel-cache entry.

`tests/test_softmax_repeat.cpp` uses constant logits, whose uniform probabilities
are independently known exactly. It failed against `db476cf` on the second
256-token dispatch. The fixed kernels passed 704 repeated resident dispatches
across 33, 256, 257 and 1024 tokens, including causal masking. The regression is
part of `scripts/test.sh` and requires neither models nor Torch.

After the fix, `scripts/test.sh --quick --native` passed in full. All 20 replays
of the previously failing final latent matched the expected RGB bytes. Rebuilding
the saved wide-kernel candidate with the corrected auxiliary kernels produced
four matching full images, including three warm repeats; decoder replays and
generation overlapped during part of this validation. These runs validate the
fix, not an isolated performance comparison. [Recorded results](benchmarks/softmax-race-2026-09-06.json)
include the failing and corrected runs.

`tools/bench_native.py` now retains the first and differing lossless PPM images,
checksums, input settings and pixel-error statistics in a unique
`build/native-mismatch-*` directory before failing. Its host regression checks
that a later call cannot overwrite the retained first image and that failure
still destroys the pipeline. The original `83a6f66b…` image was never saved, so
its exact pixels cannot be retrospectively compared with the reproduced faults.

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

## Outcome (2026-09-06, idle-box A/B)

`tools/gen_gemm.py` now generates the 256x128 tile for all three epilogues
(`kernels/gemm_i4{,_resid,_swiglu}_256.loom`) with this kernel's raster-tail
shortening and a `k_stride` operand pitch; `tests/test_gemm_i4.py` holds them
bit-exact against the 128x128 kernels and a float64 oracle. On an idle GPU the
shortened tile beat the padded raster form at 4115 tokens (where the padded form was
a wash: 17 tile rows padded to 18) and won on every GEMM from about 2k tokens; both
builders pick the tile by a waste-aware rule (`gemm_rows`). The measurements are in
`docs/benchmarks/gemm-tile-*.jsonl` and `docs/notes.md`. This kernel stays as the
hand-written original; `tools/bench_down_gemm.py` still times it.
