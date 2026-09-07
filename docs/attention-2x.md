# FP16 attention 2x experiment

Target: at least 2x the shipped fp16 attention throughput at 4,115 tokens,
48 query heads, 12 KV heads and dimension 128 on gfx1151. The fixed baseline
is `6405667:kernels/attention_gqa_lds_f16_wmma.loom`. Preserve fp16 QK and PV
with fp32 softmax and accumulation. Measure both kernels in one process on the
same resident inputs, alternating their order within each timing pair.

**Status: query32 has been withdrawn as the default after an end-to-end quality
regression. Both production builders select the original fp16 kernel at every
sequence length, and the host rejects query32 bundles. The 1.76–1.78x attention
speedup is real but does not qualify the kernel for production. The original 2x
target was not reached. query32 remains available through the benchmark harness.**

## Quality regression and corrected release gate

The controlled comparison used the same binary, noise, text states and bf16
reference, changing only `fp16_query_tiles` to select the original kernel. At
4,115 tokens, seed 0 and eight steps:

| Attention | Latent PSNR | Image PSNR |
| --- | ---: | ---: |
| Original fp16 | 24.57 dB | 33.67 dB |
| query32 | 17.81 dB | 26.28 dB |
| int4-QK Sage | 21.62 dB | 31.57 dB |

The 7.39 dB image loss contradicts the reason fp16 was selected over Sage. Block
cosine checks did not detect it: the old and new kernels score about 0.99910 and
0.99911 against the bf16 reference. The earlier pointwise exceptions and 2.057%
state drift warranted end-to-end investigation before promotion. Passing those
block checks was insufficient evidence to ship.

`tools/quality_vs_bf16.py check --baseline BASELINE --work CANDIDATE` now gates
both complete-trajectory latent and decoded-image PSNR, allowing at most 0.1 dB
loss from an archived accepted run. It requires identical job settings, noise,
text states, bf16 reference latents and decoded reference image, and records
artifact hashes and both metric losses in `quality-check.json`. This CPU-only
check rejects the saved query32 pair with losses of 6.757 and 7.386 dB.

`scripts/test.sh --quick --quality` builds the native pipeline, runs all eight
steps on frozen reference inputs in a fresh directory, decodes both trajectories
with the same VAE, then applies this gate. Set `KREA2_QUALITY_BASELINE` to an
archived accepted run; the default is `build/quality`. For a retained run:

```sh
scripts/build_native.sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/quality_vs_bf16.py regression \
  --baseline build/quality_qt1_check --work build/quality-restored
```

This adds a release requirement for attention changes. It catches this failure
on one fixed prompt/seed; additional prompts and seeds are needed before claiming
general image-quality equivalence. It does not diagnose the numerical defect in
query32, which remains unresolved.

Rollback validation: rebuilding the native pipeline and rerunning all eight steps
plus both VAE decodes reproduces 24.566572 dB latent and 33.667317 dB image PSNR,
exactly matching the accepted run (zero loss in both metrics). Python/native
shader bytes and metadata agree at 16, 2,047, 2,048 and 4,115 fp16 tokens, and
at 8,192 tokens with int4 attention. Runtime tests, constructor rejection of
query32 bundles, and five CPU quality-gate tests pass. Reports are retained in
`build/quality-restored/quality-check.json` and
`build/quality_query32/quality-check.json`.

## Historical speed and block validation

Two 80-pair runs establish the main-shape result, including V transposition.
Additional 10-pair runs measure 1.39x at 2,048 tokens, 1.98x at 8,192, 2.34x at
8,193, and 3.14x at 16,896; every run passes the unchanged synthetic oracle and
full-output baseline checks. These measurements apply to attention, not the
entire model.

Bundle metadata version 5 records the fp16 query tile count. The initial promotion
included a transpose kernel for long sequences. After rollback, both builders
record one query tile and omit the transpose. Version 4 remains compatible;
version 5 bundles requesting two query tiles are rejected. The experimental
generated kernel remains `kernels/attention_query32.loom`.

Real-input comparisons below retain five pointwise exceptions among 25.3 million
outputs in block 27 and 2.057% final-state RMS drift over the independent 28-block
trajectory. The synthetic tolerance was not relaxed; the subsequently measured
image regression above disqualifies this candidate despite those synthetic passes.

## Initial release validation (insufficient)

All 28 native blocks pass the existing same-input reference checks; the lowest
update cosine is 0.999921. Composing the individual blocks exactly reproduces
one full forward at both 1 and 28 blocks. The 28-block update cosine against
the bf16 reference is 0.99911. These checks retain the project's existing
thresholds and do not override the pointwise exceptions documented above.

Python and native builders produce identical launch metadata and shader bytes
at 16, 2,047, 2,048 and 4,115 tokens with fp16 attention, and at 8,192 tokens
with int4 attention. The production attention binary at 4,115 tokens is identical
to the binary used for the 80-pair measurements (SHA-256
`ddff43336c5bb6701a5d048ce6b8f780f99dad87ff939a7a7e3bc9337fd9a1da`).

Host and native builds, runtime and constructor checks, the CPU attention model,
and all kernel regression checks pass. The quick suite exposed stale generated
Sage sources from an older generator change; regenerating them restored source
parity, and the affected GPU checks pass. Torch and HRX block sessions run in
separate processes in the model test to avoid conflicting HSA providers.
With `scripts/env.sh` loaded, the native pipeline regression also passes all
5,050 CUDA scheduler steps, all 28 modulation tables, exact weight reuse across
64x64 / 64x128 / 64x64 resolutions, and exact RGB agreement for two-step generation
against components driven by the CUDA diffusers scheduler.

## Transposed products

`tools/gen_attention_query.py` generates
`experiments/attention_query_f16.loom`. It inherits the shipped GQA staging,
then computes `Sᵀ = K Qᵀ` and `Oᵀ = Vᵀ Pᵀ`. A gfx11 wave32 accumulator lane
owns one column and eight even or odd rows. With queries in the columns, each
lane owns eight keys of one query rather than eight queries of one key.

Consequences:

- Softmax uses an eight-element local reduction and one scalar exchange with
  lane xor 16. Local reductions use a three-level tree.
- Running max and partial sum are scalar per lane; rescaling needs one
  exponential instead of eight. There are nine exponentials per key tile
  instead of sixteen.
- The compiler's result-to-fp16-RHS repack joins even and odd keys through a
  half-wave exchange. No probability transpose through LDS is needed.
- Output fragments use the same repack to store contiguous channel vectors.
  Only the low half-wave publishes; repacking executes on the whole wave.
- Key masking is limited to the final key tile. Query publication remains
  masked for partial tiles.

The lane mapping comes from the local compiler's
`lower/matrix_fragment_repack_packed_b16.c`, which implements the gfx11
result-to-RHS transition. The CPU tests independently compare the resulting
attention arithmetic with dense float64 attention, including partial tiles,
sharp softmax distributions and single nonzero values at tile boundaries.
The CPU model also exercises all 4,115 keys with a forced late maximum change
and the candidate's explicit three-level fp32 sum, for both 16-key and 32-key
updates.
These tests cannot establish correctness of the emitted machine code.

At 4,115 tokens, the default candidate compiles to 224 VGPRs and 19,712 bytes
of LDS, with no spills. The baseline uses 240 VGPRs and 21,760 bytes of LDS.
Additional compiled candidates trade Q storage against registers:

| Q fragments hoisted | Remaining Q storage | VGPRs | LDS bytes | Private bytes/thread |
| ---: | --- | ---: | ---: | ---: |
| 0 | LDS | 192 | 27,904 | 0 |
| 2 | LDS | 208 | 23,808 | 0 |
| 4 (default) | LDS | 224 | 19,712 | 0 |
| 6 | LDS | 248 | 15,616 | 0 |
| 8 | Registers | 256 | 10,496 | 40 |
| 0 | Global reload | 200 | 10,496 | 0 |
| 2 | Global reload | 216 | 10,496 | 0 |
| 4 | Global reload | 232 | 10,496 | 0 |

These are resource measurements, not speed rankings. The generator accepts
the original `ATTN_HOIST`, `ATTN_QLDS` and `ATTN_QTILES` knobs. Key tiles are
currently fixed at 16. `ATTN_PV_GROUP` (default 4) inserts subgroup memory
ordering after groups of PV products to limit load hoisting; zero disables it.
`ATTN_DIRECT_OUT=0` and `ATTN_TAIL_ONLY=0` retain alternative output and masking
schedules for falsifying their performance benefit.

## Reproduction

CPU only:

```sh
OPENBLAS_NUM_THREADS=2 python3 tests/test_attention_query_cpu.py
python3 tools/gen_attention_query.py
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py --prepare-only --tokens 16,17,65,4115
```

The preparation command compiles both sources and saves padded inputs, sampled
CPU float64 oracle outputs, source hashes and an exact runner command under
`build/attention-2x/paired/`. It never initializes or launches the GPU. Format
generated Loom sources with `loom-format --in-place` before committing them.
New manifests also hash the compiled kernels and the host runner, so different
compiler experiments cannot be identified solely by identical source text.

Once GPU use is authorized and the device is idle:

```sh
scripts/build_host.sh
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py --tokens 16,17,65,100 --rounds 10
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py --tokens 4115 --rounds 80
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py --tokens 65,4115 --scale 3 --rounds 20
```

Use `--qtiles` to match a nondefault `ATTN_QTILES` candidate. The runner checks
both kernels against the independent CPU oracle, compares every candidate
output with the baseline, and repeats checks after timing. Output buffers start
as NaNs so unwritten rows fail. Accuracy requires relative RMS error at most
0.002 and every element within `0.002 + 0.002 * abs(reference)`. Reports include
paired speedup percentiles and both latency distributions, rather than a ratio
of unrelated best times. Compilation and CPU reference work finish before the
idle gate and timing.
Each result embeds the source/configuration manifest. Preparing new inputs or
sources at an existing run path removes its old timing report, so failed or
CPU-only preparation cannot inherit a previous speedup claim.

Do not promote a candidate from compiled resource counts alone. Establish GPU
correctness first, then repeat paired timings at 4,115 tokens and check other
sequence lengths. Include the complete cost of preprocessing in the speedup.
Matrix orientation and softmax summation change fp32 rounding even though the
operand precision remains fp16. The real-input and trajectory measurements below
document differences that the initial promotion accepted prematurely. The later
image-quality regression above overrides that decision; query32 is not qualified.

## Initial GPU results

Ten alternating pairs at 4,115 tokens, Q/K standard deviation 1, CPU oracle
checks before and after timing. These short sweeps select candidates; longer
repeat runs are required to establish the final speedup.

| Candidate | Paired median speedup | Candidate median ms |
| --- | ---: | ---: |
| Transposed products, one query tile | 1.10x | 22.40 |
| Two query tiles | 1.36x | 18.18 |
| Two query tiles, global V transpose included | 1.55x | 15.80 |
| 32 keys, two query tiles, V transpose included | 1.61x | 15.27 |

Global Q reloads, packing before probability exchange, and four query tiles
were slower. Two QK accumulation chains and split K/V staging did not produce
a clear benefit. `tools/gen_attention_query32.py` generates the 32-key
variant; it requires `--transpose-v` in `tools/bench_attention.py`, which
includes `sage_transpose` on every timed candidate call. `ATTN_DOUBLE_BUFFER=1`
uses alternating LDS slots and one barrier per tile.

## Current 32-key candidate

The defaults of `tools/gen_attention_query32.py` now reproduce the leading
candidate: two query tiles, four Q fragments hoisted, wide staging and native
fp16 probability repacking, with PV memory ordering after every two channel
groups. `ATTN_WIDE_STAGE=0` and `ATTN_NATIVE_PACK=0` retain the earlier paths.

Wide staging assigns each of 128 threads a contiguous 32-half K chunk and a
contiguous 32-half V chunk. It removes the distant second K-row load and its
base-address updates. The probability exchange converts fp32 weights once,
places them in the low halves of a native fp16 result fragment, and repacks
that fragment to the PV RHS. Softmax sums and output accumulation stay fp32.
Scheduling matters: without the PV ordering points, the native repack variant
uses 256 VGPRs and is slower; groups of two use 232 VGPRs with no spills and
37,376 bytes of LDS.

```sh
python3 tools/gen_attention_query32.py
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py attention_query32 \
  --tokens 4115 --qtiles 2 --rounds 80 --transpose-v
OPENBLAS_NUM_THREADS=2 python3 tools/bench_attention.py attention_query32 \
  --tokens 16,31,32,33,65,100,4115 --qtiles 2 --rounds 20 --scale 3 --transpose-v
```

The 80-pair report is
`build/attention-2x/paired/attention_query32-4115-s1.0-seed917/result.json`.
Its paired speedup p10/median/p90 is **1.751 / 1.783 / 1.815**. The candidate's
relative RMS error against the sampled float64 oracle is 0.000284; its
relative RMS difference from the baseline across every output is 0.000100.
The sharp-input checks above all pass, as does the 17-token edge case.
Short sequences can be slower because preprocessing and launch costs dominate.

A second 80-pair run with seed 2041 gives p10/median/p90 speedups of
1.699 / 1.756 / 1.796, with candidate/baseline median latencies of
14.02 / 24.59 ms. At 8,193 tokens, ten pairs give 2.340x and median latencies
of 56.02 / 131.49 ms; the candidate's relative RMS error against the sampled
CPU oracle is 0.000290. This longer-sequence result does not establish 2x at
the primary 4,115-token shape.

Unrolling and processing two query tiles within a single wave currently hit
compiler allocation failures. Wider 64-key tiles, prefetching, distributing
staging across 256 threads, compact Q rows, and a contiguous packed V layout
were slower. These unsuccessful candidates and their diagnostic scripts are
retained under the ignored `build/attention-2x/` directory. No shared compiler
installation was modified by these experiments.

GPU edge testing exposed a benchmark ABI error: `token_count` is an 8-byte
Loom index, but the runner put KV heads in its high 32 bits. The runner now
passes an explicit int64 token count, and the older Python attention launchers
zero that high word. The direct-output candidate additionally guards against
the compiled sequence length, as the shipped kernel does. Partial-tile checks
at 17 and 65 tokens pass after the fix.

## Hardware profiling and compiler experiments

The installed rocprofv3 runtime lacks an HSA extension required by HRX. A
profiling-only HIP adapter runs the same compiled kernels and independent CPU
oracle checks through stock ROCm. Its timings are not used for the speedup
qualification above. The deployment remains HRX/Loom.

Two sampled dispatches at 4,115 tokens give these counters:

| Counter | Baseline | Current 32-key candidate |
| --- | ---: | ---: |
| LDSBankConflict | 13.76% | 2.56% |
| L2CacheHit | 36.9–44.9% | 80.2–81.7% |
| MemUnitBusy | 64.3–66.5% | 24.8–25.2% |
| OccupancyPercent | 30.6–30.7% | 36.6–37.0% |
| VALUInsts | 82,205 | 49,693 |
| SALUInsts | 1,673 | 5,828 |

Raw databases and the profiling adapter are in `build/attention-2x/profile/`
and `build/attention-2x/profile_gpu.cpp`. Instruction-counter values use the
profiler's metric definitions, not static instruction counts.

Assembly inspection found redundant probability packing and 64 accumulator
copies per key-loop iteration. An isolated conversion-before-exchange lowering
passes GPU accuracy checks but reaches only about 1.80x in a short run. Combining
it with alternative lane exchange, wider accumulation chains, grouped loop
state, first-iteration peeling, or mandatory multiply operand reuse did not
establish a gain over the leading candidate. The mandatory-reuse experiment
also needed a growing temporary alias list in the allocator. These compiler
changes remain isolated under `build/attention-2x/`; shared installations and
other worktrees were not modified.

Wave64, swizzled 48-key tiles, and direct global V loads likewise passed some
GPU checks but were slower. Rejected sources are archived under
`build/attention-2x/variants/`.

## Compact 64-key tiles and compiler controls

A 64-key design stages XOR-swizzled K (64 × 128 halves) and transposed V
(128 × 64 halves), hoists six Q fragments, and stores the remaining Q channels
in LDS. Total LDS is 43,008 bytes. It consumes each of four probability
fragments across all eight output fragments before constructing the next one.
Q/K/V and probability operands remain fp16; maxima, sums, rescaling and output
accumulation remain fp32.

An LLVM-compiled diagnostic version passes the normal HRX runner's strict
synthetic accuracy checks and measures **1.974x** in 20 alternating pairs at
4,115 tokens (24.56 ms baseline, 12.49 ms candidate, including V transposition).
It uses 233 VGPRs with no spills. This is an algorithm/compiler control, not a
replacement for the Loom deployment kernel or proof of the 2x objective.
The source and report are under
`build/attention-2x/attention_llvm64_outer_aligned.cpp` and
`build/attention-2x/llvm-control/attention_llvm64_outer_aligned/`. The control's
16-half vector type declares 16-byte alignment, matching the padded Q rows.

The Loom port exposed a packed-storage issue: extracting individual fp16
elements and rebuilding vectors generated unnecessary unpacking and repacking.
Structural `vector.slice` and `vector.concat` preserve packed words. An isolated
compiler that also preserves eight-register CFG arguments removes the port's
spills, but the resulting kernel still copies 64 accumulator registers at each
loop backedge. Simplifying every read-side swizzle mask to `(lane_column % 8) * 8`
is valid because every row offset is a multiple of 16, and improves throughput.

| 64-key Loom experiment | VGPRs | Private bytes/thread | Median speedup, 20 pairs |
| --- | ---: | ---: | ---: |
| Structural packed vectors, preserved CFG tuples | 240 | 0 | 1.547x |
| Simplified swizzle mask | 240 | 0 | 1.718x |
| Hoisted LDS addresses | 256 | 16 | 1.401x |
| Simplified mask without PV scheduling barriers | 256 | 64 | 1.425x |

All rows passed the unchanged synthetic oracle and full-output baseline
comparison before and after timing. The simplified-mask result has oracle RMS
0.000283891 and baseline RMS 0.000131668. Its source is
`build/attention-2x/attention64_mask.loom`; reports and binary hashes are under
`build/attention-2x/precompiled/`. These results do not improve the leading
32-key candidate, and no 64-key real-model qualification has been established.

Additional isolated allocator probes identify a live tuple blocking an in-place
update of one consumed scalar slice. Proving that slice dead permits compilation,
but does not remove the accumulator transfers or improve timing. The patches
remain diagnostic and have not been installed into the shared compiler.
CPU tests now cover 64-key boundaries, a late maximum at 4,115 keys, sharp
softmax inputs, and both compact K/V swizzle layouts; all nine tests pass.

## Real model inputs and block composition

A diagnostic library captured fp16 Q/K/V from blocks 0, 13, and 27 using
`build/fixture_step0.pt`, the current INT8 ConvRot checkpoint, and the shipped
attention kernel. The fixture has 4,115 tokens. For each captured input, a CPU
float64 oracle checks 57 query rows across all 48 heads; every candidate output
is also compared with the baseline. These were correctness-only runs, with no
timing claims.

| Block | Baseline vs float64 RMS | Candidate vs float64 RMS | Candidate vs baseline RMS |
| ---: | ---: | ---: | ---: |
| 0 | 0.000809578 | 0.000809814 | 0.0000433772 |
| 13 | 0.000212708 | 0.000213035 | 0.0000492903 |
| 27 | 0.000210591 | 0.000210711 | 0.0000330246 |

The random-input absolute tolerance is not a universal real-input error bound:
the baseline itself fails it against float64 on blocks 0 and 27. The diagnostic
runner therefore reports that failure and checks oracle RMS separately. It
retains the original pointwise bound for candidate/baseline comparison.
Blocks 0 and 13 pass that bound. Block 27 fails at five of 25,282,560 outputs:
near-zero values differ by 0.00244–0.00439, above
`0.002 + 0.002 * abs(baseline)`. The failing report is preserved, rather than
classified as a successful validation. The older 16-key candidate passes the
pointwise baseline comparison on all three captured inputs.

The baseline full-stack output repeats bit-for-bit, and composing its 28
individual blocks exactly reproduces a single forward. Running all 28 candidate
blocks on the same corresponding baseline inputs gives worst update cosine
**0.999938**, maximum update relative RMS **1.112%**, and maximum state relative
RMS **0.428%**. These values are measured differences, not a new acceptance
threshold.

Independently chaining the 32-key candidate gives final-state cosine
**0.999788** and relative RMS **2.057%** versus baseline. The more closely
matching 16-key candidate gives **2.123%** final-state RMS. This supports, but
does not prove, amplification of rounding differences at the model's later
quantization boundaries. It does not establish image-level equivalence.

Sources, captured inputs, logs, same-input block results and full-stack results
are retained under `build/attention-2x/real/` and the adjacent diagnostic scripts.
The normal random-input runner retains its original accuracy criteria.
