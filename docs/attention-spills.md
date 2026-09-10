# Attention register spills

The default `krea2_attention_gqa_lds_f16_wmma` kernel exhausted its 256 VGPRs
with the pinned Loom compiler. Its twelve spill slots reserved 48 bytes of
scratch per lane. The key-tile loop contained 20 scalar scratch loads and six
scalar scratch stores, repeated for each of the 258 key tiles at 4,115 tokens.

The compiler diagnostics name the lane/staging indices and the LDS scratch
address among the spilled values. Disassembly also shows spilled running
softmax and accumulator values. Loading all eight V fragments before the PV
matrix multiplies makes those fragments compete with resident Q, the running
output accumulators, and softmax state. Reordering the source to consume each
fragment immediately still spills: the scheduler hoists independent LDS loads.

The kernel now consumes V in pairs, with three subgroup memory fences between
pairs. These bound the load hoisting without adding a workgroup rendezvous.
The compiled kernel uses 248 VGPRs, zero private scratch, and the same 21,760
bytes of LDS. Each accumulator retains its original rescale and matrix-multiply
order. The QK reduction, softmax, precision, and tile layout are unchanged.

The alternatives measured were source reordering alone (still twelve spills),
one V fragment per fence (zero spills), four per fence (eight spills), and staging
all Q in LDS (zero spills, but 29,952 bytes of LDS). Pairs were consistently
competitive with singles and faster at the longest timed sequence, while needing
only three added fences instead of seven.

Measurements use the Radeon 8060S / gfx1151, `hrx-rs` 0.1.0, and compiler SHA-256
`e66da21cc469071391a54055044bf5202a6c3d36eca4556f22d067f4e44b3fc3`.
Each sample times ten dispatches followed by stream synchronization. Eight
rounds rotate the candidate order; the first round is discarded. Inputs are
identical deterministic fp16 Q/K/V with 48 query heads, 12 KV heads, and 128
channels. Compilation, allocation, transfers, and output comparison are outside
the timed region. Results are isolated kernel latency, not whole-image speedups.

| Tokens | Before (ms) | Paired V (ms) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1,043 | 2.000554 | 1.346553 | 32.7% |
| 4,115 | 28.460474 | 19.799810 | 30.4% |
| 9,235 | 222.051548 | 160.167671 | 27.9% |

All output elements matched the original kernel bit for bit at these three
sequence lengths. The [raw measurements](benchmarks/attention-spills-2026-09-09.json)
retain the samples and alternative variants. Clock and thermal state affect
absolute timings; compare candidates within a run.

Reproduce a comparison with an original source file and the current kernel:

```sh
git show f5c7c82:kernels/attention_gqa_lds_f16_wmma.loom > before.loom
cargo run --release --example attention -- \
  4115 /tmp/attention-report before.loom \
  kernels/attention_gqa_lds_f16_wmma.loom
```

The Rust example verifies exact output equality and writes each HSACO, compiler
manifest, and diagnostic list. Inspect resources with `llvm-readelf --notes` and
instructions with `llvm-objdump -d --mcpu=gfx1151`. The ignored Rust regression
test `production_attention_has_no_scratch_spills` checks compiler diagnostics
at 1,043, 4,115, 9,235, and 16,403 tokens. The existing independent CPU softmax
test covers partial tiles, and `scripts/parity.sh` checks the complete
1024×1024 trajectory and native RGB against the frozen unquantized BF16 model.
All passed. The full trajectory retained relative RMS 0.076912 and native image
PSNR 31.777828 dB, with 0.000000 dB regression against the accepted baseline.

## End-to-end generation

For 1024×1024 Turbo at eight steps, the warm prompt-to-PNG median was **31.045 s
before and 29.535 s after**, saving about **1.51 s (4.9%)**. These are six warm
images per variant, including noise generation, prompt encoding/text fusion,
denoising, VAE decode, RGB readback, PNG encoding, and file writing. They exclude
model loading and each process's first image, which also prepares lazy state.

Both binaries used the same frozen working-tree snapshot, with only the embedded
attention source changed. Four processes ran in before/after/after/before order,
each producing seeds 0–3; seed 0 was the warm-up. Profiling and progress callbacks
were disabled. The prompt was the frozen fixture's red fox prompt, but these runs
used the complete native text encoder and seeded sampler rather than saved
conditioning or noise. All sixteen PNGs were byte-identical for their matching
seed across variants and rounds.

| Batch | Variant | Warm image times (s) | Median (s) |
| ---: | --- | --- | ---: |
| 1 | Before | 30.03, 30.02, 30.02 | 30.02 |
| 2 | After | 28.82, 29.76, 29.95 | 29.76 |
| 3 | After | 29.48, 29.52, 29.55 | 29.52 |
| 4 | Before | 32.08, 32.07, 32.06 | 32.07 |

The baseline drifted between batches, so interpret the aggregate as roughly a
5% end-to-end improvement for this workload. The isolated kernel's 30% reduction
does not translate into a 30% image-generation reduction. Native artifact and OS
file caches were not cleared; model-load and first-image differences are not
attributed to the attention change. The [raw end-to-end record](benchmarks/attention-e2e-2026-09-09.json)
includes every sample, load/process time, binary hashes, and output hashes.
