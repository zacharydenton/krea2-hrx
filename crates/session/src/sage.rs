//! The preparation pass the smoothed int4/int8 attention kernels take.
//!
//! Sage attention quantizes Q and K per token after subtracting a mean, and
//! recovers what the subtraction cost with a correction term the kernel adds
//! back to the scores. That is six kernels and a GEMM per block, which is why
//! the buffers are allocated once for the session's shape rather than per call.
//!
//! The seven are not a chain. The key mean and the query mean are computed from
//! disjoint buffers, so the two quantizations are independent of each other and
//! the V transpose of both; only the correction GEMM needs results from both
//! sides. Recorded as a graph that is a critical path of four rather than seven,
//! which is free and true but not currently faster -- these grids already fill
//! the device. `the_declared_concurrency_is_priced_against_a_chain` prices it.
//!
//! The nibble layout is head-major, `[heads][capacity][64 bytes]` for int4 and
//! 128 for int8, with `[heads][capacity]` float32 scales, so one key tile is
//! one contiguous block. The correction is
//! `[query heads][ceil(tokens / 64)][capacity]` in float32.
use hrx::{Buffer, Stream, View};
use kernels::Scalars;
use kernels::{cache::PreparedKernels, Config};

use crate::{Error, Result, Sink};

/// One of the seven launches: everything about it the shape fixes. Only the
/// bindings depend on the call, and only on which buffers the caller hands in.
struct Plan {
    name: &'static str,
    config: Config,
    scalars: Scalars,
    grid: (u32, u32),
    threads: u32,
}

/// Buffers and shapes for one sequence length, reused every block.
pub struct Sage {
    /// Resolved once, so they outlive a recording: `Graph::dispatch` borrows a
    /// kernel for the life of the graph, and a per-call cache lookup hands back
    /// a value that would not live that long.
    prepared: Vec<(Plan, hrx::Kernel)>,
    pub q4: Buffer,
    pub k4: Buffer,
    pub q_scale: Buffer,
    pub k_scale: Buffer,
    pub correction: Buffer,
    pub v_transposed: Buffer,
    key_partial: Buffer,
    key_mean: Buffer,
    query_mean: Buffer,
    query_mean_half: Buffer,
    centered_k: Buffer,
}

/// The shapes these kernels serve, checked before anything is allocated, and
/// the tile count that follows from them. Separate from [`Sage::new`] so it can
/// be tested without a stream, which naming a buffer now requires.
fn dimensions(
    tokens: usize,
    capacity: usize,
    heads: usize,
    kv_heads: usize,
    bits: u32,
) -> Result<usize> {
    let tiles = tokens.div_ceil(64);
    if !(16..=16896).contains(&tokens)
        || capacity < tiles * 64
        || !capacity.is_multiple_of(32)
        || heads < 1
        || kv_heads < 1
        || !heads.is_multiple_of(kv_heads)
        || heads / kv_heads != 4
        || (bits != 4 && bits != 8)
    {
        return Err(Error::invalid("unsupported Sage dimensions"));
    }
    Ok(tiles)
}

impl Sage {
    /// `bits` is 4 (codes -7..7, 64 bytes per head row) or 8 (-127..127, 128);
    /// the attention kernel of the same width consumes the output.
    pub fn new(
        stream: &mut Stream,
        tokens: usize,
        capacity: usize,
        heads: usize,
        kv_heads: usize,
        bits: u32,
        compiler: Option<&str>,
    ) -> Result<Sage> {
        let tiles = dimensions(tokens, capacity, heads, kv_heads, bits)?;
        // Resolved up front, not per block: a recording borrows each kernel for
        // its whole life, and these are the same seven every time.
        let cache = PreparedKernels::new(compiler);
        let mut prepared = Vec::with_capacity(7);
        for plan in Sage::plans(tokens, capacity, heads, kv_heads, tiles, bits) {
            let kernel = cache.get(stream, plan.name, plan.config.clone(), plan.grid)?;
            prepared.push((plan, kernel));
        }
        // One head's codes: a nibble or a byte per channel of 128.
        let row_bytes = if bits == 4 { 64 } else { 128 };
        let sage = Sage {
            prepared,
            q4: stream.allocate(capacity * heads * row_bytes)?,
            k4: stream.allocate(capacity * kv_heads * row_bytes)?,
            q_scale: stream.allocate(capacity * heads * 4)?,
            k_scale: stream.allocate(capacity * kv_heads * 4)?,
            correction: stream.allocate(heads * tiles * capacity * 4)?,
            v_transposed: stream.allocate(kv_heads * capacity * 128 * 2)?,
            key_partial: stream.allocate(tiles * kv_heads * 128 * 4)?,
            key_mean: stream.allocate(kv_heads * 128 * 4)?,
            query_mean: stream.allocate(heads * tiles * 128 * 4)?,
            query_mean_half: stream.allocate(heads * tiles * 128 * 2)?,
            centered_k: stream.allocate(kv_heads * capacity * 128 * 2)?,
        };
        // The padding past `tokens` is never written but is read as codes, so
        // it starts at zero rather than at whatever the allocator held.
        stream.fill(sage.q4.binding(), 0)?;
        stream.fill(sage.q_scale.binding(), 0)?;
        stream.fill(sage.v_transposed.binding(), 0)?;
        Ok(sage)
    }

    /// The seven launches, in the order they are recorded. Everything here is
    /// derived from the shape, so it is resolved once in `new` rather than
    /// rebuilt per block.
    fn plans(
        tokens: usize,
        capacity: usize,
        heads: usize,
        kv_heads: usize,
        tiles: usize,
        bits: u32,
    ) -> [Plan; 7] {
        let (t, c, h, kv) = (tokens, capacity, heads, kv_heads);
        let config = |entries: &[(&str, usize)]| -> Config {
            entries.iter().map(|(key, value)| ((*key).to_string(), *value as u64)).collect()
        };
        let (quant_q, quant_k) = match bits {
            4 => ("sage_quant_q", "sage_quant_k"),
            _ => ("sage_quant_q_i8", "sage_quant_k_i8"),
        };
        // The correction: the query means against the centered keys, which is
        // what the attention kernel adds back to each score.
        let (m, n) = (tiles * 4, c);
        let (a_stride, b_stride) = (m * 128, n * 128);
        [
            // The key mean, over the whole sequence: a partial sum per tile,
            // then one workgroup per head to finish it.
            Plan {
                name: "sage_key_partial",
                config: config(&[
                    ("tokens", t),
                    ("heads", kv),
                    ("tiles", tiles),
                    ("xsize", c * kv * 128),
                    ("ysize", tiles * kv * 128),
                ]),
                scalars: Scalars::new().index(t),
                grid: (kv as u32, tiles as u32),
                threads: 128,
            },
            Plan {
                name: "sage_key_mean",
                config: config(&[
                    ("tokens", t),
                    ("heads", kv),
                    ("tiles", tiles),
                    ("xsize", tiles * kv * 128),
                    ("ysize", kv * 128),
                ]),
                scalars: Scalars::new().index(t),
                grid: (kv as u32, 1),
                threads: 128,
            },
            // The query mean is per tile, not per sequence: each query tile
            // only ever meets the keys once.
            Plan {
                name: "sage_query_mean",
                config: config(&[
                    ("tokens", t),
                    ("heads", h),
                    ("tiles", tiles),
                    ("xsize", c * h * 128),
                    ("ysize", h * tiles * 128),
                ]),
                scalars: Scalars::new().index(t),
                grid: (tiles as u32, h as u32),
                threads: 128,
            },
            Plan {
                name: quant_q,
                config: config(&[
                    ("tokens", t),
                    ("heads", h),
                    ("tiles", tiles),
                    ("capacity", c),
                    ("xsize", c * h * 128),
                    ("msize", h * tiles * 128),
                    ("psize", c * h * 32),
                    ("ssize", c * h),
                ]),
                scalars: Scalars::new().index(t),
                grid: (t.div_ceil(8) as u32, h as u32),
                threads: 256,
            },
            Plan {
                name: quant_k,
                config: config(&[
                    ("tokens", t),
                    ("heads", kv),
                    ("tiles", tiles),
                    ("capacity", c),
                    ("xsize", c * kv * 128),
                    ("msize", kv * 128),
                    ("psize", c * kv * 32),
                    ("ssize", c * kv),
                ]),
                scalars: Scalars::new().index(t),
                // Over the capacity, not the tokens: the padded rows are read.
                grid: (c.div_ceil(8) as u32, kv as u32),
                threads: 256,
            },
            Plan {
                name: "sage_transpose",
                config: config(&[("width", kv * 128), ("row_capacity", c)]),
                scalars: Scalars::new().index(t),
                grid: (t.div_ceil(32) as u32, (kv * 128 / 32) as u32),
                threads: 256,
            },
            Plan {
                name: "gemm_f16_f32_nt",
                config: config(&[
                    ("m", m),
                    ("n", n),
                    ("k", 128),
                    ("asize", a_stride * kv),
                    ("bsize", b_stride * kv),
                    ("csize", m * n * kv),
                    ("astride", a_stride),
                    ("bstride", b_stride),
                ]),
                scalars: Scalars::new().index(m).float(1.0),
                grid: (n.div_ceil(64) as u32, (kv * m.div_ceil(64)) as u32),
                threads: 256,
            },
        ]
    }

    /// Prepares one block's Q, K and V, in the session's stream order.
    ///
    /// Recorded, the seven form a diamond rather than a chain: the key side
    /// (partial, mean, quantize) and the query side (mean, quantize) touch
    /// disjoint buffers, the V transpose touches neither, and only the
    /// correction GEMM reads from both sides. The join at the end is what the
    /// attention launch after this one depends on.
    pub(crate) fn run<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        q: View<'g>,
        k: View<'g>,
        v: View<'g>,
    ) -> Result<()> {
        self.run_after(sink, q, k, v, &Sage::DIAMOND)
    }

    /// Which already-recorded step each one waits for. The indices match
    /// `plans`: key partial, key mean, query mean, quantize q, quantize k,
    /// transpose, correction. Stating only the edges that exist is the point of
    /// recording at all; a benchmark passes a chain instead to price it.
    const DIAMOND: [&'static [usize]; 7] = [&[], &[0], &[], &[2], &[1], &[], &[2, 4]];

    fn run_after<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        q: View<'g>,
        k: View<'g>,
        v: View<'g>,
        after_table: &[&'static [usize]; 7],
    ) -> Result<()> {
        let bindings: [Vec<View<'g>>; 7] = [
            vec![k, self.key_partial.binding()],
            vec![self.key_partial.binding(), self.key_mean.binding()],
            vec![q, self.query_mean.binding(), self.query_mean_half.binding()],
            vec![q, self.query_mean.binding(), self.q4.binding(), self.q_scale.binding()],
            vec![
                k,
                self.key_mean.binding(),
                self.k4.binding(),
                self.k_scale.binding(),
                self.centered_k.binding(),
            ],
            vec![v, self.v_transposed.binding()],
            vec![
                self.query_mean_half.binding(),
                self.centered_k.binding(),
                self.correction.binding(),
            ],
        ];

        let recording = match sink {
            Sink::Stream(stream) => {
                for ((plan, kernel), operands) in self.prepared.iter().zip(&bindings) {
                    let constants = plan.scalars.pack(plan.name, kernel)?;
                    self.check(plan, kernel)?;
                    // Safety: the bindings match the shape each plan was
                    // compiled for, which `new` derived from the same fields.
                    unsafe {
                        stream.dispatch(
                            kernel,
                            [plan.grid.0, plan.grid.1, 1],
                            [plan.threads, 1, 1],
                            &constants,
                            operands,
                        )
                    }?;
                }
                return Ok(());
            }
            Sink::Record(recording) => recording,
        };

        let entry = recording.last;
        let mut nodes: Vec<hrx::Node> = Vec::with_capacity(7);
        for (index, ((plan, kernel), operands)) in
            self.prepared.iter().zip(&bindings).enumerate()
        {
            let constants = plan.scalars.pack(plan.name, kernel)?;
            self.check(plan, kernel)?;
            let after: Vec<hrx::Node> = match after_table[index] {
                // A root waits only for whatever produced q, k and v.
                [] => entry.into_iter().collect(),
                dependencies => dependencies.iter().map(|&at| nodes[at]).collect(),
            };
            // Safety: as the direct dispatch above.
            let node = unsafe {
                recording.graph.dispatch(
                    &after,
                    kernel,
                    [plan.grid.0, plan.grid.1, 1],
                    [plan.threads, 1, 1],
                    &constants,
                    operands,
                )
            }?;
            nodes.push(node);
        }
        // The leaves: everything the attention kernel reads was written by the
        // two quantizations, the transpose or the correction.
        let leaves = [nodes[3], nodes[4], nodes[5], nodes[6]];
        recording.last = Some(recording.graph.join(&leaves)?);
        Ok(())
    }

    /// The runtime validates the block against what the kernel was compiled
    /// with; disagreeing with it here names the stage instead of the export.
    fn check(&self, plan: &Plan, kernel: &hrx::Kernel) -> Result<()> {
        let compiled = kernel.info().workgroup_size;
        if compiled != [plan.threads, 1, 1] {
            return Err(Error::failed(format!(
                "{}: compiled for workgroup {compiled:?} but the host asked for {}",
                plan.name, plan.threads
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Recording;

    /// What the declared concurrency is worth, in isolation and with no
    /// checkpoint: the same seven launches recorded as the real diamond and as
    /// a chain, replayed back to back. Prints rather than asserts -- the answer
    /// depends on whether these grids leave the GPU anything to overlap with.
    #[test]
    #[ignore = "requires gfx1151; a benchmark, not an assertion"]
    fn the_declared_concurrency_is_priced_against_a_chain() {
        const CHAIN: [&[usize]; 7] = [&[], &[0], &[1], &[2], &[3], &[4], &[5]];
        let (tokens, capacity, heads, kv) = (4115usize, 4160usize, 48usize, 12usize);
        let mut stream = Stream::open().expect("a stream");
        let sage = Sage::new(&mut stream, tokens, capacity, heads, kv, 8, None)
            .expect("sage for the session shape");
        let q = stream.allocate(capacity * heads * 128 * 2).expect("q");
        let k = stream.allocate(capacity * kv * 128 * 2).expect("k");
        let v = stream.allocate(capacity * kv * 128 * 2).expect("v");
        for buffer in [&q, &k, &v] {
            stream.fill(buffer.binding(), 0x11).expect("fill");
        }

        // 28 repetitions, as one forward runs it, so the graph is the size the
        // block loop actually records.
        let record = |table: &[&'static [usize]; 7]| {
            let mut recording = Recording { graph: stream.graph().expect("graph"), last: None };
            for _ in 0..28 {
                sage.run_after(
                    &mut Sink::Record(&mut recording),
                    q.binding(),
                    k.binding(),
                    v.binding(),
                    table,
                )
                .expect("record");
            }
            recording.graph.finish().expect("instantiate")
        };
        let mut diamond = record(&Sage::DIAMOND);
        let mut chain = record(&CHAIN);

        let mut time = |graph: &mut hrx::GraphExec| {
            stream.synchronize().expect("drain");
            let began = std::time::Instant::now();
            for _ in 0..8 {
                stream.launch(graph).expect("replay");
            }
            stream.synchronize().expect("drain");
            began.elapsed().as_secs_f64() * 1e3 / 8.0
        };
        for round in 0..4 {
            let (d, c) = (time(&mut diamond), time(&mut chain));
            eprintln!("round {round}: diamond {d:.3} ms, chain {c:.3} ms");
        }
    }

    #[test]
    fn the_shapes_the_kernels_cannot_serve_are_refused() {
        // No device is touched: these fail on arithmetic alone, which is why
        // the check is a free function rather than the first act of `new`.
        for (tokens, capacity, heads, kv, bits, why) in [
            (8usize, 64usize, 48usize, 12usize, 4u32, "too few tokens"),
            (64, 64, 48, 12, 5, "an unsupported width"),
            (64, 48, 48, 12, 4, "a capacity below the tiles"),
            (64, 66, 48, 12, 4, "a capacity off the 32-row grid"),
            (64, 64, 24, 12, 4, "a group size that is not four"),
        ] {
            let Err(error) = dimensions(tokens, capacity, heads, kv, bits) else {
                panic!("{why} was accepted");
            };
            assert!(error.message.contains("unsupported Sage dimensions"), "{why}");
        }
        assert!(dimensions(4115, 4160, 48, 12, 4).is_ok(), "the session's own shape");
    }
}
