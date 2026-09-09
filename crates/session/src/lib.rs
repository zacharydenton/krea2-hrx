//! The 28 Krea 2 transformer blocks as a resident Loom session.
//!
//! Ten launches per block: prepare (RMSNorm, modulation, group-256 Hadamard
//! rotation, per-token int8 quantization) -> fused qkv|gate GEMM -> QK norm and
//! RoPE -> attention -> gated prepare -> wo GEMM with the gated residual ->
//! prepare -> fused gate|up GEMM with the SwiGLU product in its epilogue ->
//! prepare -> down GEMM with the gated residual. The residual stream stays on
//! the device between blocks, in bf16, as ComfyUI keeps it.
//!
//! That chain is identical every forward, so it is recorded once as a graph and
//! replayed. The chain is also genuinely serial -- each stage reads what the one
//! before it wrote -- so recording buys nothing on its own; what it buys is the
//! ability to state where the work is *not* serial, which today is the Sage
//! preparation pass in [`sage`].
#![deny(unsafe_op_in_unsafe_fn)]

pub mod bundle;
pub mod sage;
pub mod weights;

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use hrx::{Buffer, Stream, View};
use kernels::Scalars;

pub use bundle::{Kernels, Metadata};
pub use sage::Sage;
pub use weights::Weights;

/// The model's shape, fixed by the checkpoint.
pub const HIDDEN: i32 = 6144;
pub const KV_HEADS: i32 = 12;
pub const HEAD_DIM: i32 = 128;
pub const INTER: i32 = 16384;
/// wq | wk | wv | attention gate, concatenated into one operand.
pub const QKVG: i32 = HIDDEN + 2 * KV_HEADS * HEAD_DIM + HIDDEN;
/// Where the attention gate's rows start inside that operand.
pub const GATE_OFFSET: i32 = HIDDEN + 2 * KV_HEADS * HEAD_DIM;
const THREADS: u32 = 256;

/// A failure, and whether the caller could have prevented it. Callers use the
/// flag to separate their own mistakes from a genuine runtime failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
    pub invalid_argument: bool,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Error {
        Error { message: message.into(), invalid_argument: true }
    }

    pub fn failed(message: impl Into<String>) -> Error {
        Error { message: message.into(), invalid_argument: false }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<hrx::Error> for Error {
    fn from(error: hrx::Error) -> Self {
        Error::failed(error.to_string())
    }
}

impl From<kernels::Error> for Error {
    fn from(error: kernels::Error) -> Self {
        Error::failed(error.to_string())
    }
}

impl From<krea2_checkpoint::Error> for Error {
    fn from(error: krea2_checkpoint::Error) -> Self {
        // The checkpoint reader's rejections are all about the file the caller
        // named, so they read as invalid arguments.
        Error::failed(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// FNV-1a fingerprint of all RoPE table bytes for upload caching.
fn fingerprint(values: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytemuck::cast_slice::<f32, u8>(values) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// One block's weights, located once in the resident checkpoint.
struct Block {
    qkvg_q: crate::weights::At,
    qkvg_s: crate::weights::At,
    wo_q: crate::weights::At,
    wo_s: crate::weights::At,
    gu_q: crate::weights::At,
    gu_s: crate::weights::At,
    down_q: crate::weights::At,
    down_s: crate::weights::At,
    prenorm: crate::weights::At,
    postnorm: crate::weights::At,
    qnorm: crate::weights::At,
    knorm: crate::weights::At,
}

/// The session's scratch: the residual stream and every intermediate, sized for
/// the sequence and reused by every block.
struct Buffers {
    x: Buffer,
    a_q: Buffer,
    a_s: Buffer,
    fused: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    v_transposed: Option<Buffer>,
    attn: Buffer,
    gu: Buffer,
    mods: Buffer,
    cos: Buffer,
    sin: Buffer,
}

/// A recording in progress: the graph, and the node the next launch comes
/// after. The block loop is a hard serial chain -- every stage reads what the
/// one before it wrote -- so each launch depends on exactly its predecessor.
pub(crate) struct Recording<'g> {
    graph: hrx::Graph<'g>,
    last: Option<hrx::Node>,
}

/// Where a block's launches go: straight to the stream, or into a recording.
///
/// A graph fixes addresses, constants and grids at record time, so one is valid
/// only for the session that recorded it and only while that session's buffers
/// live. Both are the session's own lifetime, which is what `'g` ties together.
enum Sink<'g, 'r> {
    Stream(&'r mut Stream),
    Record(&'r mut Recording<'g>),
}

/// The gate half of the fused QKVG buffer: from its offset to the end.
fn gate_half(fused: &Buffer) -> Result<View<'_>> {
    let at = GATE_OFFSET as usize * 2;
    fused.try_slice(at, fused.bytes() - at).map_err(Error::from)
}

pub struct Session {
    tokens: usize,
    layers: usize,
    metadata: Metadata,
    kernels: Kernels,
    blocks: Vec<Block>,
    /// Declared before the buffers and kernels it records, because fields drop
    /// in declaration order and a graph must be released before the allocations
    /// it names are.
    graph: RefCell<Option<hrx::GraphExec>>,
    buffers: Buffers,
    // Calls on one session are serialized: they share the scratch buffers.
    /// The smoothed attention kernels' preparation pass, when the bundle was
    /// built with one (`KREA2_ATTN_QK` of 4 or 8).
    sage: Option<Sage>,
    /// What the resident RoPE tables were last filled from. Changed tables are
    /// queued before their next use on this stream.
    rope: Cell<Option<(u64, u64)>>,
    /// Opt-in synchronized wall-clock timing per kernel, off during inference.
    profile: AtomicBool,
    stages: RefCell<Vec<(&'static str, f64)>>,
    weights: std::sync::Arc<Weights>,
}

impl Session {
    /// Load a checkpoint and prepare its kernel specializations through HRX.
    pub fn open(
        stream: &mut Stream,
        checkpoint: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<Session> {
        let (file, plan) = Self::validate(checkpoint, tokens, layers)?;
        let weights = std::sync::Arc::new(Weights::upload(stream, &file, plan)?);
        Self::with_weights(stream, weights, tokens, layers, None)
    }

    /// Everything [`Session::open`] checks before it touches the device: the
    /// dimensions this build can serve, and the checkpoint's layout. It takes
    /// no stream, so an invalid argument is refused without a GPU ever being
    /// opened; `open` calls exactly this and then uploads what it returns.
    pub fn validate(
        checkpoint: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<(krea2_checkpoint::Checkpoint, krea2_checkpoint::Plan)> {
        Self::validate_dimensions(tokens, layers)?;
        Weights::plan(checkpoint)
    }

    /// Share resident weights and prepare artifacts for this sequence length.
    /// `compiler` selects a Loom shared library for all session kernels.
    pub fn with_weights(
        stream: &mut Stream,
        weights: std::sync::Arc<Weights>,
        tokens: usize,
        layers: usize,
        compiler: Option<&str>,
    ) -> Result<Session> {
        Self::validate_dimensions(tokens, layers)?;
        let shape = kernels::Shape::from_environment(tokens as i32, weights.bits() as i32)?;
        let bundle = kernels::prepare_for_target(compiler, &shape, stream.target())?;
        Self::build(stream, weights, &bundle, tokens, layers, compiler)
    }

    fn validate_dimensions(tokens: usize, layers: usize) -> Result<()> {
        if !(16..=16896).contains(&tokens) || !(1..=28).contains(&layers) {
            return Err(Error::invalid("tokens must be 16..16896 and layers must be 1..28"));
        }
        Ok(())
    }

    fn build(
        stream: &mut Stream,
        weights: std::sync::Arc<Weights>,
        bundle: &kernels::PreparedBundle,
        tokens: usize,
        layers: usize,
        compiler: Option<&str>,
    ) -> Result<Session> {
        let metadata = Metadata::from(bundle.shape());
        if weights.bits() != metadata.gemm_bits {
            return Err(Error::invalid(format!(
                "kernel bundle built for int{} GEMM operands but the weights are int{}",
                metadata.gemm_bits,
                weights.bits()
            )));
        }
        let bits = metadata.gemm_bits as usize;
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            let p = format!("blocks.{index}");
            let at = |name: &str, bytes: usize| weights.locate(&format!("{p}.{name}"), bytes);
            blocks.push(Block {
                qkvg_q: at("qkvg.q", QKVG as usize * HIDDEN as usize * bits / 8)?,
                qkvg_s: at("qkvg.s", QKVG as usize * 4)?,
                wo_q: at("wo.q", HIDDEN as usize * HIDDEN as usize * bits / 8)?,
                wo_s: at("wo.s", HIDDEN as usize * 4)?,
                gu_q: at("gu.q", 2 * INTER as usize * HIDDEN as usize * bits / 8)?,
                gu_s: at("gu.s", 2 * INTER as usize * 4)?,
                down_q: at("down.q", HIDDEN as usize * INTER as usize * bits / 8)?,
                down_s: at("down.s", HIDDEN as usize * 4)?,
                prenorm: at("prenorm", HIDDEN as usize * 4)?,
                postnorm: at("postnorm", HIDDEN as usize * 4)?,
                qnorm: at("qnorm", HEAD_DIM as usize * 4)?,
                knorm: at("knorm", HEAD_DIM as usize * 4)?,
            });
        }
        let kernels = Kernels::load(stream, bundle, &metadata)?;
        let capacity = metadata.capacity;
        let kv_bytes = KV_HEADS as usize * HEAD_DIM as usize * capacity * 2;
        let allocate = |bytes: usize| stream.allocate(bytes).map_err(Error::from);
        let buffers = Buffers {
            x: allocate(capacity * HIDDEN as usize * 2)?,
            // the widest prepared operand: down's K = 16384 at its padded pitch
            a_q: allocate(capacity * metadata.pitch_inter as usize * bits / 8)?,
            a_s: allocate(capacity * 4)?,
            fused: allocate(capacity * QKVG as usize * 2)?,
            q: allocate(capacity * HIDDEN as usize * 2)?,
            k: allocate(kv_bytes)?,
            v: allocate(kv_bytes)?,
            v_transposed: match metadata.fp16_query_tiles {
                2 => Some(allocate(kv_bytes)?),
                _ => None,
            },
            attn: allocate(capacity * HIDDEN as usize * 2)?,
            // silu(gate) * up, fused into the GEMM epilogue
            gu: allocate(capacity * INTER as usize * 2)?,
            mods: allocate(layers * 6 * HIDDEN as usize * 4)?,
            cos: allocate(capacity * HEAD_DIM as usize * 4)?,
            sin: allocate(capacity * HEAD_DIM as usize * 4)?,
        };
        // Headroom rows are read by the kernels and must be zero, not whatever
        // the allocator last held.
        for buffer in [&buffers.q, &buffers.k, &buffers.v, &buffers.fused, &buffers.x] {
            stream.fill(buffer.binding(), 0)?;
        }
        if let Some(transposed) = &buffers.v_transposed {
            stream.fill(transposed.binding(), 0)?;
        }
        let sage = match metadata.attention_bits {
            16 => None,
            bits => Some(Sage::new(
                stream,
                tokens,
                capacity,
                (KV_HEADS * 4) as usize,
                KV_HEADS as usize,
                bits,
                compiler,
            )?),
        };
        Ok(Session {
            tokens,
            layers,
            metadata,
            kernels,
            blocks,
            buffers,
            sage,
            graph: RefCell::new(None),
            rope: Cell::new(None),
            profile: AtomicBool::new(false),
            stages: RefCell::new(Vec::new()),
            weights,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Turns the per-stage timings on stderr on or off, reporting what they
    /// were. Timing synchronizes after every launch, so it is not free.
    pub fn set_profile(&self, enable: bool) -> bool {
        self.profile.swap(enable, Ordering::Relaxed)
    }

    /// One kernel, into the stream or into a recording. Timing is only possible
    /// on the stream: the synchronize on either side is what makes a stage's
    /// number mean anything, and a replay cannot stop in the middle to take one.
    #[allow(clippy::too_many_arguments)]
    fn launch<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        kernel: &'g hrx::Kernel,
        stage: &'static str,
        grid_x: u32,
        grid_y: u32,
        threads: u32,
        scalars: &Scalars,
        bindings: &[View<'g>],
    ) -> Result<()> {
        let constants = scalars.pack(stage, kernel)?;
        let compiled = kernel.info().workgroup_size;
        if compiled != [threads, 1, 1] {
            return Err(Error::failed(format!(
                "{stage}: compiled for workgroup {compiled:?} but the host asked for {threads}"
            )));
        }
        let grid = [grid_x, grid_y, 1];
        let stream = match sink {
            Sink::Record(recording) => {
                let after = match &recording.last {
                    Some(node) => std::slice::from_ref(node),
                    None => &[],
                };
                // Safety: as the direct dispatch below. The bindings borrow this
                // session, which outlives the graph being recorded.
                let node = unsafe {
                    recording
                        .graph
                        .dispatch(after, kernel, grid, compiled, &constants, bindings)
                }?;
                recording.last = Some(node);
                return Ok(());
            }
            Sink::Stream(stream) => stream,
        };
        if !self.profile.load(Ordering::Relaxed) {
            unsafe { stream.dispatch(kernel, grid, compiled, &constants, bindings) }?;
            return Ok(());
        }
        stream.synchronize()?;
        let began = std::time::Instant::now();
        unsafe { stream.dispatch(kernel, grid, compiled, &constants, bindings) }?;
        stream.synchronize()?;
        self.accumulate(stage, began.elapsed().as_secs_f64() * 1e6);
        Ok(())
    }

    /// Uploads changed RoPE tables. Both run APIs must use this method so the
    /// fingerprint stays consistent with the shared device buffers.
    fn upload_rope(&self, stream: &mut Stream, cos: &[f32], sin: &[f32]) -> Result<()> {
        let fingerprint = (fingerprint(cos), fingerprint(sin));
        if self.rope.get() == Some(fingerprint) {
            return Ok(());
        }
        // Cleared first: a failed upload must not leave the tables claimed.
        self.rope.set(None);
        stream.upload(self.buffers.cos.binding(), bytemuck::cast_slice(cos))?;
        stream.upload(self.buffers.sin.binding(), bytemuck::cast_slice(sin))?;
        self.rope.set(Some(fingerprint));
        Ok(())
    }

    /// The clock a multi-launch stage is timed against, when profiling is on.
    fn timed(&self, stream: &mut Stream) -> Option<std::time::Instant> {
        if !self.profile.load(Ordering::Relaxed) {
            return None;
        }
        let _ = stream.synchronize();
        Some(std::time::Instant::now())
    }

    /// Closes a stage opened by [`Session::timed`].
    fn record(
        &self,
        stream: &mut Stream,
        stage: &'static str,
        began: Option<std::time::Instant>,
    ) -> Result<()> {
        let Some(began) = began else {
            return Ok(());
        };
        stream.synchronize()?;
        self.accumulate(stage, began.elapsed().as_secs_f64() * 1e6);
        Ok(())
    }

    fn accumulate(&self, stage: &'static str, micros: f64) {
        let mut stages = self.stages.borrow_mut();
        match stages.iter_mut().find(|(name, _)| *name == stage) {
            Some(entry) => entry.1 += micros,
            None => stages.push((stage, micros)),
        }
    }

    /// Prints what the run spent where, longest first, and starts over.
    fn report(&self, blocks: usize) {
        if !self.profile.load(Ordering::Relaxed) {
            return;
        }
        let mut stages = self.stages.borrow_mut();
        let total: f64 = stages.iter().map(|(_, micros)| micros).sum();
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("stage profile over {blocks} block(s), {} tokens:", self.tokens);
        for (stage, micros) in stages.iter() {
            eprintln!(
                "  {stage:<24} {:9.3} ms  {:5.1}%",
                micros / 1000.0,
                100.0 * micros / total
            );
        }
        eprintln!("  {:<24} {:9.3} ms", "total", total / 1000.0);
        stages.clear();
    }

    /// Runs a contiguous range of blocks over the caller's residual stream.
    ///
    /// `x` is bf16 `[tokens][6144]`, in and out. `mods` is f32
    /// `[layers][6][6144]`, already including each block's table; `cos` and
    /// `sin` are f32 `[tokens][128]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut Stream,
        x: &mut [u16],
        mods: &[f32],
        cos: &[f32],
        sin: &[f32],
        first_block: usize,
        block_count: Option<usize>,
    ) -> Result<()> {
        let count = block_count.unwrap_or(self.layers.saturating_sub(first_block));
        if first_block >= self.layers || count < 1 || count > self.layers - first_block {
            return Err(Error::invalid("block range must be within the loaded layers"));
        }
        let tokens = self.tokens;
        if x.len() != tokens * HIDDEN as usize {
            return Err(Error::invalid(format!(
                "x has {} elements, expected {}",
                x.len(),
                tokens * HIDDEN as usize
            )));
        }
        if mods.len() != self.layers * 6 * HIDDEN as usize {
            return Err(Error::invalid("mods has the wrong element count"));
        }
        if cos.len() != tokens * HEAD_DIM as usize || sin.len() != tokens * HEAD_DIM as usize {
            return Err(Error::invalid("cos/sin have the wrong element count"));
        }
        stream.upload(self.buffers.x.binding(), bytemuck::cast_slice(x))?;
        stream.upload(self.buffers.mods.binding(), bytemuck::cast_slice(mods))?;
        self.upload_rope(stream, cos, sin)?;
        for index in first_block..first_block + count {
            self.block(&mut Sink::Stream(stream), index, self.buffers.mods.binding())?;
        }
        stream.read_blocking(self.buffers.x.binding(), bytemuck::cast_slice_mut(x))?;
        self.report(count);
        Ok(())
    }

    /// The same over a residual stream and modulation tables already on the
    /// device, which is what the pipeline has: 50 MB of x and 4 MB of mods per
    /// step never leave the GPU.
    ///
    /// `x` is bf16 `[tokens][6144]`, read and written in place; `mods` is f32
    /// `[layers][6][6144]`. The rope tables stay host-side because they change
    /// only when the image geometry does.
    ///
    /// Both views must belong to `stream` and cover the shapes above. The
    /// runtime checks the first and this checks the second, which is why the
    /// address-based version's `unsafe` is gone.
    pub fn run_device(
        &self,
        stream: &mut Stream,
        x: View<'_>,
        mods: View<'_>,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<()> {
        let tokens = self.tokens;
        if cos.len() != tokens * HEAD_DIM as usize || sin.len() != tokens * HEAD_DIM as usize {
            return Err(Error::invalid("cos/sin have the wrong element count"));
        }
        // Into the session's own buffers, whose headroom rows are already zero.
        // `mods` is copied rather than bound directly because a recorded graph
        // fixes its addresses: the caller allocates modulation from a pool and
        // gets a different address each forward, while this one is the
        // session's for its whole life. One 4 MB device copy per forward.
        let rows = tokens * HIDDEN as usize * 2;
        let table = self.buffers.mods.bytes();
        if mods.len() < table {
            return Err(Error::invalid(format!(
                "mods spans {} bytes, expected at least {table}",
                mods.len()
            )));
        }
        stream.copy(self.buffers.x.binding().slice(0, rows)?, x.slice(0, rows)?)?;
        stream.copy(self.buffers.mods.binding(), mods.slice(0, table)?)?;
        self.upload_rope(stream, cos, sin)?;
        self.blocks_through(stream)?;
        // The output copy and subsequent consumers are ordered after the
        // blocks on this session's stream; no host wait is needed here.
        self.report(self.layers);
        stream.copy(x.slice(0, rows)?, self.buffers.x.binding().slice(0, rows)?)?;
        Ok(())
    }

    /// Every block, replayed from a recording when one applies.
    ///
    /// The loop is identical every forward -- same kernels, same buffers, same
    /// constants, same grids -- so it is recorded once per session and replayed
    /// after. Profiling still dispatches directly: it synchronizes between
    /// stages, which a replay cannot stop to do.
    fn blocks_through(&self, stream: &mut Stream) -> Result<()> {
        if self.profile.load(Ordering::Relaxed) {
            for index in 0..self.layers {
                self.block(&mut Sink::Stream(stream), index, self.buffers.mods.binding())?;
            }
            return Ok(());
        }
        let mut recorded = self.graph.borrow_mut();
        if recorded.is_none() {
            let mut recording = Recording { graph: stream.graph()?, last: None };
            for index in 0..self.layers {
                self.block(
                    &mut Sink::Record(&mut recording),
                    index,
                    self.buffers.mods.binding(),
                )?;
            }
            // `finish` ends the graph's borrow of this session and the stream.
            *recorded = Some(recording.graph.finish()?);
        }
        let graph = recorded.as_mut().expect("just recorded");
        stream.launch(graph)?;
        Ok(())
    }

    /// This block's slice of the modulation tables: prescale, preshift,
    /// pregate, postscale, postshift, postgate, each 6144 floats.
    fn modulation<'a>(&self, mods: View<'a>, index: usize, part: usize) -> Result<View<'a>> {
        let row = HIDDEN as usize * 4;
        let stride = 6 * row;
        mods.slice(index * stride + part * row, row).map_err(Error::from)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        kernel: &'g hrx::Kernel,
        stage: &'static str,
        weights: View<'g>,
        scales: View<'g>,
        n: i32,
        out: View<'g>,
        gate: Option<View<'g>>,
    ) -> Result<()> {
        let scalars = Scalars::new().index(self.tokens);
        let bindings =
            [self.buffers.a_q.binding(), weights, scales, self.buffers.a_s.binding(), out];
        let mut bindings = bindings.to_vec();
        if let Some(gate) = gate {
            bindings.push(gate);
        }
        let grid_y = kernels::shape::gemm_grid_rows(
            self.tokens as i32,
            self.metadata.gemm_rows as i32,
            self.metadata.m_group as i32,
        );
        self.launch(
            sink,
            kernel,
            stage,
            (n / 128) as u32,
            grid_y as u32,
            THREADS,
            &scalars,
            &bindings,
        )
    }

    fn block<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        index: usize,
        mods: View<'g>,
    ) -> Result<()> {
        let w = &self.weights;
        let block = &self.blocks[index];
        let tokens = self.tokens;
        let b = &self.buffers;

        let norm_scalars = Scalars::new().index(tokens);
        let norm = [
            b.x.binding(),
            w.view(block.prenorm),
            self.modulation(mods, index, 0)?,
            self.modulation(mods, index, 1)?,
            b.a_q.binding(),
            b.a_s.binding(),
        ];
        self.launch(
            sink,
            &self.kernels.prepare_norm,
            "prepare",
            tokens as u32,
            1,
            THREADS,
            &norm_scalars,
            &norm,
        )?;

        self.gemm(
            sink,
            &self.kernels.gemm_qkvg,
            "gemm qkvg",
            w.view(block.qkvg_q),
            w.view(block.qkvg_s),
            QKVG,
            b.fused.binding(),
            None,
        )?;

        let rope_scalars = Scalars::new().index(tokens);
        let rope = [
            b.fused.binding(),
            w.view(block.qnorm),
            w.view(block.knorm),
            b.cos.binding(),
            b.sin.binding(),
            b.q.binding(),
            b.k.binding(),
            b.v.binding(),
        ];
        self.launch(
            sink,
            &self.kernels.rope,
            "qk norm + rope",
            tokens as u32,
            1,
            THREADS,
            &rope_scalars,
            &rope,
        )?;

        match &self.sage {
            // Smoothed int4/int8 QK: the preparation pass quantizes Q and K
            // against their means and works out the correction the kernel adds
            // back to the scores.
            Some(sage) => {
                // Seven launches, timed as one stage rather than each. Timing
                // needs a stream to synchronize on, so it applies only to the
                // direct path -- which is the only one profiling takes anyway.
                let preparing = match &mut *sink {
                    Sink::Stream(stream) => self.timed(stream),
                    Sink::Record(_) => None,
                };
                sage.run(sink, b.q.binding(), b.k.binding(), b.v.binding())?;
                if let Sink::Stream(stream) = &mut *sink {
                    self.record(stream, "SA2 preprocessing", preparing)?;
                }
                let attention_scalars = Scalars::new().index(tokens).index(KV_HEADS as usize);
                let attention = [
                    sage.q4.binding(),
                    sage.k4.binding(),
                    sage.v_transposed.binding(),
                    sage.q_scale.binding(),
                    sage.k_scale.binding(),
                    sage.correction.binding(),
                    b.attn.binding(),
                ];
                let waves = self.metadata.attention_waves as usize;
                let rows = 16 * (waves / 4);
                self.launch(
                    sink,
                    &self.kernels.attention,
                    "SA2 attention",
                    tokens.div_ceil(rows) as u32,
                    KV_HEADS as u32,
                    32 * waves as u32,
                    &attention_scalars,
                    &attention,
                )?;
            }
            // fp16 QK and PV straight from the RoPE outputs.
            None => {
                let values = match (&b.v_transposed, &self.kernels.attention_transpose) {
                    (Some(transposed), Some(kernel)) => {
                        let scalars = Scalars::new().index(tokens);
                        let bindings = [b.v.binding(), transposed.binding()];
                        self.launch(
                            sink,
                            kernel,
                            "f16 V transpose",
                            tokens.div_ceil(32) as u32,
                            (KV_HEADS * HEAD_DIM / 32) as u32,
                            256,
                            &scalars,
                            &bindings,
                        )?;
                        transposed.binding()
                    }
                    _ => b.v.binding(),
                };
                let attention_scalars = Scalars::new().index(tokens);
                let attention = [b.q.binding(), b.k.binding(), values, b.attn.binding()];
                let rows = 16 * self.metadata.fp16_query_tiles as usize;
                self.launch(
                    sink,
                    &self.kernels.attention,
                    "f16 attention",
                    tokens.div_ceil(rows) as u32,
                    KV_HEADS as u32,
                    128 * self.metadata.fp16_query_tiles,
                    &attention_scalars,
                    &attention,
                )?;
            }
        }

        let gated_scalars = Scalars::new().index(tokens);
        let gated = [b.attn.binding(), gate_half(&b.fused)?, b.a_q.binding(), b.a_s.binding()];
        self.launch(
            sink,
            &self.kernels.prepare_gated,
            "prepare gated",
            tokens as u32,
            1,
            THREADS,
            &gated_scalars,
            &gated,
        )?;

        self.gemm(
            sink,
            &self.kernels.gemm_wo,
            "gemm wo + residual",
            w.view(block.wo_q),
            w.view(block.wo_s),
            HIDDEN,
            b.x.binding(),
            Some(self.modulation(mods, index, 2)?),
        )?;

        let post_scalars = Scalars::new().index(tokens);
        let post = [
            b.x.binding(),
            w.view(block.postnorm),
            self.modulation(mods, index, 3)?,
            self.modulation(mods, index, 4)?,
            b.a_q.binding(),
            b.a_s.binding(),
        ];
        self.launch(
            sink,
            &self.kernels.prepare_norm,
            "prepare",
            tokens as u32,
            1,
            THREADS,
            &post_scalars,
            &post,
        )?;

        self.gemm(
            sink,
            &self.kernels.gemm_gu,
            "gemm gate|up + swiglu",
            w.view(block.gu_q),
            w.view(block.gu_s),
            2 * INTER,
            b.gu.binding(),
            None,
        )?;

        let swiglu_scalars = Scalars::new().index(tokens);
        let swiglu = [b.gu.binding(), b.a_q.binding(), b.a_s.binding()];
        self.launch(
            sink,
            &self.kernels.prepare_swiglu,
            "prepare swiglu",
            tokens as u32,
            1,
            THREADS,
            &swiglu_scalars,
            &swiglu,
        )?;

        self.gemm(
            sink,
            &self.kernels.gemm_down,
            "gemm down + residual",
            w.view(block.down_q),
            w.view(block.down_s),
            HIDDEN,
            b.x.binding(),
            Some(self.modulation(mods, index, 5)?),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host runs must update the RoPE fingerprint when tables change A → B → A.
    /// Requires a local checkpoint and GPU. The bundle is compiled by HRX.
    #[test]
    #[ignore = "requires Krea checkpoint and gfx1151"]
    fn every_path_that_fills_the_rope_buffers_records_what_it_put_there() {
        let (checkpoint, tokens) = fixture();
        let mut stream = Stream::open().expect("a stream");
        let session =
            Session::open(&mut stream, &checkpoint, tokens, 1).expect("resident session");
        let resident = || session.rope.get();
        assert_eq!(resident(), None, "nothing is resident before the first call");

        let mods = vec![0.01f32; 6 * HIDDEN as usize];
        let a = (vec![1.0f32; tokens * 128], vec![0.0f32; tokens * 128]);
        let b = (vec![0.0f32; tokens * 128], vec![1.0f32; tokens * 128]);
        let mut x = vec![0x3f00u16; tokens * HIDDEN as usize];

        session.run(&mut stream, &mut x, &mods, &a.0, &a.1, 0, Some(1)).expect("A");
        let after_a = resident();
        assert!(after_a.is_some(), "A left nothing recorded");

        session.run(&mut stream, &mut x, &mods, &b.0, &b.1, 0, Some(1)).expect("B");
        assert_ne!(resident(), after_a, "B's upload was not recorded, so A looks resident");

        session.run(&mut stream, &mut x, &mods, &a.0, &a.1, 0, Some(1)).expect("A again");
        assert_eq!(resident(), after_a, "A's second upload was not recorded");
    }

    /// A recorded replay must produce exactly what direct dispatch produces,
    /// and must actually be taken: the graph is only populated by the path
    /// under test, so an empty one means the recording never happened.
    #[test]
    #[ignore = "requires Krea checkpoint and gfx1151"]
    fn a_recorded_block_loop_replays_to_the_same_bytes() {
        let (checkpoint, tokens) = fixture();
        let mut stream = Stream::open().expect("a stream");
        let session =
            Session::open(&mut stream, &checkpoint, tokens, 2).expect("resident session");

        let mods = vec![0.01f32; 2 * 6 * HIDDEN as usize];
        let cos = vec![1.0f32; tokens * HEAD_DIM as usize];
        let sin = vec![0.0f32; tokens * HEAD_DIM as usize];
        let start: Vec<u16> = (0..tokens * HIDDEN as usize)
            .map(|i| 0x3c00u16.wrapping_add((i % 97) as u16))
            .collect();

        // The direct path: profiling forces every launch onto the stream.
        let mut host = start.clone();
        session.set_profile(true);
        session.run(&mut stream, &mut host, &mods, &cos, &sin, 0, None).expect("direct");
        session.set_profile(false);
        assert!(session.graph.borrow().is_none(), "profiling must not record");

        // The recorded path, through the device entry point.
        let x = stream.allocate(tokens * HIDDEN as usize * 2).expect("x");
        let table = stream.allocate(mods.len() * 4).expect("mods");
        stream.upload(x.binding(), bytemuck::cast_slice(&start)).expect("x upload");
        stream.upload(table.binding(), bytemuck::cast_slice(&mods)).expect("mods upload");
        session
            .run_device(&mut stream, x.binding(), table.binding(), &cos, &sin)
            .expect("recorded");
        assert!(session.graph.borrow().is_some(), "the block loop was not recorded");

        let mut replayed = vec![0u16; start.len()];
        stream
            .read_blocking(x.binding(), bytemuck::cast_slice_mut(&mut replayed))
            .expect("readback");
        assert_eq!(replayed, host, "replay diverged from direct dispatch");

        // A second forward reuses the recording rather than rebuilding it.
        session
            .run_device(&mut stream, x.binding(), table.binding(), &cos, &sin)
            .expect("replayed");
        // Nothing drains here on purpose: the session, its graph and the
        // buffers that graph recorded are all released with the replay still in
        // flight, which must be safe.
    }

    fn fixture() -> (std::path::PathBuf, usize) {
        use std::path::PathBuf;
        let checkpoint =
            std::env::var_os("KREA2_MODEL").map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join("comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors")
            });
        assert!(checkpoint.is_file(), "set KREA2_MODEL to a local checkpoint");
        let tokens = 4115;
        (checkpoint, tokens)
    }
}
