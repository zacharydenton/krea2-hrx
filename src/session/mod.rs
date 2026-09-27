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
//! replayed. Measurements have not established an end-to-end speedup. See
//! [graph recording](https://github.com/zacharydenton/krea2-hrx/blob/master/docs/graph-recording.md)
//! for the benchmark scope and replay constraints.

pub mod bundle;
pub mod sage;
pub mod weights;

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::kernels::{Attention, Scalars, Shape, shape};
use hrx::{Buffer, Stream, View};

pub use bundle::Kernels;
pub use sage::Sage;
use sage::VTranspose;
pub use weights::Weights;

pub use crate::{Error, Result};

/// The residual stream's width. This and the constants below are the
/// model's shape, fixed by the checkpoint.
pub const HIDDEN: usize = 6144;
/// Key and value heads; each serves four of the 48 query heads.
pub const KV_HEADS: usize = 12;
/// Channels per attention head.
pub const HEAD_DIM: usize = 128;
/// The SwiGLU MLP's inner width.
pub const INTER: usize = 16384;
/// wq | wk | wv | attention gate, concatenated into one operand.
pub const QKVG: usize = HIDDEN + 2 * KV_HEADS * HEAD_DIM + HIDDEN;
/// Where the attention gate's rows start inside that operand.
pub const GATE_OFFSET: usize = HIDDEN + 2 * KV_HEADS * HEAD_DIM;
const THREADS: u32 = 256;

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
    qkvg_q: self::weights::At,
    qkvg_s: self::weights::At,
    wo_q: self::weights::At,
    wo_s: self::weights::At,
    gu_q: self::weights::At,
    gu_s: self::weights::At,
    down_q: self::weights::At,
    down_s: self::weights::At,
    prenorm: self::weights::At,
    postnorm: self::weights::At,
    qnorm: self::weights::At,
    knorm: self::weights::At,
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
    // Attention is consumed by gated prepare before gate/up writes its result.
    // Neither value has persistent zero-filled headroom rows.
    attn_gu: Buffer,
    mods: Buffer,
    cos: Buffer,
    sin: Buffer,
}

/// Dependencies for the next dispatch. A fan-in goes on its consumer directly,
/// without inserting an empty native graph node.
#[derive(Clone, Copy, Default)]
enum After {
    #[default]
    None,
    One(hrx::Node),
    Three([hrx::Node; 3]),
}

impl After {
    fn as_slice(&self) -> &[hrx::Node] {
        match self {
            Self::None => &[],
            Self::One(node) => std::slice::from_ref(node),
            Self::Three(nodes) => nodes,
        }
    }
}

/// A recording in progress, the dependencies of its next launch, and a label
/// for every launch so far: a profiled graph reports one interval per launch.
pub(crate) struct Recording<'g> {
    graph: hrx::Graph<'g>,
    last: After,
    labels: Vec<&'static str>,
}

impl<'g> Recording<'g> {
    fn new(stream: &'g Stream) -> Result<Recording<'g>> {
        Ok(Recording { graph: stream.graph()?, last: After::None, labels: Vec::new() })
    }
}

/// Where a block's launches go: straight to the stream, or into a recording.
///
/// A graph fixes addresses, constants and grids at record time, so one is valid
/// only for the session that recorded it and only while that session's buffers
/// live. Both are the session's own lifetime, which is what `'g` ties together.
pub(crate) enum Sink<'g, 'r> {
    Stream(&'r mut Stream),
    Record(&'r mut Recording<'g>),
}

/// This block's slice of the modulation tables: prescale, preshift, pregate,
/// postscale, postshift, postgate, each 6144 floats.
fn modulation(mods: View<'_>, index: usize, part: usize) -> Result<View<'_>> {
    let row = HIDDEN * 4;
    Ok(mods.slice(index * 6 * row + part * row, row)?)
}

/// The gate half of the fused QKVG buffer: from its offset to the end.
fn gate_half(fused: &Buffer) -> Result<View<'_>> {
    let at = GATE_OFFSET * 2;
    fused.try_slice(at, fused.bytes() - at).map_err(Error::from)
}

/// Resident block workspace and recording, used on the stream that created it.
pub struct Session {
    stream_id: usize,
    tokens: usize,
    layers: usize,
    shape: Shape,
    kernels: Kernels,
    blocks: Vec<Block>,
    /// The native graph retains its recorded resources until its final replay
    /// completes. These fixed session buffers are never returned to a pool.
    graph: RefCell<Option<hrx::GraphExec>>,
    buffers: Buffers,
    // Calls on one session are serialized: they share the scratch buffers.
    /// The smoothed attention kernels' preparation pass, when the bundle was
    /// built with one (`KREA2_ATTN_QK` of 4 or 8).
    sage: Option<Sage>,
    /// V transposed for the fp16 attention kernel; `None` beside `sage`,
    /// which transposes V itself.
    v_transposed: Option<VTranspose>,
    /// What the resident RoPE tables were last filled from. Changed tables are
    /// queued before their next use on this stream.
    rope: Cell<Option<(u64, u64)>>,
    /// Per-kernel timing, from `KREA2_NATIVE_PROFILE` when the session is
    /// built or [`Session::set_profile`] after. Profiled forwards replay a
    /// graph with GPU-clock markers, or dispatch directly between host
    /// synchronizations where the runtime has no device clock.
    profile: AtomicBool,
    stages: RefCell<Vec<(&'static str, f64)>>,
    /// The block loop recorded with device-clock markers, and the stage of
    /// each interval it reports. Built on the first profiled forward.
    profiled: RefCell<Option<(hrx::GraphExec, Vec<&'static str>)>>,
    /// Why device-clock profiling is unavailable, once it has been refused;
    /// profiling then times each kernel between host synchronizations.
    device_clock_refused: RefCell<Option<String>>,
    weights: Arc<Weights>,
}

impl Session {
    /// Move an existing fixed-shape session and its stream into the shared
    /// scheduler. One private slot preserves the session's graph and RoPE cache.
    /// Inputs are BF16 `[tokens,6144]` and F32 `[layers,6,6144]`; the BF16
    /// residual output is independently owned and retains the slot while live.
    pub fn into_prepared(
        self,
        context: &hrx::inference::ModelContext,
        mut stream: Stream,
        cos: Vec<f32>,
        sin: Vec<f32>,
    ) -> Result<hrx::inference::PreparedModel> {
        use hrx::{
            Access,
            execution::GpuAccess,
            inference::{InferenceGraph, PreparedModel},
            tensor::{DType, Layout, TensorDesc},
        };
        self.check_stream(&stream)?;
        if stream.device_id()
            != hrx::Device::open(context.runtime().gpu()?.index())?.stream()?.device_id()
        {
            return Err(Error::invalid("session belongs to another model context device"));
        }
        if cos.len() != self.tokens * HEAD_DIM || sin.len() != cos.len() {
            return Err(Error::invalid("cos/sin have the wrong element count"));
        }
        self.upload_rope(&mut stream, &cos, &sin)?;
        // Record before handing the stream to workers; no compilation or graph
        // preparation is deferred to warm inference.
        if self.graph.borrow().is_none() {
            *self.graph.borrow_mut() = Some(self.record_graph(&stream)?);
        }
        stream.synchronize()?;
        let x = TensorDesc::new(DType::BF16, vec![self.tokens, HIDDEN])?
            .with_layout(Layout::Rows)?;
        let mods = TensorDesc::new(DType::F32, vec![self.layers, 6, HIDDEN])?;
        let mut owned = Some((self, stream, cos, sin));
        Ok(PreparedModel::prepare(context, 1, |context| {
            let inputs = vec![context.allocate(x.clone())?, context.allocate(mods.clone())?];
            let outputs = vec![context.allocate(x.clone())?];
            let (session, mut stream, cos, sin) = owned.take().expect("one fixed-shape slot");
            let bindings = [
                GpuAccess { view: inputs[0].binding().unwrap(), access: Access::Read },
                GpuAccess { view: inputs[1].binding().unwrap(), access: Access::Read },
                GpuAccess { view: outputs[0].binding().unwrap(), access: Access::Write },
            ];
            let mut graph = context.runtime().graph();
            // Safety: the closure owns the only session/stream and retains all
            // graph buffers and weights. Only declared views escape the private
            // workspace, and synchronization drains every access before return.
            unsafe {
                graph.gpu_scoped(&bindings, move |views| {
                    stream.copy(views[2], views[0])?;
                    session
                        .run_device(&mut stream, views[2], views[1], &cos, &sin)
                        .map_err(hrx::Error::from)?;
                    stream.synchronize()
                })?;
            }
            Ok(InferenceGraph { inputs, outputs, graph: graph.prepare()? })
        })?)
    }

    /// Load a checkpoint and prepare its kernel specializations through HRX,
    /// with the attention kernels [`Attention::from_environment`] selects.
    pub fn open(
        stream: &mut Stream,
        checkpoint: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<Session> {
        let (file, plan) = Self::validate(checkpoint, tokens, layers)?;
        let weights = Arc::new(Weights::upload(stream, &file, plan)?);
        Self::with_weights(
            stream,
            weights,
            tokens,
            layers,
            Attention::from_environment()?,
            None,
        )
    }

    /// Everything [`Session::open`] checks before it touches the device: the
    /// dimensions this build can serve, and the checkpoint's layout. It takes
    /// no stream, so an invalid argument is refused without a GPU ever being
    /// opened; `open` calls exactly this and then uploads what it returns.
    pub fn validate(
        checkpoint: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<(crate::checkpoint::Checkpoint, crate::checkpoint::Plan)> {
        Self::validate_dimensions(tokens, layers)?;
        Weights::plan(checkpoint)
    }

    /// Share resident weights and prepare artifacts for this sequence length.
    /// `compiler` selects a Loom shared library for all session kernels.
    pub fn with_weights(
        stream: &mut Stream,
        weights: Arc<Weights>,
        tokens: usize,
        layers: usize,
        attention: Attention,
        compiler: Option<&str>,
    ) -> Result<Session> {
        Self::validate_dimensions(tokens, layers)?;
        let shape = Shape::new(tokens, attention)?;
        let bundle = crate::kernels::prepare_for_target(compiler, &shape, stream.target())?;
        Self::build(stream, weights, &bundle, tokens, layers, compiler)
    }

    fn validate_dimensions(tokens: usize, layers: usize) -> Result<()> {
        if !shape::TOKENS.contains(&tokens) || !(1..=28).contains(&layers) {
            return Err(Error::invalid(format!(
                "tokens must be {}..{} and layers must be 1..28",
                shape::TOKENS.start(),
                shape::TOKENS.end()
            )));
        }
        Ok(())
    }

    fn build(
        stream: &mut Stream,
        weights: Arc<Weights>,
        bundle: &crate::kernels::PreparedBundle,
        tokens: usize,
        layers: usize,
        compiler: Option<&str>,
    ) -> Result<Session> {
        let shape = bundle.shape().clone();
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            let p = format!("blocks.{index}");
            let at = |name: &str, bytes: usize| weights.locate(&format!("{p}.{name}"), bytes);
            blocks.push(Block {
                qkvg_q: at("qkvg.q", QKVG * HIDDEN)?,
                qkvg_s: at("qkvg.s", QKVG * 4)?,
                wo_q: at("wo.q", HIDDEN * HIDDEN)?,
                wo_s: at("wo.s", HIDDEN * 4)?,
                gu_q: at("gu.q", 2 * INTER * HIDDEN)?,
                gu_s: at("gu.s", 2 * INTER * 4)?,
                down_q: at("down.q", HIDDEN * INTER)?,
                down_s: at("down.s", HIDDEN * 4)?,
                prenorm: at("prenorm", HIDDEN * 4)?,
                postnorm: at("postnorm", HIDDEN * 4)?,
                qnorm: at("qnorm", HEAD_DIM * 4)?,
                knorm: at("knorm", HEAD_DIM * 4)?,
            });
        }
        let kernels = Kernels::load(stream, bundle)?;
        let capacity = shape.capacity;
        let kv_bytes = KV_HEADS * HEAD_DIM * capacity * 2;
        let allocate = |bytes: usize| stream.allocate(bytes).map_err(Error::from);
        let buffers = Buffers {
            x: allocate(capacity * HIDDEN * 2)?,
            // the widest prepared operand: down's K = 16384 at its padded pitch
            a_q: allocate(capacity * shape::gemm_pitch(INTER))?,
            a_s: allocate(capacity * 4)?,
            fused: allocate(capacity * QKVG * 2)?,
            q: allocate(capacity * HIDDEN * 2)?,
            k: allocate(kv_bytes)?,
            v: allocate(kv_bytes)?,
            // Also holds silu(gate) * up, the wider of these two intermediates.
            attn_gu: allocate(capacity * INTER * 2)?,
            mods: allocate(layers * 6 * HIDDEN * 4)?,
            cos: allocate(capacity * HEAD_DIM * 4)?,
            sin: allocate(capacity * HEAD_DIM * 4)?,
        };
        // Headroom rows are read by the kernels and must be zero, not whatever
        // the allocator last held.
        for buffer in [&buffers.q, &buffers.k, &buffers.v, &buffers.fused, &buffers.x] {
            stream.fill(buffer.binding(), 0)?;
        }
        let v_transposed = match shape.attention {
            Attention::F16 => {
                Some(VTranspose::new(stream, tokens, capacity, KV_HEADS, compiler)?)
            }
            _ => None,
        };
        let sage = match shape.attention {
            Attention::F16 => None,
            smoothed => Some(Sage::new(
                stream,
                tokens,
                capacity,
                KV_HEADS * 4,
                KV_HEADS,
                smoothed.bits(),
                compiler,
            )?),
        };
        Ok(Session {
            stream_id: stream.id(),
            tokens,
            layers,
            shape,
            kernels,
            blocks,
            buffers,
            sage,
            v_transposed,
            graph: RefCell::new(None),
            rope: Cell::new(None),
            profile: AtomicBool::new(crate::kernels::native_profile()),
            stages: RefCell::new(Vec::new()),
            profiled: RefCell::new(None),
            device_clock_refused: RefCell::new(None),
            weights,
        })
    }

    /// The sequence length this session was built for.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The blocks this session runs.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Turns the per-stage timings on stderr on or off, reporting what they
    /// were. Timing synchronizes after every launch, so it is not free.
    pub fn set_profile(&self, enable: bool) -> bool {
        self.profile.swap(enable, Ordering::Relaxed)
    }

    fn check_stream(&self, stream: &Stream) -> Result<()> {
        if stream.id() != self.stream_id {
            return Err(Error::invalid("a session must use the stream that created it"));
        }
        Ok(())
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
        grid_x: usize,
        grid_y: usize,
        threads: u32,
        scalars: &Scalars,
        bindings: &[View<'g>],
    ) -> Result<()> {
        let constants = scalars.pack(stage, kernel)?;
        let grid = crate::kernels::grid(stage, grid_x, grid_y)?;
        let compiled = crate::kernels::workgroup(stage, kernel, threads)?;
        let stream = match sink {
            Sink::Record(recording) => {
                let after = recording.last.as_slice();
                // Safety: as the direct dispatch below. The bindings borrow this
                // session, which outlives the graph being recorded.
                let node = unsafe {
                    recording
                        .graph
                        .dispatch(after, kernel, grid, compiled, &constants, bindings)
                }?;
                recording.labels.push(stage);
                recording.last = After::One(node);
                return Ok(());
            }
            Sink::Stream(stream) => stream,
        };
        if !self.profile.load(Ordering::Relaxed) {
            // SAFETY: every binding is one of this session's buffers or resident
            // weights, allocated in `build` at the capacity the kernels were
            // compiled for, and the block matches the compiled workgroup.
            unsafe { stream.dispatch(kernel, grid, compiled, &constants, bindings) }?;
            return Ok(());
        }
        stream.synchronize()?;
        let began = std::time::Instant::now();
        // SAFETY: as above.
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
    fn timed(&self, stream: &mut Stream) -> Result<Option<std::time::Instant>> {
        if !self.profile.load(Ordering::Relaxed) {
            return Ok(None);
        }
        stream.synchronize()?;
        Ok(Some(std::time::Instant::now()))
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

    /// Appends `profile` as one JSON line to `KREA2_PROFILE_JSON`, when set:
    /// the raw device intervals, for comparing kernel changes stage by stage.
    fn export(&self, profile: &hrx::fabric::DeviceProfile) -> Result<()> {
        use std::io::Write;
        let Some(path) = std::env::var_os("KREA2_PROFILE_JSON") else {
            return Ok(());
        };
        let record = serde_json::json!({
            "tokens": self.tokens,
            "layers": self.layers,
            "attention": format!("{:?}", self.shape.attention),
            "profile": profile,
        });
        let failed = |e: &dyn std::fmt::Display| {
            Error::invalid(format!("KREA2_PROFILE_JSON {}: {e}", Path::new(&path).display()))
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| failed(&e))?;
        writeln!(file, "{record}").map_err(|e| failed(&e))
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
        let clock = match *self.device_clock_refused.borrow() {
            None => "GPU clock, kernels serialized",
            Some(_) => "host clock, synchronized per kernel",
        };
        eprintln!("stage profile over {blocks} block(s), {} tokens ({clock}):", self.tokens);
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
        self.check_stream(stream)?;
        let count = block_count.unwrap_or(self.layers.saturating_sub(first_block));
        if first_block >= self.layers || count < 1 || count > self.layers - first_block {
            return Err(Error::invalid("block range must be within the loaded layers"));
        }
        let tokens = self.tokens;
        if x.len() != tokens * HIDDEN {
            return Err(Error::invalid(format!(
                "x has {} elements, expected {}",
                x.len(),
                tokens * HIDDEN
            )));
        }
        if mods.len() != self.layers * 6 * HIDDEN {
            return Err(Error::invalid("mods has the wrong element count"));
        }
        if cos.len() != tokens * HEAD_DIM || sin.len() != tokens * HEAD_DIM {
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
    /// Use the stream that created this session. Both views must belong to its
    /// device and cover the shapes above. Inputs are copied into fixed session
    /// buffers before replay, so external addresses may change between calls.
    pub fn run_device(
        &self,
        stream: &mut Stream,
        x: View<'_>,
        mods: View<'_>,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<()> {
        self.check_stream(stream)?;
        let tokens = self.tokens;
        if cos.len() != tokens * HEAD_DIM || sin.len() != tokens * HEAD_DIM {
            return Err(Error::invalid("cos/sin have the wrong element count"));
        }
        // Into the session's own buffers, whose headroom rows are already zero.
        // `mods` is copied rather than bound directly because a recorded graph
        // fixes its addresses: the caller allocates modulation from a pool and
        // gets a different address each forward, while this one is the
        // session's for its whole life. One 4 MB device copy per forward.
        let rows = tokens * HIDDEN * 2;
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
    /// after. Profiling replays a second recording with device-clock markers
    /// around each kernel. Where the runtime cannot provide those, it
    /// dispatches directly and synchronizes around every kernel instead.
    fn blocks_through(&self, stream: &mut Stream) -> Result<()> {
        if self.profile.load(Ordering::Relaxed) {
            if self.device_clock_refused.borrow().is_none() {
                match self.profile_on_device(stream) {
                    Err(Error::Runtime(hrx::Error::Unsupported(reason))) => {
                        eprintln!(
                            "stage profile: no device clock ({reason}); \
                             timing each kernel between host synchronizations"
                        );
                        *self.device_clock_refused.borrow_mut() = Some(reason);
                    }
                    result => return result,
                }
            }
            for index in 0..self.layers {
                self.block(&mut Sink::Stream(stream), index, self.buffers.mods.binding())?;
            }
            return Ok(());
        }
        let mut recorded = self.graph.borrow_mut();
        let graph = match &mut *recorded {
            Some(graph) => graph,
            None => recorded.insert(self.record_graph(stream)?),
        };
        stream.launch(graph)?;
        Ok(())
    }

    /// Every block, recorded once into a graph over the session's own buffers.
    fn record_graph(&self, stream: &Stream) -> Result<hrx::GraphExec> {
        let recording = self.record_blocks(stream)?;
        // `finish` ends the graph's borrow of this session and the stream.
        Ok(recording.graph.finish()?)
    }

    fn record_blocks<'g>(&'g self, stream: &'g Stream) -> Result<Recording<'g>> {
        let mut recording = Recording::new(stream)?;
        for index in 0..self.layers {
            self.block(&mut Sink::Record(&mut recording), index, self.buffers.mods.binding())?;
        }
        Ok(recording)
    }

    /// One replay of the block loop with a device-clock interval around every
    /// kernel, added to the stage profile. The markers serialize the kernels,
    /// so the times are per-kernel costs, not the unprofiled forward's latency.
    fn profile_on_device(&self, stream: &mut Stream) -> Result<()> {
        let mut profiled = self.profiled.borrow_mut();
        let (graph, stages) = match &mut *profiled {
            Some(recorded) => recorded,
            None => {
                let recording = self.record_blocks(stream)?;
                let labels: Vec<String> = recording.labels.iter().map(|&s| s.into()).collect();
                let stages = recording.labels.clone();
                profiled.insert((recording.graph.finish_profiled(&labels)?, stages))
            }
        };
        let profile = stream.launch_profiled(graph)?;
        self.export(&profile)?;
        let micros_per_tick = 1e6 / profile.frequency_hz as f64;
        for (interval, &stage) in profile.intervals.iter().zip(stages.iter()) {
            let ticks = interval.end_tick - interval.start_tick;
            self.accumulate(stage, ticks as f64 * micros_per_tick);
        }
        self.accumulate("(idle between kernels)", profile.gaps_ms * 1e3);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm<'g>(
        &'g self,
        sink: &mut Sink<'g, '_>,
        kernel: &'g hrx::Kernel,
        stage: &'static str,
        weights: View<'g>,
        scales: View<'g>,
        n: usize,
        out: View<'g>,
        gate: Option<View<'g>>,
        tile: shape::GemmTile,
    ) -> Result<()> {
        let scalars = Scalars::new().index(self.tokens);
        let (a_q, a_s) = (self.buffers.a_q.binding(), self.buffers.a_s.binding());
        let all = [a_q, weights, scales, a_s, out, gate.unwrap_or(out)];
        let bindings = &all[..5 + usize::from(gate.is_some())];
        let grid_y = shape::gemm_grid_rows(self.tokens);
        let columns = n / tile.columns();
        self.launch(sink, kernel, stage, columns, grid_y, tile.threads(), &scalars, bindings)
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
            modulation(mods, index, 0)?,
            modulation(mods, index, 1)?,
            b.a_q.binding(),
            b.a_s.binding(),
        ];
        self.launch(
            sink,
            &self.kernels.prepare_norm,
            "prepare",
            tokens,
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
            shape::gemm_tile("gemm_qkvg"),
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
            tokens,
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
                    Sink::Stream(stream) => self.timed(stream)?,
                    Sink::Record(_) => None,
                };
                sage.run(sink, b.q.binding(), b.k.binding(), b.v.binding())?;
                if let Sink::Stream(stream) = &mut *sink {
                    self.record(stream, "SA2 preprocessing", preparing)?;
                }
                let attention_scalars = Scalars::new().index(tokens).index(KV_HEADS);
                let attention = [
                    sage.q4.binding(),
                    sage.k4.binding(),
                    sage.v_transposed.binding(),
                    sage.q_scale.binding(),
                    sage.k_scale.binding(),
                    sage.correction.binding(),
                    b.attn_gu.binding(),
                ];
                let rows = 16 * (self.shape.attention_waves as usize / 4);
                self.launch(
                    sink,
                    &self.kernels.attention,
                    "SA2 attention",
                    tokens.div_ceil(rows),
                    KV_HEADS,
                    32 * self.shape.attention_waves,
                    &attention_scalars,
                    &attention,
                )?;
            }
            // fp16 QK and PV from the RoPE outputs and V transposed, 16 query
            // rows per workgroup.
            None => {
                let transposed = self
                    .v_transposed
                    .as_ref()
                    .ok_or_else(|| Error::internal("an fp16 session has no V transpose"))?;
                let (kernel, (x, y)) = transposed.launch();
                let scalars = Scalars::new().index(tokens);
                let operands = [b.v.binding(), transposed.output.binding()];
                self.launch(sink, kernel, "V transpose", x, y, 256, &scalars, &operands)?;
                let attention_scalars = Scalars::new().index(tokens);
                let attention = [
                    b.q.binding(),
                    b.k.binding(),
                    transposed.output.binding(),
                    b.attn_gu.binding(),
                ];
                self.launch(
                    sink,
                    &self.kernels.attention,
                    "f16 attention",
                    tokens.div_ceil(16),
                    KV_HEADS,
                    128,
                    &attention_scalars,
                    &attention,
                )?;
            }
        }

        let gated_scalars = Scalars::new().index(tokens);
        let gated =
            [b.attn_gu.binding(), gate_half(&b.fused)?, b.a_q.binding(), b.a_s.binding()];
        self.launch(
            sink,
            &self.kernels.prepare_gated,
            "prepare gated",
            tokens,
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
            Some(modulation(mods, index, 2)?),
            shape::gemm_tile("gemm_wo"),
        )?;

        let post_scalars = Scalars::new().index(tokens);
        let post = [
            b.x.binding(),
            w.view(block.postnorm),
            modulation(mods, index, 3)?,
            modulation(mods, index, 4)?,
            b.a_q.binding(),
            b.a_s.binding(),
        ];
        self.launch(
            sink,
            &self.kernels.prepare_norm,
            "prepare",
            tokens,
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
            b.attn_gu.binding(),
            None,
            shape::gemm_tile("gemm_gu"),
        )?;

        let swiglu_scalars = Scalars::new().index(tokens);
        let swiglu = [b.attn_gu.binding(), b.a_q.binding(), b.a_s.binding()];
        self.launch(
            sink,
            &self.kernels.prepare_swiglu,
            "prepare swiglu",
            tokens,
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
            Some(modulation(mods, index, 5)?),
            shape::gemm_tile("gemm_down"),
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

        let mods = vec![0.01f32; 6 * HIDDEN];
        let a = (vec![1.0f32; tokens * 128], vec![0.0f32; tokens * 128]);
        let b = (vec![0.0f32; tokens * 128], vec![1.0f32; tokens * 128]);
        let mut x = vec![0x3f00u16; tokens * HIDDEN];

        session.run(&mut stream, &mut x, &mods, &a.0, &a.1, 0, Some(1)).expect("A");
        let after_a = resident();
        assert!(after_a.is_some(), "A left nothing recorded");

        session.run(&mut stream, &mut x, &mods, &b.0, &b.1, 0, Some(1)).expect("B");
        assert_ne!(resident(), after_a, "B's upload was not recorded, so A looks resident");

        session.run(&mut stream, &mut x, &mods, &a.0, &a.1, 0, Some(1)).expect("A again");
        assert_eq!(resident(), after_a, "A's second upload was not recorded");
    }

    fn inputs(
        tokens: usize,
        layers: usize,
        phase: usize,
    ) -> (Vec<u16>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = (0..tokens * HIDDEN)
            .map(|i| ((0.25 + ((i + phase * 17) % 97) as f32 / 128.0).to_bits() >> 16) as u16)
            .collect();
        let mods =
            (0..layers * 6 * HIDDEN).map(|i| 0.01 + ((i + phase) % 7) as f32 * 0.001).collect();
        let angle = phase as f32 * 0.125;
        let cos = vec![angle.cos(); tokens * HEAD_DIM];
        let sin = vec![angle.sin(); cos.len()];
        (x, mods, cos, sin)
    }

    #[test]
    #[ignore = "requires cached Krea checkpoint and gfx1151"]
    fn shared_context_slots_preserve_native_results_and_retained_outputs() {
        let (checkpoint, tokens) = fixture();
        let context = hrx::inference::ModelContext::new(Default::default()).unwrap();
        let mut stream = Stream::open().unwrap();
        let session = Session::open(&mut stream, &checkpoint, tokens, 1).unwrap();
        let (start, mods, cos, sin) = inputs(tokens, 1, 0);
        let mut expected = start.clone();
        session.run(&mut stream, &mut expected, &mods, &cos, &sin, 0, None).unwrap();
        let model = session.into_prepared(&context, stream, cos, sin).unwrap();
        let result = model
            .submit_host(&[bytemuck::cast_slice(&start), bytemuck::cast_slice(&mods)])
            .unwrap();
        let output = result.outputs()[0].clone();
        let actual = result.download().unwrap().wait().unwrap().remove(0);
        assert_eq!(actual, bytemuck::cast_slice::<_, u8>(&expected));
        assert!(matches!(model.try_acquire(), Err(hrx::Error::Busy(_))));
        drop(output);
        drop(model.try_acquire().unwrap());
    }

    #[test]
    #[ignore = "requires Krea checkpoint and gfx1151"]
    fn a_recorded_block_loop_replays_to_the_same_bytes() {
        let (checkpoint, tokens) = fixture();
        let device = hrx::Device::open(0).expect("device");
        let mut stream = device.stream().expect("stream");
        let mut other = device.stream().expect("sibling stream");
        let session = Session::open(&mut stream, &checkpoint, tokens, 2).expect("session");
        // Keep previous external buffers alive so the allocator cannot disguise
        // a stale captured address by handing out the same allocation again.
        let mut external = Vec::new();
        for phase in 0..2 {
            let (start, mods, cos, sin) = inputs(tokens, 2, phase);
            let mut expected = start.clone();
            session.run(&mut stream, &mut expected, &mods, &cos, &sin, 0, None).expect("eager");
            assert!(expected.iter().all(|v| v & 0x7f80 != 0x7f80), "finite reference");
            let x = stream.allocate(start.len() * 2).unwrap();
            let table = stream.allocate(mods.len() * 4).unwrap();
            stream.upload(x.binding(), bytemuck::cast_slice(&start)).unwrap();
            stream.upload(table.binding(), bytemuck::cast_slice(&mods)).unwrap();
            stream.synchronize().unwrap();

            // A sibling stream is rejected before it can overwrite session inputs.
            let error = session
                .run_device(&mut other, x.binding(), table.binding(), &cos, &sin)
                .expect_err("stream affinity");
            assert!(error.is_invalid_argument());
            other.synchronize().unwrap();
            let mut actual = vec![0u16; start.len()];
            stream
                .read_blocking(
                    session.buffers.x.binding(),
                    bytemuck::cast_slice_mut(&mut actual),
                )
                .unwrap();
            assert_eq!(actual, expected, "a rejected call changed session storage");
            assert!(session.run(&mut other, &mut actual, &mods, &cos, &sin, 0, None).is_err());

            session
                .run_device(&mut stream, x.binding(), table.binding(), &cos, &sin)
                .expect("replay");
            assert!(session.graph.borrow().is_some());
            stream.read_blocking(x.binding(), bytemuck::cast_slice_mut(&mut actual)).unwrap();
            assert_eq!(actual, expected, "replay with changed inputs, tables and addresses");

            // Diagnostic execution must remain correct after a graph has been cached.
            session.set_profile(true);
            stream.upload(x.binding(), bytemuck::cast_slice(&start)).unwrap();
            session
                .run_device(&mut stream, x.binding(), table.binding(), &cos, &sin)
                .expect("profiled eager");
            stream.read_blocking(x.binding(), bytemuck::cast_slice_mut(&mut actual)).unwrap();
            assert_eq!(actual, expected);
            session.set_profile(false);
            external.push((x, table));
        }
        let (x, table) = external.last().unwrap();
        let (_, _, cos, sin) = inputs(tokens, 2, 1);
        session.run_device(&mut stream, x.binding(), table.binding(), &cos, &sin).unwrap();
        // Exercise destruction with the final replay still in flight.
    }

    /// Compare two independent two-block prefixes with a serial recording.
    /// Inputs are initialized, reset before every sample, and outputs checked.
    #[test]
    #[ignore = "requires Krea checkpoint and gfx1151; a benchmark, not a speed assertion"]
    fn two_independent_block_prefixes_are_priced_against_a_chain() {
        for tokens in [275usize, 1043, 4115] {
            overlap_probe(tokens);
        }
    }

    fn overlap_probe(tokens: usize) {
        let (checkpoint, _) = fixture();
        let mut stream = Stream::open().unwrap();
        let (file, plan) = Session::validate(&checkpoint, tokens, 2).unwrap();
        let weights = Arc::new(Weights::upload(&mut stream, &file, plan).unwrap());
        let first =
            Session::with_weights(&mut stream, weights.clone(), tokens, 2, attention(), None)
                .unwrap();
        let second =
            Session::with_weights(&mut stream, weights, tokens, 2, attention(), None).unwrap();
        let mut starts = Vec::new();
        let mut expected = Vec::new();
        for (phase, session) in [&first, &second].into_iter().enumerate() {
            let (start, mods, cos, sin) = inputs(tokens, 2, phase);
            let initial = stream.allocate(start.len() * 2).unwrap();
            stream.upload(initial.binding(), bytemuck::cast_slice(&start)).unwrap();
            let mut result = start;
            session.run(&mut stream, &mut result, &mods, &cos, &sin, 0, None).unwrap();
            assert!(result.iter().all(|v| v & 0x7f80 != 0x7f80));
            starts.push(initial);
            expected.push(result);
        }
        let record = |independent| {
            let mut recording = Recording::new(&stream).unwrap();
            for (index, session) in [&first, &second].into_iter().enumerate() {
                if independent && index == 1 {
                    recording.last = After::None;
                }
                for block in 0..session.layers {
                    session
                        .block(
                            &mut Sink::Record(&mut recording),
                            block,
                            session.buffers.mods.binding(),
                        )
                        .unwrap();
                }
            }
            recording.graph.finish().unwrap()
        };
        let mut graphs = [record(false), record(true)];
        let mut samples = [Vec::new(), Vec::new()];
        for round in 0..12 {
            for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                for (session, initial) in [&first, &second].into_iter().zip(&starts) {
                    stream
                        .copy(
                            session.buffers.x.binding().slice(0, initial.bytes()).unwrap(),
                            initial.binding(),
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                let began = std::time::Instant::now();
                stream.launch(&mut graphs[arm]).unwrap();
                stream.synchronize().unwrap();
                if round >= 3 {
                    samples[arm].push(began.elapsed().as_secs_f64() * 1e3);
                }
                for (session, expected) in [&first, &second].into_iter().zip(&expected) {
                    let mut actual = vec![0u16; expected.len()];
                    stream
                        .read_blocking(
                            session.buffers.x.binding(),
                            bytemuck::cast_slice_mut(&mut actual),
                        )
                        .unwrap();
                    assert_eq!(&actual, expected, "two-block prefix parity");
                }
            }
        }
        for arm in &mut samples {
            arm.sort_by(f64::total_cmp);
        }
        eprintln!(
            "{tokens} tokens, two sessions with two blocks each: serial {:.3} ms, independent {:.3} ms",
            samples[0][4], samples[1][4]
        );
    }

    /// The attention width under test: `KREA2_ATTN_QK`, so one run of these
    /// tests covers each kernel family in turn.
    fn attention() -> Attention {
        Attention::from_environment().expect("KREA2_ATTN_QK must be 4, 8 or 16")
    }

    fn fixture() -> (std::path::PathBuf, usize) {
        use std::path::PathBuf;
        let checkpoint =
            std::env::var_os("KREA2_MODEL").map(PathBuf::from).unwrap_or_else(|| {
                crate::models::hub::file(
                    "diffusion_models/krea2_turbo_int8_convrot.safetensors",
                    true,
                )
                .expect("cache the Turbo checkpoint or set KREA2_MODEL")
            });
        assert!(checkpoint.is_file(), "set KREA2_MODEL to a local checkpoint");
        let tokens = 4115;
        (checkpoint, tokens)
    }
}
