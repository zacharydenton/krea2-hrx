//! The 28 Krea 2 transformer blocks as a resident Loom session.
//!
//! Ten launches per block: prepare (RMSNorm, modulation, group-256 Hadamard
//! rotation, per-token int8 quantization) -> fused qkv|gate GEMM -> QK norm and
//! RoPE -> attention -> gated prepare -> wo GEMM with the gated residual ->
//! prepare -> fused gate|up GEMM with the SwiGLU product in its epilogue ->
//! prepare -> down GEMM with the gated residual. The residual stream stays on
//! the device between blocks, in bf16, as ComfyUI keeps it.
#![deny(unsafe_code)]

pub mod bundle;
pub mod sage;
pub mod weights;

use std::path::Path;
use std::sync::Mutex;

use hrx::{device, Args, Buffer, DevicePtr};

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

/// A failure, and whether the caller could have prevented it. The C ABI maps
/// these onto `KREA2_INVALID_ARGUMENT` and `KREA2_ERROR`.
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
        Error::failed(error.0)
    }
}

impl From<loom::Error> for Error {
    fn from(error: loom::Error) -> Self {
        Error::failed(error.0)
    }
}

impl From<krea2_checkpoint::Error> for Error {
    fn from(error: krea2_checkpoint::Error) -> Self {
        // The checkpoint reader's rejections are all about the file the caller
        // named, so they read as invalid arguments through the C ABI.
        Error::failed(error.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One block's weights, as device addresses into the resident checkpoint.
struct Block {
    qkvg_q: DevicePtr,
    qkvg_s: DevicePtr,
    wo_q: DevicePtr,
    wo_s: DevicePtr,
    gu_q: DevicePtr,
    gu_s: DevicePtr,
    down_q: DevicePtr,
    down_s: DevicePtr,
    prenorm: DevicePtr,
    postnorm: DevicePtr,
    qnorm: DevicePtr,
    knorm: DevicePtr,
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

pub struct Session {
    tokens: usize,
    layers: usize,
    metadata: Metadata,
    kernels: Kernels,
    blocks: Vec<Block>,
    buffers: Buffers,
    // Calls on one session are serialized: they share the scratch buffers.
    /// The smoothed attention kernels' preparation pass, when the bundle was
    /// built with one (`KREA2_ATTN_QK` of 4 or 8).
    sage: Option<Sage>,
    lock: Mutex<()>,
    _weights: std::sync::Arc<Weights>,
}

impl Session {
    /// Loads a checkpoint and a kernel bundle compiled for exactly `tokens`.
    ///
    /// The bundle is validated first: a checkpoint is thirteen gigabytes, and
    /// nothing should read it to then reject the metadata beside it.
    pub fn open(
        checkpoint: &Path,
        kernels_dir: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<Session> {
        let metadata = Session::metadata(kernels_dir, tokens, layers)?;
        let weights = std::sync::Arc::new(Weights::load(checkpoint)?);
        Session::build(weights, kernels_dir, metadata, tokens, layers)
    }

    /// The same, sharing a checkpoint already on the device.
    pub fn with_weights(
        weights: std::sync::Arc<Weights>,
        kernels_dir: &Path,
        tokens: usize,
        layers: usize,
    ) -> Result<Session> {
        let metadata = Session::metadata(kernels_dir, tokens, layers)?;
        Session::build(weights, kernels_dir, metadata, tokens, layers)
    }

    /// The bundle's `launch.txt`, checked against the shape rules for `tokens`.
    /// Nothing here touches the device.
    fn metadata(kernels_dir: &Path, tokens: usize, layers: usize) -> Result<Metadata> {
        if !(16..=16896).contains(&tokens) {
            return Err(Error::invalid("tokens must be 16..16896"));
        }
        if !(1..=28).contains(&layers) {
            return Err(Error::invalid("layers must be 1..28"));
        }
        let path = kernels_dir.join("launch.txt");
        let text = std::fs::read_to_string(&path).map_err(|_| {
            Error::invalid(
                "invalid kernel launch metadata; rebuild with scripts/build_kernels.py",
            )
        })?;
        Metadata::parse(&text, tokens)
    }

    fn build(
        weights: std::sync::Arc<Weights>,
        kernels_dir: &Path,
        metadata: Metadata,
        tokens: usize,
        layers: usize,
    ) -> Result<Session> {
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
            let at = |name: &str, bytes: usize| weights.at(&format!("{p}.{name}"), bytes);
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
        let kernels = Kernels::load(kernels_dir, &metadata)?;
        let capacity = metadata.capacity;
        let kv_bytes = KV_HEADS as usize * HEAD_DIM as usize * capacity * 2;
        let allocate = |bytes: usize| device().allocate(bytes).map_err(Error::from);
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
        for (buffer, bytes) in [
            (&buffers.q, capacity * HIDDEN as usize * 2),
            (&buffers.k, kv_bytes),
            (&buffers.v, kv_bytes),
            (&buffers.fused, capacity * QKVG as usize * 2),
            (&buffers.x, capacity * HIDDEN as usize * 2),
        ] {
            device().zero(buffer.ptr(), bytes)?;
        }
        if let Some(transposed) = &buffers.v_transposed {
            device().zero(transposed.ptr(), kv_bytes)?;
        }
        let sage = match metadata.attention_bits {
            16 => None,
            bits => Some(Sage::new(
                tokens,
                capacity,
                (KV_HEADS * 4) as usize,
                KV_HEADS as usize,
                bits,
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
            lock: Mutex::new(()),
            _weights: weights,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Runs a contiguous range of blocks over the caller's residual stream.
    ///
    /// `x` is bf16 `[tokens][6144]`, in and out. `mods` is f32
    /// `[layers][6][6144]`, already including each block's table; `cos` and
    /// `sin` are f32 `[tokens][128]`.
    pub fn run(
        &self,
        x: &mut [u16],
        mods: &[f32],
        cos: &[f32],
        sin: &[f32],
        first_block: usize,
        block_count: Option<usize>,
    ) -> Result<()> {
        let _serialized = self.lock.lock().unwrap_or_else(|e| e.into_inner());
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
        device().write(self.buffers.x.ptr(), x)?;
        device().write(self.buffers.mods.ptr(), mods)?;
        device().write(self.buffers.cos.ptr(), cos)?;
        device().write(self.buffers.sin.ptr(), sin)?;
        for index in first_block..first_block + count {
            self.block(index, self.buffers.mods.ptr())?;
        }
        device().synchronize()?;
        device().read(x, self.buffers.x.ptr())?;
        Ok(())
    }

    /// The same over a residual stream and modulation tables already on the
    /// device, which is what the pipeline has: 50 MB of x and 4 MB of mods per
    /// step never leave the GPU.
    ///
    /// `x` is bf16 `[tokens][6144]`, read and written in place; `mods` is f32
    /// `[layers][6][6144]`. The rope tables stay host-side because they change
    /// only when the image geometry does.
    pub fn run_device(
        &self,
        x: DevicePtr,
        mods: DevicePtr,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<()> {
        let _serialized = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let tokens = self.tokens;
        if cos.len() != tokens * HEAD_DIM as usize || sin.len() != tokens * HEAD_DIM as usize {
            return Err(Error::invalid("cos/sin have the wrong element count"));
        }
        // Into the session's own buffer, whose headroom rows are already zero.
        device().copy_device_to_device(
            self.buffers.x.ptr(),
            x,
            tokens * HIDDEN as usize * 2,
        )?;
        device().write(self.buffers.cos.ptr(), cos)?;
        device().write(self.buffers.sin.ptr(), sin)?;
        for index in 0..self.layers {
            self.block(index, mods)?;
        }
        device().synchronize()?;
        device().copy_device_to_device(
            x,
            self.buffers.x.ptr(),
            tokens * HIDDEN as usize * 2,
        )?;
        Ok(())
    }

    /// This block's slice of the modulation tables: prescale, preshift,
    /// pregate, postscale, postshift, postgate, each 6144 floats.
    fn modulation(&self, mods: DevicePtr, index: usize, part: usize) -> DevicePtr {
        let stride = 6 * HIDDEN as usize * 4;
        mods.offset(index * stride + part * HIDDEN as usize * 4)
    }

    fn gemm(
        &self,
        kernel: &hrx::Kernel,
        weights: DevicePtr,
        scales: DevicePtr,
        n: i32,
        out: DevicePtr,
        gate: Option<DevicePtr>,
    ) -> Result<()> {
        let mut args = Args::new();
        args.i32(self.tokens as i32)
            .ptr(self.buffers.a_q.ptr())
            .ptr(weights)
            .ptr(scales)
            .ptr(self.buffers.a_s.ptr())
            .ptr(out);
        if let Some(gate) = gate {
            args.ptr(gate);
        }
        let grid_y = loom::shape::gemm_grid_rows(
            self.tokens as i32,
            self.metadata.gemm_rows as i32,
            self.metadata.m_group as i32,
        );
        kernel.launch_2d((n / 128) as u32, grid_y as u32, THREADS, &args)?;
        Ok(())
    }

    fn block(&self, index: usize, mods: DevicePtr) -> Result<()> {
        let block = &self.blocks[index];
        let tokens = self.tokens as i32;
        let b = &self.buffers;

        let mut norm = Args::new();
        norm.i32(tokens)
            .ptr(b.x.ptr())
            .ptr(block.prenorm)
            .ptr(self.modulation(mods, index, 0))
            .ptr(self.modulation(mods, index, 1))
            .ptr(b.a_q.ptr())
            .ptr(b.a_s.ptr());
        self.kernels.prepare_norm.launch_2d(tokens as u32, 1, THREADS, &norm)?;

        self.gemm(
            &self.kernels.gemm_qkvg,
            block.qkvg_q,
            block.qkvg_s,
            QKVG,
            b.fused.ptr(),
            None,
        )?;

        let mut rope = Args::new();
        rope.i32(tokens)
            .ptr(b.fused.ptr())
            .ptr(block.qnorm)
            .ptr(block.knorm)
            .ptr(b.cos.ptr())
            .ptr(b.sin.ptr())
            .ptr(b.q.ptr())
            .ptr(b.k.ptr())
            .ptr(b.v.ptr());
        self.kernels.rope.launch_2d(tokens as u32, 1, THREADS, &rope)?;

        match &self.sage {
            // Smoothed int4/int8 QK: the preparation pass quantizes Q and K
            // against their means and works out the correction the kernel adds
            // back to the scores.
            Some(sage) => {
                sage.run(b.q.ptr(), b.k.ptr(), b.v.ptr())?;
                let mut attention = Args::new();
                attention
                    .i32(tokens)
                    .i32(KV_HEADS)
                    .ptr(sage.q4.ptr())
                    .ptr(sage.k4.ptr())
                    .ptr(sage.v_transposed.ptr())
                    .ptr(sage.q_scale.ptr())
                    .ptr(sage.k_scale.ptr())
                    .ptr(sage.correction.ptr())
                    .ptr(b.attn.ptr());
                let waves = self.metadata.attention_waves as i32;
                let rows = 16 * (waves / 4);
                self.kernels.attention.launch_2d(
                    ((tokens + rows - 1) / rows) as u32,
                    KV_HEADS as u32,
                    32 * waves as u32,
                    &attention,
                )?;
            }
            // fp16 QK and PV straight from the RoPE outputs.
            None => {
                let values = match (&b.v_transposed, &self.kernels.attention_transpose) {
                    (Some(transposed), Some(kernel)) => {
                        let mut args = Args::new();
                        args.i32(tokens).ptr(b.v.ptr()).ptr(transposed.ptr());
                        kernel.launch_2d(
                            tokens.div_euclid(32) as u32 + u32::from(tokens % 32 != 0),
                            (KV_HEADS * HEAD_DIM / 32) as u32,
                            256,
                            &args,
                        )?;
                        transposed.ptr()
                    }
                    _ => b.v.ptr(),
                };
                let mut attention = Args::new();
                attention
                    .i32(tokens)
                    .ptr(b.q.ptr())
                    .ptr(b.k.ptr())
                    .ptr(values)
                    .ptr(b.attn.ptr());
                let rows = 16 * self.metadata.fp16_query_tiles as i32;
                self.kernels.attention.launch_2d(
                    ((tokens + rows - 1) / rows) as u32,
                    KV_HEADS as u32,
                    128 * self.metadata.fp16_query_tiles,
                    &attention,
                )?;
            }
        }

        let mut gated = Args::new();
        gated
            .i32(tokens)
            .ptr(b.attn.ptr())
            .ptr(b.fused.ptr().offset(GATE_OFFSET as usize * 2))
            .ptr(b.a_q.ptr())
            .ptr(b.a_s.ptr());
        self.kernels.prepare_gated.launch_2d(tokens as u32, 1, THREADS, &gated)?;

        self.gemm(
            &self.kernels.gemm_wo,
            block.wo_q,
            block.wo_s,
            HIDDEN,
            b.x.ptr(),
            Some(self.modulation(mods, index, 2)),
        )?;

        let mut post = Args::new();
        post.i32(tokens)
            .ptr(b.x.ptr())
            .ptr(block.postnorm)
            .ptr(self.modulation(mods, index, 3))
            .ptr(self.modulation(mods, index, 4))
            .ptr(b.a_q.ptr())
            .ptr(b.a_s.ptr());
        self.kernels.prepare_norm.launch_2d(tokens as u32, 1, THREADS, &post)?;

        self.gemm(&self.kernels.gemm_gu, block.gu_q, block.gu_s, 2 * INTER, b.gu.ptr(), None)?;

        let mut swiglu = Args::new();
        swiglu.i32(tokens).ptr(b.gu.ptr()).ptr(b.a_q.ptr()).ptr(b.a_s.ptr());
        self.kernels.prepare_swiglu.launch_2d(tokens as u32, 1, THREADS, &swiglu)?;

        self.gemm(
            &self.kernels.gemm_down,
            block.down_q,
            block.down_s,
            HIDDEN,
            b.x.ptr(),
            Some(self.modulation(mods, index, 5)),
        )?;
        Ok(())
    }
}
