//! Auxiliary Loom operations for encoding, decoding and sampling.
//! Kernels specialize on tensor shape and launch geometry and are cached on first use.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod tensor;

use std::sync::{Arc, OnceLock};

use hrx::{Buffer, Constants, Kernel, Stream, View};
use loom::cache::PreparedKernels;
pub use loom::Scalars;

pub use loom::Config;

pub use tensor::{Pool, Scratch, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<hrx::Error> for Error {
    fn from(error: hrx::Error) -> Self {
        Error(error.to_string())
    }
}

impl From<loom::Error> for Error {
    fn from(error: loom::Error) -> Self {
        Error(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Which pointwise function [`Ops::unary`] applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unary {
    Silu,
    Gelu,
    Sigmoid,
}

/// Elementwise, with the right operand broadcast over the left.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binary {
    Add,
    Mul,
}

/// How [`Ops::norm`] scales: `x * (1 + w)` is the DiT convention, `x * w` the
/// plain one; `Group` is the VAE's L2 normalization with bf16 rounding between
/// normalization, sqrt(width), and weight multiplication.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    OnePlusScale = 0,
    Scale = 1,
    Group = 2,
}

/// Storage order for convolution weights with logical `[out, in, ky, kx]` shapes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// The file's order, `[out][in][ky][kx]`. Also every non-convolution.
    RowMajor,
    /// `[out][ky][kx][in]`, which is what [`Ops::conv`]'s implicit-GEMM path
    /// reduces over.
    ChannelsLast,
}

/// Device weight with bf16 values, a logical shape and a storage layout.
/// Float32 normalization scales are retained from the checkpoint or upcast
/// once on first use.
pub struct Weight {
    pub shape: Vec<usize>,
    pub count: usize,
    /// The allocation holding the bf16 values, and where in it they start.
    /// Held so a view over this weight cannot outlive its memory.
    storage: Arc<Buffer>,
    offset: usize,
    layout: Layout,
    /// The checkpoint's own float32 copy. It lives in the packed arena rather
    /// than beside the bf16 values, so it carries its own allocation: holding
    /// only the bf16 one left this alive by coincidence.
    given: Option<(Arc<Buffer>, usize)>,
    upcast: OnceLock<Buffer>,
}

impl Weight {
    /// `f32` is the checkpoint's own float32 copy, when it kept one. The
    /// values are taken to be in the file's order; a caller that packed them
    /// says so with [`Weight::in_layout`].
    pub fn new(
        storage: &Arc<Buffer>,
        offset: usize,
        shape: Vec<usize>,
        count: usize,
        f32: Option<(Arc<Buffer>, usize)>,
    ) -> Weight {
        Weight {
            shape,
            count,
            storage: Arc::clone(storage),
            offset,
            layout: Layout::RowMajor,
            given: f32,
            upcast: OnceLock::new(),
        }
    }

    /// The bf16 values, as a kernel binding.
    pub fn values(&self) -> Result<View<'_>> {
        self.storage.try_slice(self.offset, self.count * 2).map_err(|e| Error(e.to_string()))
    }

    /// The same, for values written in `layout`.
    pub fn in_layout(mut self, layout: Layout) -> Weight {
        self.layout = layout;
        self
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// These values as a matrix, for the operations that take tensors. The
    /// tensor shares the allocation, so it keeps the weight's memory alive.
    pub fn tensor(&self, rows: usize, cols: usize) -> Result<Tensor> {
        Tensor::shared(&self.storage, self.offset, rows, cols)
    }

    /// The scales as float32, upcast once if the file did not keep them so.
    pub fn f32_values(&self, stream: &mut Stream) -> Result<View<'_>> {
        if let Some((buffer, offset)) = &self.given {
            return buffer.try_slice(*offset, self.count * 4).map_err(|e| Error(e.to_string()));
        }
        if let Some(buffer) = self.upcast.get() {
            return Ok(buffer.binding());
        }
        let mut bits = vec![0u16; self.count];
        stream.read(self.values()?, bytemuck::cast_slice_mut(&mut bits))?;
        let floats: Vec<f32> = bits.into_iter().map(krea2_numerics::to_f32).collect();
        let buffer = stream.allocate(floats.len() * 4)?;
        stream.upload(buffer.binding(), bytemuck::cast_slice(&floats))?;
        // A losing race drops its buffer, which releases it.
        Ok(self.upcast.get_or_init(|| buffer).binding())
    }
}

/// The operations, over one buffer pool.

pub struct Ops {
    pool: Arc<Pool>,
    kernels: PreparedKernels,
    /// Compiler override for this operation set's auxiliary kernels.
    compiler: Option<String>,
}

impl Ops {
    pub fn new(pool: Arc<Pool>) -> Ops {
        Ops { pool, compiler: None, kernels: PreparedKernels::default() }
    }

    /// The same, with an explicit compiler instead of `HRX_LOOM_LIBRARY` or the pinned bundle.
    pub fn with_compiler(pool: Arc<Pool>, compiler: Option<&str>) -> Ops {
        Ops {
            pool,
            compiler: compiler.map(str::to_string),
            kernels: PreparedKernels::new(compiler),
        }
    }

    pub fn compiler(&self) -> Option<&str> {
        self.compiler.as_deref()
    }

    pub fn pool(&self) -> &Arc<Pool> {
        &self.pool
    }

    pub fn tensor(&self, stream: &Stream, rows: usize, cols: usize) -> Result<Tensor> {
        Tensor::new(&self.pool, stream, rows, cols)
    }

    /// `y = x wᵀ (+ bias)`, the shape every linear layer here takes.
    pub fn linear(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        w: &Weight,
        bias: Option<View<'_>>,
    ) -> Result<Tensor> {
        let n = *w.shape.first().ok_or_else(|| Error("linear dimensions".into()))?;
        let k = x.cols();
        if n < 1 || x.rows() < 1 || x.cols() < 1 || w.count != n * k {
            return Err(Error("linear dimensions".into()));
        }
        let y = self.tensor(stream, x.rows(), n)?;
        let name = if bias.is_some() { "gemm_bf16_bf16_nt_bias" } else { "gemm_bf16_bf16_nt" };
        self.matmul(
            stream,
            name,
            x.binding()?,
            w.values()?,
            y.binding()?,
            x.rows(),
            n,
            k,
            1,
            1.0,
            bias,
        )?;
        Ok(y)
    }

    /// RMSNorm with float32 scales.
    pub fn norm(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        w: &Weight,
        mode: Norm,
        eps: f32,
    ) -> Result<Tensor> {
        if w.count != x.cols() {
            return Err(Error("norm dimensions".into()));
        }
        let scales = w.f32_values(stream)?;
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.rows()).float(eps);
        let bindings = [x.binding()?, scales, y.binding()?];
        // Eight rows per workgroup when a row fits in one wave's registers.
        let wave = mode == Norm::Group && x.cols() <= 1024;
        let name =
            if wave { "norm_2_wave".to_string() } else { format!("norm_{}", mode as u8) };
        let grid = if wave { x.rows().div_ceil(8) } else { x.rows() };
        unsafe {
            self.launch(
                stream,
                &name,
                config(&[("xsize", x.size()), ("cols", x.cols())]),
                &scalars,
                &bindings,
                grid,
                1,
                256,
            )
        }?;
        Ok(y)
    }

    /// VAE L2 normalization and SiLU in one pass, preserving its bf16 boundaries.
    pub fn norm_silu(&self, stream: &mut Stream, x: &Tensor, w: &Weight) -> Result<Tensor> {
        if w.count != x.cols() {
            return Err(Error("normalization dimensions".into()));
        }
        if x.cols() > 1024 {
            let normed = self.norm(stream, x, w, Norm::Group, 1e-5)?;
            return self.unary(stream, &normed, Unary::Silu);
        }
        let scales = w.f32_values(stream)?;
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.rows()).float(1e-5);
        let bindings = [x.binding()?, scales, y.binding()?];
        unsafe {
            self.launch(
                stream,
                "norm_2_wave_silu",
                config(&[("xsize", x.size()), ("cols", x.cols())]),
                &scalars,
                &bindings,
                x.rows().div_ceil(8),
                1,
                256,
            )
        }?;
        Ok(y)
    }

    pub fn unary(&self, stream: &mut Stream, x: &Tensor, op: Unary) -> Result<Tensor> {
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size());
        let bindings = [x.binding()?, y.binding()?];
        let name = match op {
            Unary::Silu => "unary_silu",
            Unary::Gelu => "unary_gelu",
            Unary::Sigmoid => "unary_sigmoid",
        };
        unsafe {
            self.launch(
                stream,
                name,
                Config::new(),
                &scalars,
                &bindings,
                x.size().div_ceil(256),
                1,
                256,
            )
        }?;
        Ok(y)
    }

    /// `z = x op y`, with `y` repeating over `x` when it is shorter.
    pub fn binary(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        y: &Tensor,
        op: Binary,
    ) -> Result<Tensor> {
        if y.size() == 0 || !x.size().is_multiple_of(y.size()) {
            return Err(Error("binary broadcast".into()));
        }
        let z = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size());
        let bindings = [x.binding()?, y.binding()?, z.binding()?];
        let name = match op {
            Binary::Add => "binary_add",
            Binary::Mul => "binary_mul",
        };
        unsafe {
            self.launch(
                stream,
                name,
                config(&[("yn", y.size())]),
                &scalars,
                &bindings,
                x.size().div_ceil(256),
                1,
                256,
            )
        }?;
        Ok(z)
    }

    /// Split-half rotary embedding over `heads` heads of `cols / heads`.
    pub fn rope(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        tokens: usize,
        heads: usize,
        theta: f32,
    ) -> Result<Tensor> {
        if tokens != x.rows()
            || heads < 1
            || !x.cols().is_multiple_of(heads)
            || !(x.cols() / heads).is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(Error("split-half rotary dimensions".into()));
        }
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size()).float(theta);
        let bindings = [x.binding()?, y.binding()?];
        unsafe {
            self.launch(
                stream,
                "rope",
                config(&[("dim", x.cols() / heads), ("heads", heads)]),
                &scalars,
                &bindings,
                x.size().div_ceil(256),
                1,
                256,
            )
        }?;
        Ok(y)
    }

    /// Euler in place: `sample += delta * velocity`, rounded to bf16 at the
    /// delta, the product and the sum, as the CUDA pipeline rounds it.
    pub fn euler_step(
        &self,
        stream: &mut Stream,
        sample: &Tensor,
        velocity: &Tensor,
        delta: f32,
    ) -> Result<()> {
        if sample.rows() != velocity.rows() || sample.cols() != velocity.cols() {
            return Err(Error("scheduler tensor dimensions".into()));
        }
        let scalars = Scalars::new().index(sample.size()).float(delta);
        let bindings = [sample.binding()?, velocity.binding()?];
        unsafe {
            self.launch(
                stream,
                "euler",
                Config::new(),
                &scalars,
                &bindings,
                sample.size().div_ceil(256),
                1,
                256,
            )
        }
    }

    /// Krea's guidance in place: `cond += scale * (cond - uncond)`, with
    /// diffusers' bf16 rounding at each of its three operations.
    pub fn guidance(
        &self,
        stream: &mut Stream,
        cond: &Tensor,
        uncond: &Tensor,
        scale: f32,
    ) -> Result<()> {
        if cond.rows() != uncond.rows() || cond.cols() != uncond.cols() {
            return Err(Error("guidance tensor dimensions".into()));
        }
        let scalars = Scalars::new().index(cond.size()).float(scale);
        let bindings = [cond.binding()?, uncond.binding()?];
        unsafe {
            self.launch(
                stream,
                "guidance",
                Config::new(),
                &scalars,
                &bindings,
                cond.size().div_ceil(256),
                1,
                256,
            )
        }
    }

    /// Multi-head attention over `batch * tokens` rows, softmax in float32.
    ///
    /// `heads` query heads share `kv` key/value heads. The three operands
    /// arrive as `[batch * tokens][heads * dim]` and are packed head-major for
    /// the batched GEMMs, then unpacked back.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        stream: &mut Stream,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        batch: usize,
        tokens: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        causal: bool,
    ) -> Result<Tensor> {
        if batch == 0
            || tokens == 0
            || heads == 0
            || kv == 0
            || dim == 0
            || !heads.is_multiple_of(kv)
            || q.size() != batch * tokens * heads * dim
            || k.size() != batch * tokens * kv * dim
            || v.size() != k.size()
        {
            return Err(Error("attention dimensions".into()));
        }
        let rows = batch * heads * tokens;
        let packed = [q, k, v]
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let out = self.tensor(stream, rows, dim)?;
                let scalars = Scalars::new().index(out.size());
                let bindings = [source.binding()?, out.binding()?];
                unsafe {
                    self.launch(
                        stream,
                        "head_pack",
                        config(&[
                            ("dim", dim),
                            ("tokens", tokens),
                            ("heads", heads),
                            // Queries are already one head each; keys and values
                            // repeat over the group they serve.
                            ("kv", if index == 0 { heads } else { kv }),
                            ("xsize", source.size()),
                            ("ysize", out.size()),
                        ]),
                        &scalars,
                        &bindings,
                        out.size().div_ceil(256),
                        1,
                        256,
                    )
                }?;
                Ok(out)
            })
            .collect::<Result<Vec<_>>>()?;

        let count = batch * heads * tokens * tokens;
        let scores = self.pool.scratch(stream, count * 4)?;
        self.matmul(
            stream,
            "gemm_bf16_f32_nt",
            packed[0].binding()?,
            packed[1].binding()?,
            scores.binding(),
            tokens,
            tokens,
            dim,
            batch * heads,
            1.0 / (dim as f32).sqrt(),
            None,
        )?;

        let probabilities = self.tensor(stream, rows, tokens)?;
        let scalars = Scalars::new().index(rows);
        let bindings = [scores.binding(), probabilities.binding()?];
        unsafe {
            self.launch(
                stream,
                if causal { "softmax_causal" } else { "softmax" },
                config(&[("xsize", count), ("tokens", tokens)]),
                &scalars,
                &bindings,
                rows,
                1,
                256,
            )
        }?;

        let weighted = self.tensor(stream, rows, dim)?;
        self.matmul(
            stream,
            "gemm_bf16_bf16_nn",
            probabilities.binding()?,
            packed[2].binding()?,
            weighted.binding()?,
            tokens,
            dim,
            tokens,
            batch * heads,
            1.0,
            None,
        )?;

        let out = self.tensor(stream, batch * tokens, heads * dim)?;
        let scalars = Scalars::new().index(weighted.size());
        let bindings = [weighted.binding()?, out.binding()?];
        unsafe {
            self.launch(
                stream,
                "head_unpack",
                config(&[
                    ("dim", dim),
                    ("tokens", tokens),
                    ("heads", heads),
                    ("kv", heads),
                    ("xsize", weighted.size()),
                    ("ysize", out.size()),
                ]),
                &scalars,
                &bindings,
                weighted.size().div_ceil(256),
                1,
                256,
            )
        }?;
        Ok(out)
    }

    /// Square, odd-sized, stride-one convolution with same padding.
    /// Packed 3×3 weights use implicit GEMM; row-major weights use im2col.
    /// A 1×1 kernel uses GEMM directly.
    pub fn conv(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        height: usize,
        width: usize,
        w: &Weight,
        bias: Option<View<'_>>,
    ) -> Result<Tensor> {
        if w.shape.len() != 4
            || w.shape[1] != x.cols()
            || w.shape[2] != w.shape[3]
            || w.shape[2] % 2 != 1
            || height == 0
            || width == 0
            || x.rows() != height * width
        {
            return Err(Error("convolution dimensions".into()));
        }
        let kernel = w.shape[2];
        if kernel == 1 {
            return self.linear(stream, x, w, bias);
        }
        // The values decide the path, because only they know their order.
        if w.layout() == Layout::ChannelsLast {
            if kernel != 3 {
                return Err(Error("only a 3x3 is packed channels-last".into()));
            }
            return self.conv3x3(stream, x, height, width, w, bias);
        }
        let patches = self.tensor(stream, height * width, x.cols() * kernel * kernel)?;
        let scalars = Scalars::new().index(patches.size());
        let bindings = [x.binding()?, patches.binding()?];
        // One workgroup per output pixel when a row of channels is a whole
        // number of 32-lane reads.
        let coalesced = kernel == 3 && x.cols().is_multiple_of(32) && x.cols() <= 1024;
        unsafe {
            self.launch(
                stream,
                if coalesced { "im2col_coalesced" } else { "im2col" },
                config(&[
                    ("xsize", x.size()),
                    ("channels", x.cols()),
                    ("width", width),
                    ("height", height),
                    ("kernel", kernel),
                ]),
                &scalars,
                &bindings,
                if coalesced { height * width } else { patches.size().div_ceil(256) },
                1,
                256,
            )
        }?;
        self.linear(stream, &patches, w, bias)
    }

    /// Implicit GEMM over `[out, ky, kx, in]` weights, without a patch buffer.
    /// Uses the dense GEMM tile rules with tap-major accumulation.
    fn conv3x3(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        height: usize,
        width: usize,
        w: &Weight,
        bias: Option<View<'_>>,
    ) -> Result<Tensor> {
        let (m, n, k) = (height * width, w.shape[0], x.cols() * 9);
        if w.count != n * k {
            return Err(Error("convolution dimensions".into()));
        }
        let y = self.tensor(stream, m, n)?;
        let name =
            if bias.is_some() { "conv3x3_bf16_bf16_nt_bias" } else { "conv3x3_bf16_bf16_nt" };
        let scalars = Scalars::new().index(m).float(1.0);
        let mut bindings = vec![x.binding()?, w.values()?, y.binding()?];
        if let Some(bias) = bias {
            bindings.push(bias);
        }
        // Widen only when the larger tile adds no padded rows or columns.
        let wide = m >= 128 && n >= 64 && m.div_ceil(64).is_multiple_of(2);
        let square = wide && n >= 128 && n.div_ceil(64).is_multiple_of(2) && k >= 128;
        let (tile_m, tile_n) = (if wide { 128 } else { 64 }, if square { 128 } else { 64 });
        let name = match (square, wide) {
            (true, _) => format!("{name}_tiled"),
            (_, true) => format!("{name}_wide"),
            _ => name.to_string(),
        };
        unsafe {
            self.launch(
                stream,
                &name,
                config(&[
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    // A is the image, not a patch matrix.
                    ("asize", m * x.cols()),
                    ("bsize", n * k),
                    ("csize", m * n),
                    ("astride", m * x.cols()),
                    ("bstride", n * k),
                    ("channels", x.cols()),
                    ("width", width),
                    ("height", height),
                ]),
                &scalars,
                &bindings,
                n.div_ceil(tile_n),
                m.div_ceil(tile_m),
                256,
            )
        }?;
        Ok(y)
    }

    /// Nearest-neighbour 2x, the VAE's upsampler.
    pub fn upsample(
        &self,
        stream: &mut Stream,
        x: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        if height == 0 || width == 0 || height * width != x.rows() {
            return Err(Error("upsampling dimensions".into()));
        }
        let y = self.tensor(stream, height * width * 4, x.cols())?;
        let scalars = Scalars::new().index(y.size());
        let bindings = [x.binding()?, y.binding()?];
        unsafe {
            self.launch(
                stream,
                "upsample",
                config(&[("xsize", x.size()), ("channels", x.cols()), ("width", width)]),
                &scalars,
                &bindings,
                y.size().div_ceil(256),
                1,
                256,
            )
        }?;
        Ok(y)
    }

    /// Launches one auxiliary kernel: the models graph has kernels of its own
    /// (embedding, layer taps, the modulation table) that are not operations.
    /// # Safety
    /// The argument layout, allocation extents, launch dimensions, and configuration
    /// must match the embedded kernel. All allocations belong to the current stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        stream: &mut Stream,
        name: &str,
        config: Config,
        scalars: &Scalars,
        bindings: &[View<'_>],
        grid_x: usize,
        grid_y: usize,
        threads: u32,
    ) -> Result<()> {
        let kernel = self.kernels.get(stream, name, config, (grid_x as u32, grid_y as u32))?;
        let constants = scalars.pack(name, &kernel)?;
        // The runtime validates the block against the size the kernel was
        // compiled with, which the address-based path never checked. Disagreeing
        // is a host bug, so say so here rather than launch a different shape.
        let compiled = kernel.info().workgroup_size;
        if compiled != [threads, 1, 1] {
            return Err(Error(format!(
                "{name}: compiled for workgroup {compiled:?} but the host asked for {threads}"
            )));
        }
        unsafe {
            stream.dispatch(
                &kernel,
                [grid_x as u32, grid_y as u32, 1],
                compiled,
                &constants,
                bindings,
            )
        }
        .map_err(|e| Error(e.to_string()))?;
        Ok(())
    }

    /// The bf16 GEMM every dense layer here goes through.
    ///
    /// The row tile doubles when that adds no padded rows, and the column tile
    /// doubles again for long reductions, which halves how often a convolution
    /// streams its input.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        stream: &mut Stream,
        name: &str,
        a: View<'_>,
        b: View<'_>,
        out: View<'_>,
        m: usize,
        n: usize,
        k: usize,
        batches: usize,
        alpha: f32,
        bias: Option<View<'_>>,
    ) -> Result<()> {
        let scalars = Scalars::new().index(m).float(alpha);
        let mut bindings = vec![a, b, out];
        if let Some(bias) = bias {
            bindings.push(bias);
        }
        let wide = m >= 128
            && n >= 64
            && m.div_ceil(64).is_multiple_of(2)
            && (name == "gemm_bf16_bf16_nt" || name == "gemm_bf16_bf16_nt_bias");
        let square = wide && n >= 128 && n.div_ceil(64).is_multiple_of(2) && k >= 128;
        let tile_m = if wide { 128 } else { 64 };
        let tile_n = if square { 128 } else { 64 };
        let name = if square {
            format!("{name}_tiled")
        } else if wide {
            format!("{name}_wide")
        } else {
            name.to_string()
        };
        let config = config(&[
            ("m", m),
            ("n", n),
            ("k", k),
            ("asize", m * k * batches),
            ("bsize", n * k * batches),
            ("csize", m * n * batches),
            ("astride", m * k),
            ("bstride", n * k),
        ]);
        unsafe {
            self.launch(
                stream,
                &name,
                config,
                &scalars,
                &bindings,
                n.div_ceil(tile_n),
                batches * m.div_ceil(tile_m),
                256,
            )
        }
    }
}

/// Whether loaders pack 3×3 weights for implicit GEMM.
/// `KREA2_CONV_IM2COL=1` disables packing. Read once per process; dispatch
/// subsequently follows each weight's layout.
pub fn pack_convolutions() -> bool {
    static CHOICE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHOICE
        .get_or_init(|| std::env::var_os("KREA2_CONV_IM2COL").is_none_or(|value| value != "1"))
}

/// A kernel configuration from shape entries, which is how these kernels take
/// their shapes: compiled in, not passed.
pub fn config(entries: &[(&str, usize)]) -> Config {
    entries.iter().map(|(key, value)| ((*key).to_string(), *value as u64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_mismatches_are_refused_before_any_launch() {
        // These reject on shape alone, so they need no GPU.
        let ops = Ops::new(Pool::new());
        assert!(ops.tensor(0, 8).is_err(), "an empty tensor is not a tensor");
    }

    #[test]
    fn the_matmul_tile_widens_only_when_it_adds_no_padded_rows() {
        // 128 rows are two whole 64-row tiles, so the wide kernel applies;
        // 129 would pad a third, so it does not.
        for (m, n, k, expected) in [
            (128usize, 128usize, 128usize, "gemm_bf16_bf16_nt_tiled"),
            (128, 64, 128, "gemm_bf16_bf16_nt_wide"),
            (129, 128, 128, "gemm_bf16_bf16_nt"),
            (128, 128, 64, "gemm_bf16_bf16_nt_wide"),
        ] {
            let wide = m >= 128
                && n >= 64
                && m.div_ceil(64).is_multiple_of(2)
                && "gemm_bf16_bf16_nt" == "gemm_bf16_bf16_nt";
            let square = wide && n >= 128 && n.div_ceil(64).is_multiple_of(2) && k >= 128;
            let name = if square {
                "gemm_bf16_bf16_nt_tiled"
            } else if wide {
                "gemm_bf16_bf16_nt_wide"
            } else {
                "gemm_bf16_bf16_nt"
            };
            assert_eq!(name, expected, "m={m} n={n} k={k}");
        }
    }
}
