//! Auxiliary Loom operations for encoding, decoding and sampling.
//! Kernels specialize on tensor shape and launch geometry and are cached on first use.

pub mod tensor;

use std::sync::{Arc, OnceLock};

pub use crate::kernels::Scalars;
use crate::kernels::cache::PreparedKernels;
use hrx::{Buffer, BufferPool, Stream, View};

pub use crate::kernels::Config;

pub use crate::{Error, Result};
pub use tensor::Tensor;

/// Which pointwise function [`Ops::unary`] applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unary {
    /// `x * sigmoid(x)`.
    Silu,
    /// The tanh approximation of GELU.
    Gelu,
    /// `1 / (1 + exp(-x))`.
    Sigmoid,
}

/// Elementwise, with the right operand broadcast over the left.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Binary {
    /// `x + y`.
    Add,
    /// `x * y`.
    Mul,
}

/// How [`Ops::norm`] scales: `x * (1 + w)` is the DiT convention, `x * w` the
/// plain one; `Group` is the VAE's L2 normalization with bf16 rounding between
/// normalization, sqrt(width), and weight multiplication.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    /// `x * (1 + w)`, the DiT convention.
    OnePlusScale,
    /// `x * w`.
    Scale,
    /// The VAE's L2 normalization.
    Group,
}

impl Norm {
    /// The kernel that applies this scaling to one row per workgroup.
    fn kernel(self) -> &'static str {
        match self {
            Norm::OnePlusScale => "norm_0",
            Norm::Scale => "norm_1",
            Norm::Group => "norm_2",
        }
    }
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
    /// The logical shape, whatever the storage [`Layout`].
    pub shape: Vec<usize>,
    /// The element count, the product of `shape`.
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
        self.storage.try_slice(self.offset, self.count * 2).map_err(Error::from)
    }

    /// The same, for values written in `layout`.
    pub fn in_layout(mut self, layout: Layout) -> Weight {
        self.layout = layout;
        self
    }

    /// The order the values are stored in.
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
            return buffer.try_slice(*offset, self.count * 4).map_err(Error::from);
        }
        if let Some(buffer) = self.upcast.get() {
            return Ok(buffer.binding());
        }
        let mut bits = vec![0u16; self.count];
        stream.read_blocking(self.values()?, bytemuck::cast_slice_mut(&mut bits))?;
        let floats: Vec<f32> = bits.into_iter().map(crate::numerics::to_f32).collect();
        let buffer = stream.allocate(floats.len() * 4)?;
        stream.upload(buffer.binding(), bytemuck::cast_slice(&floats))?;
        // A losing race drops its buffer, which releases it.
        Ok(self.upcast.get_or_init(|| buffer).binding())
    }
}

/// The operations, over one buffer pool.
pub struct Ops {
    pool: Arc<BufferPool>,
    kernels: PreparedKernels,
    /// Compiler override for this operation set's auxiliary kernels.
    compiler: Option<String>,
}

impl Ops {
    /// Operations over `pool`, compiling with `HRX_LOOM_LIBRARY` or the
    /// pinned bundle.
    pub fn new(pool: Arc<BufferPool>) -> Ops {
        Ops { pool, compiler: None, kernels: PreparedKernels::default() }
    }

    /// The same, with an explicit compiler instead of `HRX_LOOM_LIBRARY` or the pinned bundle.
    pub fn with_compiler(pool: Arc<BufferPool>, compiler: Option<&str>) -> Ops {
        Ops {
            pool,
            compiler: compiler.map(str::to_string),
            kernels: PreparedKernels::new(compiler),
        }
    }

    /// The compiler override, if one was given.
    pub fn compiler(&self) -> Option<&str> {
        self.compiler.as_deref()
    }

    /// The pool every tensor these operations make comes from.
    pub fn pool(&self) -> &Arc<BufferPool> {
        &self.pool
    }

    /// An uninitialized `rows x cols` tensor from the pool.
    pub fn tensor(&self, stream: &Stream, rows: usize, cols: usize) -> Result<Tensor> {
        Tensor::new(&self.pool, stream, rows, cols)
    }

    /// `y = x wᵀ (+ bias)`, the shape every linear layer here takes.
    pub fn linear(
        &self,
        stream: &Stream,
        x: &Tensor,
        w: &Weight,
        bias: Option<View<'_>>,
    ) -> Result<Tensor> {
        let n = *w.shape.first().ok_or_else(|| Error::invalid("linear dimensions"))?;
        let k = x.cols();
        if n < 1 || x.rows() < 1 || x.cols() < 1 || w.count != n * k {
            return Err(Error::invalid("linear dimensions"));
        }
        check_bias(bias, n)?;
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
            return Err(Error::invalid("norm dimensions"));
        }
        let scales = w.f32_values(stream)?;
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.rows()).float(eps);
        let bindings = [x.binding()?, scales, y.binding()?];
        // Eight rows per workgroup when a row fits in one wave's registers.
        let wave = mode == Norm::Group && x.cols() <= 1024;
        let name = if wave { "norm_2_wave" } else { mode.kernel() };
        let grid = if wave { x.rows().div_ceil(8) } else { x.rows() };
        // SAFETY: `x` and `y` share a shape and the scales hold one float per
        // column, checked above; one workgroup per row, or per eight rows.
        unsafe {
            self.launch(
                stream,
                name,
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
            return Err(Error::invalid("normalization dimensions"));
        }
        if x.cols() > 1024 {
            let normed = self.norm(stream, x, w, Norm::Group, 1e-5)?;
            return self.unary(stream, &normed, Unary::Silu);
        }
        let scales = w.f32_values(stream)?;
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.rows()).float(1e-5);
        let bindings = [x.binding()?, scales, y.binding()?];
        // SAFETY: as `norm`'s wave path: rows of at most 1024 columns, checked
        // above, eight to a workgroup.
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

    /// `op` applied to every element of `x`.
    pub fn unary(&self, stream: &Stream, x: &Tensor, op: Unary) -> Result<Tensor> {
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size());
        let bindings = [x.binding()?, y.binding()?];
        let name = match op {
            Unary::Silu => "unary_silu",
            Unary::Gelu => "unary_gelu",
            Unary::Sigmoid => "unary_sigmoid",
        };
        // SAFETY: `x` and `y` are equal-shape tensors, one element per thread.
        unsafe { self.launch_1d(stream, name, Config::new(), &scalars, &bindings, x.size()) }?;
        Ok(y)
    }

    /// `z = x op y`, with `y` repeating over `x` when it is shorter.
    pub fn binary(
        &self,
        stream: &Stream,
        x: &Tensor,
        y: &Tensor,
        op: Binary,
    ) -> Result<Tensor> {
        if y.size() == 0 || !x.size().is_multiple_of(y.size()) {
            return Err(Error::invalid("binary broadcast"));
        }
        let z = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size());
        let bindings = [x.binding()?, y.binding()?, z.binding()?];
        let name = match op {
            Binary::Add => "binary_add",
            Binary::Mul => "binary_mul",
        };
        // SAFETY: `y` divides `x`, which the check above guarantees; `z` is `x`'s shape.
        unsafe {
            self.launch_1d(
                stream,
                name,
                config(&[("yn", y.size())]),
                &scalars,
                &bindings,
                x.size(),
            )
        }?;
        Ok(z)
    }

    /// Split-half rotary embedding over `heads` heads of `cols / heads`.
    pub fn rope(
        &self,
        stream: &Stream,
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
            return Err(Error::invalid("split-half rotary dimensions"));
        }
        let y = self.tensor(stream, x.rows(), x.cols())?;
        let scalars = Scalars::new().index(x.size()).float(theta);
        let bindings = [x.binding()?, y.binding()?];
        // SAFETY: `x` and `y` share a shape the check above validated against `heads`.
        unsafe {
            self.launch_1d(
                stream,
                "rope",
                config(&[("dim", x.cols() / heads), ("heads", heads)]),
                &scalars,
                &bindings,
                x.size(),
            )
        }?;
        Ok(y)
    }

    /// Euler in place: `sample += delta * velocity`, rounded to bf16 at the
    /// delta, the product and the sum, as the CUDA pipeline rounds it.
    pub fn euler_step(
        &self,
        stream: &Stream,
        sample: &Tensor,
        velocity: &Tensor,
        delta: f32,
    ) -> Result<()> {
        if sample.rows() != velocity.rows() || sample.cols() != velocity.cols() {
            return Err(Error::invalid("scheduler tensor dimensions"));
        }
        let scalars = Scalars::new().index(sample.size()).float(delta);
        let bindings = [sample.binding()?, velocity.binding()?];
        // SAFETY: `sample` and `velocity` share the shape checked above.
        unsafe {
            self.launch_1d(stream, "euler", Config::new(), &scalars, &bindings, sample.size())
        }
    }

    /// Krea's guidance in place: `cond += scale * (cond - uncond)`, with
    /// diffusers' bf16 rounding at each of its three operations.
    pub fn guidance(
        &self,
        stream: &Stream,
        cond: &Tensor,
        uncond: &Tensor,
        scale: f32,
    ) -> Result<()> {
        if cond.rows() != uncond.rows() || cond.cols() != uncond.cols() {
            return Err(Error::invalid("guidance tensor dimensions"));
        }
        let scalars = Scalars::new().index(cond.size()).float(scale);
        let bindings = [cond.binding()?, uncond.binding()?];
        // SAFETY: `cond` and `uncond` share the shape checked above.
        unsafe {
            self.launch_1d(stream, "guidance", Config::new(), &scalars, &bindings, cond.size())
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
        stream: &Stream,
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
        let dimensions = || Error::invalid("attention dimensions");
        let queries = product(&[batch, tokens, heads, dim]).ok_or_else(dimensions)?;
        let keys = product(&[batch, tokens, kv, dim]).ok_or_else(dimensions)?;
        // The score matrix, `rows x tokens` float32, is the largest operand.
        let count = product(&[batch, heads, tokens, tokens, 4]).ok_or_else(dimensions)? / 4;
        if kv == 0
            || !heads.is_multiple_of(kv)
            || q.size() != queries
            || k.size() != keys
            || v.size() != keys
        {
            return Err(dimensions());
        }
        let rows = batch * heads * tokens;
        let packed = [q, k, v]
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let out = self.tensor(stream, rows, dim)?;
                let scalars = Scalars::new().index(out.size());
                let bindings = [source.binding()?, out.binding()?];
                // SAFETY: `source` is `[batch * tokens][heads or kv][dim]`, checked above; `out`
                // holds `rows * dim` and the configuration names both sizes.
                unsafe {
                    self.launch_1d(
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
                        out.size(),
                    )
                }?;
                Ok(out)
            })
            .collect::<Result<Vec<_>>>()?;

        let scores = self.pool.acquire(stream, count * 4)?;
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
        // SAFETY: `scores` holds `count` floats and `probabilities` as many
        // bf16 values, one row of `tokens` per workgroup.
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
        // SAFETY: `weighted` is `[batch * heads * tokens][dim]` and `out` its unpacked
        // `[batch * tokens][heads * dim]`; the configuration names both sizes.
        unsafe {
            self.launch_1d(
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
                weighted.size(),
            )
        }?;
        Ok(out)
    }

    /// Square, odd-sized, stride-one convolution with same padding.
    /// Packed 3×3 weights use implicit GEMM; row-major weights use im2col.
    /// A 1×1 kernel uses GEMM directly.
    pub fn conv(
        &self,
        stream: &Stream,
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
            return Err(Error::invalid("convolution dimensions"));
        }
        let kernel = w.shape[2];
        if kernel == 1 {
            return self.linear(stream, x, w, bias);
        }
        // The values decide the path, because only they know their order.
        if w.layout() == Layout::ChannelsLast {
            if kernel != 3 {
                return Err(Error::invalid("only a 3x3 is packed channels-last"));
            }
            return self.conv3x3(stream, x, height, width, w, bias);
        }
        let patches = self.tensor(stream, height * width, x.cols() * kernel * kernel)?;
        let scalars = Scalars::new().index(patches.size());
        let bindings = [x.binding()?, patches.binding()?];
        // One workgroup per output pixel when a row of channels is a whole
        // number of 32-lane reads.
        let coalesced = kernel == 3 && x.cols().is_multiple_of(32) && x.cols() <= 1024;
        // SAFETY: `x` is `height * width` pixels, checked above, and `patches`
        // holds `kernel²` taps of every channel for each of them.
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
        stream: &Stream,
        x: &Tensor,
        height: usize,
        width: usize,
        w: &Weight,
        bias: Option<View<'_>>,
    ) -> Result<Tensor> {
        let (m, n, k) = (height * width, w.shape[0], x.cols() * 9);
        if w.count != n * k {
            return Err(Error::invalid("convolution dimensions"));
        }
        check_bias(bias, n)?;
        let y = self.tensor(stream, m, n)?;
        let base =
            if bias.is_some() { "conv3x3_bf16_bf16_nt_bias" } else { "conv3x3_bf16_bf16_nt" };
        let scalars = Scalars::new().index(m).float(1.0);
        let (x_values, y_values) = (x.binding()?, y.binding()?);
        let all = [x_values, w.values()?, y_values, bias.unwrap_or(y_values)];
        let bindings = &all[..3 + usize::from(bias.is_some())];
        let Tile { name, rows: tile_m, columns: tile_n } = tile(base, m, n, k);
        // SAFETY: the weights hold `n * k` values (checked above), the bias
        // `n` (checked by `check_bias`), and `x` and `y` are `m` rows of the
        // input and output channels; the grid covers `y` in whole tiles.
        unsafe {
            self.launch(
                stream,
                name,
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
                bindings,
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
        stream: &Stream,
        x: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        if height == 0 || width == 0 || height * width != x.rows() {
            return Err(Error::invalid("upsampling dimensions"));
        }
        let y = self.tensor(stream, height * width * 4, x.cols())?;
        let scalars = Scalars::new().index(y.size());
        let bindings = [x.binding()?, y.binding()?];
        // SAFETY: `y` holds four pixels per input pixel of `x`, whose rows are checked
        // against `height * width` above.
        unsafe {
            self.launch_1d(
                stream,
                "upsample",
                config(&[("xsize", x.size()), ("channels", x.cols()), ("width", width)]),
                &scalars,
                &bindings,
                y.size(),
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
        stream: &Stream,
        name: &str,
        config: Config,
        scalars: &Scalars,
        bindings: &[View<'_>],
        grid_x: usize,
        grid_y: usize,
        threads: u32,
    ) -> Result<()> {
        let grid = crate::kernels::grid(name, grid_x, grid_y)?;
        let kernel = self.kernels.get(stream, name, config, (grid[0], grid[1]))?;
        let constants = scalars.pack(name, &kernel)?;
        let block = crate::kernels::workgroup(name, &kernel, threads)?;
        // SAFETY: the caller vouches for the bindings, extents and grid (this
        // function's contract), and the block is the one the kernel was
        // compiled for.
        unsafe { stream.dispatch(&kernel, grid, block, &constants, bindings) }?;
        Ok(())
    }

    /// [`Ops::launch`] for an elementwise kernel: one 256-thread workgroup per
    /// 256 of `elements`.
    ///
    /// # Safety
    /// As [`Ops::launch`].
    pub unsafe fn launch_1d(
        &self,
        stream: &Stream,
        name: &str,
        config: Config,
        scalars: &Scalars,
        bindings: &[View<'_>],
        elements: usize,
    ) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe {
            self.launch(stream, name, config, scalars, bindings, elements.div_ceil(256), 1, 256)
        }
    }

    /// The bf16 GEMM every dense layer here goes through, at the tile [`tile`]
    /// chooses for its shape.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        stream: &Stream,
        name: &'static str,
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
        let all = [a, b, out, bias.unwrap_or(out)];
        let bindings = &all[..3 + usize::from(bias.is_some())];
        let Tile { name, rows: tile_m, columns: tile_n } = tile(name, m, n, k);
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
        // SAFETY: every caller passes operands of `m x k`, `n x k` and `m x n`
        // per batch, and a bias of `n` checked by `check_bias`; the grid
        // covers the output in whole tiles.
        unsafe {
            self.launch(
                stream,
                name,
                config,
                &scalars,
                bindings,
                n.div_ceil(tile_n),
                batches * m.div_ceil(tile_m),
                256,
            )
        }
    }
}

/// The dense kernel a GEMM runs as, and the output tile it covers per
/// workgroup.
#[derive(Debug, PartialEq, Eq)]
struct Tile {
    name: &'static str,
    rows: usize,
    columns: usize,
}

/// The tile for an `m x n x k` product through the `base` kernel.
///
/// The row tile doubles when that adds no padded rows, and the column tile
/// doubles again for long reductions, which halves how often a convolution
/// streams its input. Only the bf16-output kernels have the larger tiles.
fn tile(base: &'static str, m: usize, n: usize, k: usize) -> Tile {
    const VARIANTS: [(&str, &str, &str); 4] = [
        ("gemm_bf16_bf16_nt", "gemm_bf16_bf16_nt_wide", "gemm_bf16_bf16_nt_tiled"),
        (
            "gemm_bf16_bf16_nt_bias",
            "gemm_bf16_bf16_nt_bias_wide",
            "gemm_bf16_bf16_nt_bias_tiled",
        ),
        ("conv3x3_bf16_bf16_nt", "conv3x3_bf16_bf16_nt_wide", "conv3x3_bf16_bf16_nt_tiled"),
        (
            "conv3x3_bf16_bf16_nt_bias",
            "conv3x3_bf16_bf16_nt_bias_wide",
            "conv3x3_bf16_bf16_nt_bias_tiled",
        ),
    ];
    let variants = VARIANTS.iter().find(|(name, ..)| *name == base);
    let wide = variants.is_some() && m >= 128 && n >= 64 && m.div_ceil(64).is_multiple_of(2);
    let square = wide && n >= 128 && n.div_ceil(64).is_multiple_of(2) && k >= 128;
    let name = match variants {
        Some((_, _, tiled)) if square => tiled,
        Some((_, wide_name, _)) if wide => wide_name,
        _ => base,
    };
    Tile { name, rows: if wide { 128 } else { 64 }, columns: if square { 128 } else { 64 } }
}

/// The product of `dimensions`, or `None` when it is zero or overflows.
fn product(dimensions: &[usize]) -> Option<usize> {
    dimensions.iter().try_fold(1usize, |total, &d| total.checked_mul(d)).filter(|&n| n > 0)
}

/// A bias is one bf16 per output column. The kernels read `n` of them with no
/// bounds check of their own, so a shorter view would be read past its end.
fn check_bias(bias: Option<View<'_>>, n: usize) -> Result<()> {
    match bias {
        Some(bias) if bias.len() != n * 2 => Err(Error::invalid(format!(
            "bias spans {} bytes, expected {} for {n} bf16 columns",
            bias.len(),
            n * 2
        ))),
        _ => Ok(()),
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
pub fn config(entries: &[(&'static str, usize)]) -> Config {
    entries.iter().map(|&(key, value)| (key, value as u64)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_tile_widens_only_when_it_adds_no_padded_rows() {
        // 128 rows are two whole 64-row tiles, so the wide kernel applies;
        // 129 would pad a third, so it does not.
        for (base, m, n, k, expected) in [
            ("gemm_bf16_bf16_nt", 128, 128, 128, ("gemm_bf16_bf16_nt_tiled", 128, 128)),
            ("gemm_bf16_bf16_nt", 128, 64, 128, ("gemm_bf16_bf16_nt_wide", 128, 64)),
            ("gemm_bf16_bf16_nt", 129, 128, 128, ("gemm_bf16_bf16_nt", 64, 64)),
            ("gemm_bf16_bf16_nt", 128, 128, 64, ("gemm_bf16_bf16_nt_wide", 128, 64)),
            (
                "gemm_bf16_bf16_nt_bias",
                256,
                256,
                256,
                ("gemm_bf16_bf16_nt_bias_tiled", 128, 128),
            ),
            (
                "conv3x3_bf16_bf16_nt_bias",
                256,
                64,
                1152,
                ("conv3x3_bf16_bf16_nt_bias_wide", 128, 64),
            ),
            // The float32-output and batched kernels have only the one tile.
            ("gemm_bf16_f32_nt", 4096, 4096, 128, ("gemm_bf16_f32_nt", 64, 64)),
            ("gemm_bf16_bf16_nn", 4096, 128, 4096, ("gemm_bf16_bf16_nn", 64, 64)),
        ] {
            let (name, rows, columns) = expected;
            assert_eq!(tile(base, m, n, k), Tile { name, rows, columns }, "{base} {m}x{n}x{k}");
        }
    }

    #[test]
    fn every_tile_variant_is_an_embedded_kernel() {
        for base in [
            "gemm_bf16_bf16_nt",
            "gemm_bf16_bf16_nt_bias",
            "conv3x3_bf16_bf16_nt",
            "conv3x3_bf16_bf16_nt_bias",
        ] {
            for (m, n, k) in [(64, 64, 64), (128, 64, 64), (128, 128, 128)] {
                let name = tile(base, m, n, k).name;
                assert!(crate::kernels::sources::auxiliary(name).is_some(), "{name}");
            }
        }
    }
}
