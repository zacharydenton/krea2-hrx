//! The auxiliary operations the pipeline is built from: everything outside the
//! 28 blocks, in Loom kernels, driven from here.
//!
//! Each call compiles (or finds in the cache) the kernel for its exact shape
//! and launches it. The shapes are configuration, not arguments, which is why
//! the cache is keyed by them.
#![deny(unsafe_code)]

pub mod tensor;

use std::sync::{Arc, OnceLock};

use hrx::{device, Buffer, DevicePtr};
use loom::auxiliary_kernel;

pub use loom::Config;

pub use hrx::Args;
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
        Error(error.0)
    }
}

impl From<loom::Error> for Error {
    fn from(error: loom::Error) -> Self {
        Error(error.0)
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
/// plain one, and the third is GroupNorm-style over the row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Norm {
    OnePlusScale = 0,
    Scale = 1,
    Group = 2,
}

/// A weight already on the device: bf16 values, and the shape they came with.
///
/// The norms want their scales in float32. A checkpoint that kept a tensor in
/// float32 supplies them; for one stored as bf16 the upcast is made on first
/// use and kept, which is what the C++ `Weight::as_f32` did behind `mutable`.
pub struct Weight {
    /// bf16 values: what every kernel but the norms reads.
    pub values: DevicePtr,
    pub shape: Vec<usize>,
    pub count: usize,
    given: Option<DevicePtr>,
    upcast: OnceLock<Buffer>,
}

impl Weight {
    /// `f32` is the checkpoint's own float32 copy, when it kept one.
    pub fn new(
        values: DevicePtr,
        shape: Vec<usize>,
        count: usize,
        f32: Option<DevicePtr>,
    ) -> Weight {
        Weight { values, shape, count, given: f32, upcast: OnceLock::new() }
    }

    /// The scales as float32, upcast once if the file did not keep them so.
    pub fn f32_values(&self) -> Result<DevicePtr> {
        if let Some(pointer) = self.given {
            return Ok(pointer);
        }
        if let Some(buffer) = self.upcast.get() {
            return Ok(buffer.ptr());
        }
        let mut bits = vec![0u16; self.count];
        device().read(&mut bits, self.values)?;
        let floats: Vec<f32> = bits.into_iter().map(krea2_numerics::to_f32).collect();
        let buffer = device().allocate(floats.len() * 4)?;
        device().write(buffer.ptr(), &floats)?;
        // A losing race drops its buffer, which releases it.
        Ok(self.upcast.get_or_init(|| buffer).ptr())
    }
}

/// The operations, over one buffer pool.
pub struct Ops {
    pool: Arc<Pool>,
}

impl Ops {
    pub fn new(pool: Arc<Pool>) -> Ops {
        Ops { pool }
    }

    pub fn pool(&self) -> &Arc<Pool> {
        &self.pool
    }

    pub fn tensor(&self, rows: usize, cols: usize) -> Result<Tensor> {
        Tensor::new(&self.pool, rows, cols)
    }

    /// `y = x wᵀ (+ bias)`, the shape every linear layer here takes.
    pub fn linear(&self, x: &Tensor, w: &Weight, bias: Option<DevicePtr>) -> Result<Tensor> {
        let n = *w.shape.first().ok_or_else(|| Error("linear dimensions".into()))?;
        let k = x.cols;
        if n < 1 || x.rows < 1 || x.cols < 1 || w.count != n * k {
            return Err(Error("linear dimensions".into()));
        }
        let y = self.tensor(x.rows, n)?;
        let name = if bias.is_some() { "gemm_bf16_bf16_nt_bias" } else { "gemm_bf16_bf16_nt" };
        self.matmul(name, x.ptr(), w.values, y.ptr(), x.rows, n, k, 1, 1.0, bias)?;
        Ok(y)
    }

    /// RMSNorm with float32 scales.
    pub fn norm(&self, x: &Tensor, w: &Weight, mode: Norm, eps: f32) -> Result<Tensor> {
        if w.count != x.cols {
            return Err(Error("norm dimensions".into()));
        }
        let scales = w.f32_values()?;
        let y = self.tensor(x.rows, x.cols)?;
        let mut args = Args::new();
        args.i32(x.rows as i32).f32(eps).ptr(x.ptr()).ptr(scales).ptr(y.ptr());
        // Eight rows per workgroup when a row fits in one wave's registers.
        let wave = mode == Norm::Group && x.cols <= 1024;
        let name =
            if wave { "norm_2_wave".to_string() } else { format!("norm_{}", mode as u8) };
        let grid = if wave { x.rows.div_ceil(8) } else { x.rows };
        self.launch(
            &name,
            config(&[("xsize", x.size()), ("cols", x.cols)]),
            &args,
            grid,
            1,
            256,
        )?;
        Ok(y)
    }

    /// GroupNorm and SiLU in one pass, for the VAE's residual blocks.
    pub fn norm_silu(&self, x: &Tensor, w: &Weight) -> Result<Tensor> {
        if w.count != x.cols {
            return Err(Error("normalization dimensions".into()));
        }
        if x.cols > 1024 {
            let normed = self.norm(x, w, Norm::Group, 1e-5)?;
            return self.unary(&normed, Unary::Silu);
        }
        let scales = w.f32_values()?;
        let y = self.tensor(x.rows, x.cols)?;
        let mut args = Args::new();
        args.i32(x.rows as i32).f32(1e-5).ptr(x.ptr()).ptr(scales).ptr(y.ptr());
        self.launch(
            "norm_2_wave_silu",
            config(&[("xsize", x.size()), ("cols", x.cols)]),
            &args,
            x.rows.div_ceil(8),
            1,
            256,
        )?;
        Ok(y)
    }

    pub fn unary(&self, x: &Tensor, op: Unary) -> Result<Tensor> {
        let y = self.tensor(x.rows, x.cols)?;
        let mut args = Args::new();
        args.i32(x.size() as i32).ptr(x.ptr()).ptr(y.ptr());
        let name = match op {
            Unary::Silu => "unary_silu",
            Unary::Gelu => "unary_gelu",
            Unary::Sigmoid => "unary_sigmoid",
        };
        self.launch(name, Config::new(), &args, x.size().div_ceil(256), 1, 256)?;
        Ok(y)
    }

    /// `z = x op y`, with `y` repeating over `x` when it is shorter.
    pub fn binary(&self, x: &Tensor, y: &Tensor, op: Binary) -> Result<Tensor> {
        if y.size() == 0 || !x.size().is_multiple_of(y.size()) {
            return Err(Error("binary broadcast".into()));
        }
        let z = self.tensor(x.rows, x.cols)?;
        let mut args = Args::new();
        args.i32(x.size() as i32).ptr(x.ptr()).ptr(y.ptr()).ptr(z.ptr());
        let name = match op {
            Binary::Add => "binary_add",
            Binary::Mul => "binary_mul",
        };
        self.launch(name, config(&[("yn", y.size())]), &args, x.size().div_ceil(256), 1, 256)?;
        Ok(z)
    }

    /// Split-half rotary embedding over `heads` heads of `cols / heads`.
    pub fn rope(&self, x: &Tensor, tokens: usize, heads: usize, theta: f32) -> Result<Tensor> {
        if tokens != x.rows
            || heads < 1
            || !x.cols.is_multiple_of(heads)
            || !(x.cols / heads).is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(Error("split-half rotary dimensions".into()));
        }
        let y = self.tensor(x.rows, x.cols)?;
        let mut args = Args::new();
        args.i32(x.size() as i32).f32(theta).ptr(x.ptr()).ptr(y.ptr());
        self.launch(
            "rope",
            config(&[("dim", x.cols / heads), ("heads", heads)]),
            &args,
            x.size().div_ceil(256),
            1,
            256,
        )?;
        Ok(y)
    }

    /// Euler in place: `sample += delta * velocity`, rounded to bf16 at the
    /// delta, the product and the sum, as the CUDA pipeline rounds it.
    pub fn euler_step(&self, sample: &Tensor, velocity: &Tensor, delta: f32) -> Result<()> {
        if sample.rows != velocity.rows || sample.cols != velocity.cols {
            return Err(Error("scheduler tensor dimensions".into()));
        }
        let mut args = Args::new();
        args.i32(sample.size() as i32).f32(delta).ptr(sample.ptr()).ptr(velocity.ptr());
        self.launch("euler", Config::new(), &args, sample.size().div_ceil(256), 1, 256)
    }

    /// Krea's guidance in place: `cond += scale * (cond - uncond)`, with
    /// diffusers' bf16 rounding at each of its three operations.
    pub fn guidance(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<()> {
        if cond.rows != uncond.rows || cond.cols != uncond.cols {
            return Err(Error("guidance tensor dimensions".into()));
        }
        let mut args = Args::new();
        args.i32(cond.size() as i32).f32(scale).ptr(cond.ptr()).ptr(uncond.ptr());
        self.launch("guidance", Config::new(), &args, cond.size().div_ceil(256), 1, 256)
    }

    /// Multi-head attention over `batch * tokens` rows, softmax in float32.
    ///
    /// `heads` query heads share `kv` key/value heads. The three operands
    /// arrive as `[batch * tokens][heads * dim]` and are packed head-major for
    /// the batched GEMMs, then unpacked back.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
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
                let out = self.tensor(rows, dim)?;
                let mut args = Args::new();
                args.i32(out.size() as i32).ptr(source.ptr()).ptr(out.ptr());
                self.launch(
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
                    &args,
                    out.size().div_ceil(256),
                    1,
                    256,
                )?;
                Ok(out)
            })
            .collect::<Result<Vec<_>>>()?;

        let count = batch * heads * tokens * tokens;
        let scores = self.pool.scratch(count * 4)?;
        self.matmul(
            "gemm_bf16_f32_nt",
            packed[0].ptr(),
            packed[1].ptr(),
            scores.ptr(),
            tokens,
            tokens,
            dim,
            batch * heads,
            1.0 / (dim as f32).sqrt(),
            None,
        )?;

        let probabilities = self.tensor(rows, tokens)?;
        let mut args = Args::new();
        args.i32(rows as i32).ptr(scores.ptr()).ptr(probabilities.ptr());
        self.launch(
            if causal { "softmax_causal" } else { "softmax" },
            config(&[("xsize", count), ("tokens", tokens)]),
            &args,
            rows,
            1,
            256,
        )?;

        let weighted = self.tensor(rows, dim)?;
        self.matmul(
            "gemm_bf16_bf16_nn",
            probabilities.ptr(),
            packed[2].ptr(),
            weighted.ptr(),
            tokens,
            dim,
            tokens,
            batch * heads,
            1.0,
            None,
        )?;

        let out = self.tensor(batch * tokens, heads * dim)?;
        let mut args = Args::new();
        args.i32(weighted.size() as i32).ptr(weighted.ptr()).ptr(out.ptr());
        self.launch(
            "head_unpack",
            config(&[
                ("dim", dim),
                ("tokens", tokens),
                ("heads", heads),
                ("kv", heads),
                ("xsize", weighted.size()),
                ("ysize", out.size()),
            ]),
            &args,
            weighted.size().div_ceil(256),
            1,
            256,
        )?;
        Ok(out)
    }

    /// A square, odd-sized, stride-one, same-padded convolution as im2col and
    /// a GEMM. A 1x1 kernel is the GEMM alone.
    pub fn conv(
        &self,
        x: &Tensor,
        height: usize,
        width: usize,
        w: &Weight,
        bias: Option<DevicePtr>,
    ) -> Result<Tensor> {
        if w.shape.len() != 4
            || w.shape[1] != x.cols
            || w.shape[2] != w.shape[3]
            || w.shape[2] % 2 != 1
            || height == 0
            || width == 0
            || x.rows != height * width
        {
            return Err(Error("convolution dimensions".into()));
        }
        let kernel = w.shape[2];
        if kernel == 1 {
            return self.linear(x, w, bias);
        }
        let patches = self.tensor(height * width, x.cols * kernel * kernel)?;
        let mut args = Args::new();
        args.i32(patches.size() as i32).ptr(x.ptr()).ptr(patches.ptr());
        // One workgroup per output pixel when a row of channels is a whole
        // number of 32-lane reads.
        let coalesced = kernel == 3 && x.cols.is_multiple_of(32) && x.cols <= 1024;
        self.launch(
            if coalesced { "im2col_coalesced" } else { "im2col" },
            config(&[
                ("xsize", x.size()),
                ("channels", x.cols),
                ("width", width),
                ("height", height),
                ("kernel", kernel),
            ]),
            &args,
            if coalesced { height * width } else { patches.size().div_ceil(256) },
            1,
            256,
        )?;
        self.linear(&patches, w, bias)
    }

    /// Nearest-neighbour 2x, the VAE's upsampler.
    pub fn upsample(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        if height == 0 || width == 0 || height * width != x.rows {
            return Err(Error("upsampling dimensions".into()));
        }
        let y = self.tensor(height * width * 4, x.cols)?;
        let mut args = Args::new();
        args.i32(y.size() as i32).ptr(x.ptr()).ptr(y.ptr());
        self.launch(
            "upsample",
            config(&[("xsize", x.size()), ("channels", x.cols), ("width", width)]),
            &args,
            y.size().div_ceil(256),
            1,
            256,
        )?;
        Ok(y)
    }

    /// Launches one auxiliary kernel: the models graph has kernels of its own
    /// (embedding, layer taps, the modulation table) that are not operations.
    pub fn launch(
        &self,
        name: &str,
        config: Config,
        args: &Args,
        grid_x: usize,
        grid_y: usize,
        threads: u32,
    ) -> Result<()> {
        let kernel = auxiliary_kernel(name, &config, (grid_x as u32, grid_y as u32))?;
        kernel.launch_2d(grid_x as u32, grid_y as u32, threads, args)?;
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
        name: &str,
        a: DevicePtr,
        b: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        batches: usize,
        alpha: f32,
        bias: Option<DevicePtr>,
    ) -> Result<()> {
        let mut args = Args::new();
        args.i32(m as i32).f32(alpha).ptr(a).ptr(b).ptr(out);
        if let Some(bias) = bias {
            args.ptr(bias);
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
        self.launch(&name, config, &args, n.div_ceil(tile_n), batches * m.div_ceil(tile_m), 256)
    }
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
