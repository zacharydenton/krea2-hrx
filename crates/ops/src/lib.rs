//! The auxiliary operations the pipeline is built from: everything outside the
//! 28 blocks, in Loom kernels, driven from here.
//!
//! Each call compiles (or finds in the cache) the kernel for its exact shape
//! and launches it. The shapes are configuration, not arguments, which is why
//! the cache is keyed by them.
#![deny(unsafe_code)]

pub mod tensor;

use std::sync::Arc;

use hrx::{Args, DevicePtr};
use loom::{auxiliary_kernel, Config};

pub use tensor::{Pool, Tensor};

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
pub struct Weight {
    pub tensor: Tensor,
    pub shape: Vec<usize>,
    /// The float32 copy, for the norms that take their scales in f32.
    pub f32_values: Option<DevicePtr>,
}

impl Weight {
    pub fn count(&self) -> usize {
        self.tensor.size()
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
        if n < 1 || x.rows < 1 || x.cols < 1 || w.tensor.size() != n * k {
            return Err(Error("linear dimensions".into()));
        }
        let y = self.tensor(x.rows, n)?;
        let name = if bias.is_some() { "gemm_bf16_bf16_nt_bias" } else { "gemm_bf16_bf16_nt" };
        self.matmul(name, x.ptr(), w.tensor.ptr(), y.ptr(), x.rows, n, k, 1, 1.0, bias)?;
        Ok(y)
    }

    /// RMSNorm with float32 scales.
    pub fn norm(&self, x: &Tensor, w: &Weight, mode: Norm, eps: f32) -> Result<Tensor> {
        let scales = w.f32_values.ok_or_else(|| Error("norm scales are not float32".into()))?;
        if w.count() != x.cols {
            return Err(Error("norm dimensions".into()));
        }
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
        if w.count() != x.cols {
            return Err(Error("normalization dimensions".into()));
        }
        if x.cols > 1024 {
            let normed = self.norm(x, w, Norm::Group, 1e-5)?;
            return self.unary(&normed, Unary::Silu);
        }
        let scales = w.f32_values.ok_or_else(|| Error("norm scales are not float32".into()))?;
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

    fn launch(
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

fn config(entries: &[(&str, usize)]) -> Config {
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
