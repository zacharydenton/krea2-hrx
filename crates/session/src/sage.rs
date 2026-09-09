//! The preparation pass the smoothed int4/int8 attention kernels take.
//!
//! Sage attention quantizes Q and K per token after subtracting a mean, and
//! recovers what the subtraction cost with a correction term the kernel adds
//! back to the scores. That is five kernels and a GEMM per block, which is why
//! the buffers are allocated once for the session's shape rather than per call.
//!
//! The nibble layout is head-major, `[heads][capacity][64 bytes]` for int4 and
//! 128 for int8, with `[heads][capacity]` float32 scales, so one key tile is
//! one contiguous block. The correction is
//! `[query heads][ceil(tokens / 64)][capacity]` in float32.
use hrx::{device, Args, Buffer, DevicePtr};
use loom::{cache::PreparedKernels, Config};

use crate::{Error, Result};

/// Buffers and shapes for one sequence length, reused every block.
pub struct Sage {
    kernels: PreparedKernels,
    tokens: usize,
    capacity: usize,
    heads: usize,
    kv_heads: usize,
    tiles: usize,
    bits: u32,
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

impl Sage {
    /// `bits` is 4 (codes -7..7, 64 bytes per head row) or 8 (-127..127, 128);
    /// the attention kernel of the same width consumes the output.
    pub fn new(
        tokens: usize,
        capacity: usize,
        heads: usize,
        kv_heads: usize,
        bits: u32,
        compiler: Option<&str>,
    ) -> Result<Sage> {
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
        // One head's codes: a nibble or a byte per channel of 128.
        let row_bytes = if bits == 4 { 64 } else { 128 };
        let sage = Sage {
            kernels: PreparedKernels::new(compiler),
            tokens,
            capacity,
            heads,
            kv_heads,
            tiles,
            bits,
            q4: device().allocate(capacity * heads * row_bytes)?,
            k4: device().allocate(capacity * kv_heads * row_bytes)?,
            q_scale: device().allocate(capacity * heads * 4)?,
            k_scale: device().allocate(capacity * kv_heads * 4)?,
            correction: device().allocate(heads * tiles * capacity * 4)?,
            v_transposed: device().allocate(kv_heads * capacity * 128 * 2)?,
            key_partial: device().allocate(tiles * kv_heads * 128 * 4)?,
            key_mean: device().allocate(kv_heads * 128 * 4)?,
            query_mean: device().allocate(heads * tiles * 128 * 4)?,
            query_mean_half: device().allocate(heads * tiles * 128 * 2)?,
            centered_k: device().allocate(kv_heads * capacity * 128 * 2)?,
        };
        // The padding past `tokens` is never written but is read as codes, so
        // it starts at zero rather than at whatever the allocator held.
        device().zero(sage.q4.ptr(), capacity * heads * row_bytes)?;
        device().zero(sage.q_scale.ptr(), capacity * heads * 4)?;
        device().zero(sage.v_transposed.ptr(), kv_heads * capacity * 128 * 2)?;
        Ok(sage)
    }

    /// Prepares one block's Q, K and V, in the session's stream order.
    pub fn run(&self, q: DevicePtr, k: DevicePtr, v: DevicePtr) -> Result<()> {
        let (t, c, h, kv, tiles) =
            (self.tokens, self.capacity, self.heads, self.kv_heads, self.tiles);

        // The key mean, over the whole sequence: a partial sum per tile, then
        // one workgroup per head to finish it.
        let mut args = Args::new();
        args.i32(t as i32).ptr(k).ptr(self.key_partial.ptr());
        self.launch(
            "sage_key_partial",
            &[
                ("tokens", t),
                ("heads", kv),
                ("tiles", tiles),
                ("xsize", c * kv * 128),
                ("ysize", tiles * kv * 128),
            ],
            &args,
            (kv as u32, tiles as u32),
            128,
        )?;
        let mut args = Args::new();
        args.i32(t as i32).ptr(self.key_partial.ptr()).ptr(self.key_mean.ptr());
        self.launch(
            "sage_key_mean",
            &[
                ("tokens", t),
                ("heads", kv),
                ("tiles", tiles),
                ("xsize", tiles * kv * 128),
                ("ysize", kv * 128),
            ],
            &args,
            (kv as u32, 1),
            128,
        )?;

        // The query mean is per tile, not per sequence: each query tile only
        // ever meets the keys once.
        let mut args = Args::new();
        args.i32(t as i32).ptr(q).ptr(self.query_mean.ptr()).ptr(self.query_mean_half.ptr());
        self.launch(
            "sage_query_mean",
            &[
                ("tokens", t),
                ("heads", h),
                ("tiles", tiles),
                ("xsize", c * h * 128),
                ("ysize", h * tiles * 128),
            ],
            &args,
            (tiles as u32, h as u32),
            128,
        )?;

        let mut args = Args::new();
        args.i32(t as i32)
            .ptr(q)
            .ptr(self.query_mean.ptr())
            .ptr(self.q4.ptr())
            .ptr(self.q_scale.ptr());
        self.launch(
            self.quantize("q"),
            &[
                ("tokens", t),
                ("heads", h),
                ("tiles", tiles),
                ("capacity", c),
                ("xsize", c * h * 128),
                ("msize", h * tiles * 128),
                ("psize", c * h * 32),
                ("ssize", c * h),
            ],
            &args,
            (t.div_ceil(8) as u32, h as u32),
            256,
        )?;
        let mut args = Args::new();
        args.i32(t as i32)
            .ptr(k)
            .ptr(self.key_mean.ptr())
            .ptr(self.k4.ptr())
            .ptr(self.k_scale.ptr())
            .ptr(self.centered_k.ptr());
        self.launch(
            self.quantize("k"),
            &[
                ("tokens", t),
                ("heads", kv),
                ("tiles", tiles),
                ("capacity", c),
                ("xsize", c * kv * 128),
                ("msize", kv * 128),
                ("psize", c * kv * 32),
                ("ssize", c * kv),
            ],
            &args,
            // Over the capacity, not the tokens: the padded rows are read.
            (c.div_ceil(8) as u32, kv as u32),
            256,
        )?;

        let mut args = Args::new();
        args.i32(t as i32).ptr(v).ptr(self.v_transposed.ptr());
        self.launch(
            "sage_transpose",
            &[("width", kv * 128), ("row_capacity", c)],
            &args,
            (t.div_ceil(32) as u32, (kv * 128 / 32) as u32),
            256,
        )?;

        // The correction: the query means against the centered keys, which is
        // what the kernel adds back to each score.
        let (m, n) = (tiles * 4, c);
        let (a_stride, b_stride) = (m * 128, n * 128);
        let mut args = Args::new();
        args.i32(m as i32)
            .f32(1.0)
            .ptr(self.query_mean_half.ptr())
            .ptr(self.centered_k.ptr())
            .ptr(self.correction.ptr());
        self.launch(
            "gemm_f16_f32_nt",
            &[
                ("m", m),
                ("n", n),
                ("k", 128),
                ("asize", a_stride * kv),
                ("bsize", b_stride * kv),
                ("csize", m * n * kv),
                ("astride", a_stride),
                ("bstride", b_stride),
            ],
            &args,
            (n.div_ceil(64) as u32, (kv * m.div_ceil(64)) as u32),
            256,
        )
    }

    fn quantize(&self, operand: &str) -> &'static str {
        match (self.bits, operand) {
            (4, "q") => "sage_quant_q",
            (4, _) => "sage_quant_k",
            (_, "q") => "sage_quant_q_i8",
            (_, _) => "sage_quant_k_i8",
        }
    }

    fn launch(
        &self,
        name: &str,
        config: &[(&str, usize)],
        args: &Args,
        grid: (u32, u32),
        threads: u32,
    ) -> Result<()> {
        let config: Config =
            config.iter().map(|(key, value)| ((*key).to_string(), *value as u64)).collect();
        let kernel = self.kernels.get(name, config, grid)?;
        unsafe { kernel.launch_2d(grid.0, grid.1, threads, args) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shapes_the_kernels_cannot_serve_are_refused() {
        // No device is touched: these fail on arithmetic alone.
        for (tokens, capacity, heads, kv, bits, why) in [
            (8usize, 64usize, 48usize, 12usize, 4u32, "too few tokens"),
            (64, 64, 48, 12, 5, "an unsupported width"),
            (64, 48, 48, 12, 4, "a capacity below the tiles"),
            (64, 66, 48, 12, 4, "a capacity off the 32-row grid"),
            (64, 64, 24, 12, 4, "a group size that is not four"),
        ] {
            let Err(error) = Sage::new(tokens, capacity, heads, kv, bits, None) else {
                panic!("{why} was accepted");
            };
            assert!(error.message.contains("unsupported Sage dimensions"), "{why}");
        }
    }
}
