//! V transposed to `[kv_heads * 128][capacity]` for the fp16 attention kernel,
//! which stages a channel's 16 keys with one contiguous load.
use crate::kernels::{Config, cache::PreparedKernels};
use hrx::{Buffer, Stream};

use super::Result;

/// `v_transpose` for V at this capacity: `[tokens][kv_heads * 128]` in,
/// `[kv_heads * 128][capacity]` out.
fn transpose_config(kv_heads: usize, capacity: usize) -> Config {
    [("width", kv_heads * 128), ("row_capacity", capacity)]
        .into_iter()
        .map(|(key, value)| (key, value as u64))
        .collect()
}

/// A 32x32 tile per workgroup.
fn transpose_grid(tokens: usize, kv_heads: usize) -> (usize, usize) {
    (tokens.div_ceil(32), kv_heads * 128 / 32)
}

/// V transposed for the fp16 attention kernel, which stages a channel's keys
/// with one contiguous load.
pub(crate) struct VTranspose {
    kernel: hrx::Kernel,
    grid: (usize, usize),
    /// `[kv_heads * 128][capacity]`, zero past `tokens`.
    pub(crate) output: Buffer,
}

impl VTranspose {
    pub(crate) fn new(
        stream: &mut Stream,
        tokens: usize,
        capacity: usize,
        kv_heads: usize,
        compiler: Option<&str>,
    ) -> Result<VTranspose> {
        let grid = transpose_grid(tokens, kv_heads);
        let [x, y, _] = crate::kernels::grid("v_transpose", grid.0, grid.1)?;
        let config = transpose_config(kv_heads, capacity);
        let kernel =
            PreparedKernels::new(compiler).get(stream, "v_transpose", config, (x, y))?;
        let output = stream.allocate(kv_heads * capacity * 128 * 2)?;
        // The attention kernel reads the headroom columns, which are never written.
        stream.fill(output.binding(), 0)?;
        Ok(VTranspose { kernel, grid, output })
    }

    /// The kernel and its grid, for the session to launch with
    /// `[v, output]` bound and the token count as its scalar.
    pub(crate) fn launch(&self) -> (&hrx::Kernel, (usize, usize)) {
        (&self.kernel, self.grid)
    }
}
