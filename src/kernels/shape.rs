//! The int8 GEMM launch-shape rules, shared by the kernel builder, the upload
//! plan and the session.
//!
//! The prepared artifact shape carries these rules. A session rejects a
//! bundle built under different rules.

/// The sequence lengths the block kernels serve, text and image tokens together.
pub const TOKENS: std::ops::RangeInclusive<usize> = 16..=16896;

/// Buffer rows for a sequence: `tokens + 16` of headroom rounded to 32-row
/// query tiles, and at least whole 64-key blocks.
pub const fn capacity(tokens: usize) -> usize {
    let tiles = (tokens + 16).div_ceil(32) * 32;
    let blocks = tokens.div_ceil(64) * 64;
    if tiles > blocks {
        tiles
    } else {
        blocks
    }
}

/// Workgroup tile rows of the int8 (W8A8) GEMMs. The family exists only on the
/// 256x128 tile, which shortens its last raster group in-kernel, so the launch
/// grid is simply the tiles themselves.
pub const GEMM_ROWS: usize = 256;

/// m-tiles per raster group of the int8 GEMMs.
pub const GEMM_M_GROUP: usize = 4;

/// Operand row pitch in int8 elements, which are also bytes.
///
/// A row of a multiple of 8192 bytes makes the rows a GEMM step touches alias
/// in the cache; one extra k step of padding took the down projection (K =
/// 16384) from 61 to 77 TOPS. Rows of 6144 bytes showed no such effect, so they
/// stay dense.
pub const fn gemm_pitch(k: usize) -> usize {
    if k.is_multiple_of(8192) {
        k + 64
    } else {
        k
    }
}

/// Launch grid rows for `tokens` rows of activations.
pub const fn gemm_grid_rows(tokens: usize) -> usize {
    tokens.div_ceil(GEMM_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pitches the checkpoint plan, the kernels and the session must agree
    /// on, or a bundle is rejected at load.
    #[test]
    fn only_rows_that_alias_are_padded() {
        assert_eq!([gemm_pitch(6144), gemm_pitch(16384)], [6144, 16448]);
    }

    #[test]
    fn capacity_leaves_headroom_and_whole_key_blocks() {
        assert_eq!(capacity(4115), 4160);
        assert_eq!(capacity(16), 64);
        assert_eq!(capacity(2048), 2080);
    }

    #[test]
    fn the_grid_is_the_tiles_themselves() {
        assert_eq!(gemm_grid_rows(4115), 17, "4115 / 256 rounded up");
        assert_eq!(gemm_grid_rows(256), 1);
        assert_eq!(gemm_grid_rows(16), 1);
    }
}
