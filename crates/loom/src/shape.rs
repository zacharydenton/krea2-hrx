//! The GEMM launch-shape rules, shared by the kernel builder and the session.
//!
//! `scripts/build_kernels.py` mirrors these for the Python path, and the
//! bundle's `launch.txt` records what they produced, so a session rejects a
//! bundle built under different rules.

/// The fp16 attention kernel's query tiles per workgroup.
///
/// `attention_query32` is 1.77x faster but regresses eight-step image quality
/// (17.81 dB against 24.57 on the trajectory gate) while passing every
/// per-block cosine check, so it stays a benchmark and every sequence length
/// runs one query tile.
pub const fn fp16_query_tiles(_tokens: i32) -> i32 {
    1
}

/// Operand row pitch in k elements.
///
/// A row of 8192 bytes (K = 16384 int4) makes the rows a GEMM step touches
/// alias in the cache; one extra k step of padding took the down projection
/// from 61 to 77 TOPS. Rows of 3072 bytes (K = 6144) showed no such effect, so
/// they stay dense.
pub const fn gemm_pitch(k: i32, bits: i32) -> i32 {
    if (k * bits / 8) % 8192 == 0 {
        k + 512 / bits
    } else {
        k
    }
}

/// Ceiling division for the positive counts these rules work in (`div_ceil` is
/// stable only for unsigned integers).
const fn tiles_of(count: i32, per_tile: i32) -> i32 {
    (count + per_tile - 1) / per_tile
}

/// m-tiles per raster group.
///
/// The 256-row kernels shorten their last raster group in-kernel, so they
/// always take the full group of 4. The 128-row kernels pad the grid: 1 for a
/// single tile row, else whichever of 4, 3, 2 pads the tile rows least, ties to
/// the larger group.
pub fn gemm_m_group(tokens: i32, rows: i32) -> i32 {
    let tiles = tiles_of(tokens, rows);
    if rows == 256 {
        return 4;
    }
    if tiles == 1 {
        return 1;
    }
    let mut best = 4;
    for candidate in [3, 2] {
        if tiles_of(tiles, candidate) * candidate < tiles_of(tiles, best) * best {
            best = candidate;
        }
    }
    best
}

/// Launch grid rows: the tiles themselves for the shortening 256-row kernels,
/// the tiles padded to whole raster groups for the 128-row kernels.
pub fn gemm_grid_rows(tokens: i32, rows: i32, m_group: i32) -> i32 {
    let tiles = tiles_of(tokens, rows);
    if rows == 256 {
        tiles
    } else {
        tiles_of(tiles, m_group) * m_group
    }
}

/// Below this many tokens the 256-row tile's rounding costs more than its rate.
pub const WIDE_TILE_TOKENS: i32 = 2048;

/// Workgroup tile rows.
///
/// The 256x128 tile runs 4-11% faster per row than the 128x128 tile (paired A/B
/// at 2064..16896 tokens) but rounds M up to 256, so it is chosen when its rows
/// are within 8% of the 128-row grid's padded rows and there are at least 2048
/// tokens. The int8 (W8A8) family exists only on the 256-row tile.
pub fn gemm_rows(tokens: i32, bits: i32) -> i32 {
    if bits == 8 {
        return 256;
    }
    if tokens < WIDE_TILE_TOKENS {
        return 128;
    }
    let wide = tiles_of(tokens, 256);
    let narrow = gemm_grid_rows(tokens, 128, gemm_m_group(tokens, 128));
    if 50 * wide <= 27 * narrow {
        256
    } else {
        128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values `tests/test_runtime.py` pins for the Python builder; both
    /// sides must agree or a bundle is rejected at load.
    #[test]
    fn the_shapes_match_the_pinned_ones() {
        assert_eq!([gemm_pitch(6144, 4), gemm_pitch(16384, 4)], [6144, 16512]);
        assert_eq!([gemm_pitch(6144, 8), gemm_pitch(16384, 8)], [6144, 16448]);
        assert_eq!(
            [16, 1040, 4115].map(|tokens| gemm_rows(tokens, 8)),
            [256, 256, 256],
            "int8 is always the wide tile"
        );
        assert_eq!(
            [16, 1040, 2047, 2064, 4115, 4353, 8192, 16896].map(|t| gemm_rows(t, 4)),
            [128, 128, 128, 256, 256, 256, 256, 256]
        );
        assert_eq!([16, 129, 4115, 8192].map(|t| gemm_m_group(t, 128)), [1, 2, 3, 4]);
        assert_eq!([4096, 4115, 16896].map(|t| gemm_m_group(t, 256)), [4, 4, 4]);
    }

    #[test]
    fn the_wide_tile_grid_is_its_tiles_and_the_narrow_one_pads_to_groups() {
        assert_eq!(gemm_grid_rows(4115, 256, 4), 17, "4115 / 256 rounded up");
        assert_eq!(gemm_grid_rows(4115, 128, 3), 33, "32.15 tiles padded to a multiple of 3");
        assert_eq!(gemm_grid_rows(16, 128, 1), 1);
    }
}
