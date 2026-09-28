//! The transformer launch-shape rules, shared by the kernel builder, the upload
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
    if tiles > blocks { tiles } else { blocks }
}

/// Query rows per attention workgroup. Two groups of sixteen rows share K/V
/// staging for long sequences; shorter sequences retain the four-wave kernel.
/// The crossover is conservative: the measured gain at 6163 tokens was only 3%,
/// versus 11% at 8195 and 25% at 12301 on gfx1151.
pub const fn attention_rows(tokens: usize) -> usize {
    if tokens >= 8192 { 32 } else { 16 }
}

/// Attention source paired with [`attention_rows`].
pub const fn attention_source(tokens: usize) -> &'static str {
    if attention_rows(tokens) == 32 {
        "attention_gqa_lds_f16_wmma_q32"
    } else {
        super::ATTENTION_SOURCE
    }
}

/// Workgroup tile rows of the int8 (W8A8) GEMMs. The family exists on 256x128
/// and 256x256 tiles, which shorten their last raster group in-kernel, so the
/// launch grid is simply the tiles themselves.
pub const GEMM_ROWS: usize = 256;

/// Tile columns and workgroup threads of an int8 GEMM kernel. The 256x256 tile
/// reads a third less operand data per product, which pays only where K is
/// long: in the 1024x1024 pipeline it takes the down projection (K = 16384)
/// from 825 to 538 ms per step on the GPU clock, while gate/up, qkvg and wo
/// (K = 6144) gain nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmTile {
    /// 256x128, eight waves.
    Narrow,
    /// 256x256, sixteen waves.
    Wide,
}

impl GemmTile {
    /// Output columns per workgroup.
    pub const fn columns(self) -> usize {
        match self {
            GemmTile::Narrow => 128,
            GemmTile::Wide => 256,
        }
    }

    /// Threads per workgroup: four rows of waves, one wave per 64 columns.
    pub const fn threads(self) -> u32 {
        match self {
            GemmTile::Narrow => 256,
            GemmTile::Wide => 512,
        }
    }
}

/// The tile each transformer GEMM runs on, by bundle stem; see [`GemmTile`].
pub const GEMM_TILES: [(&str, GemmTile); 4] = [
    ("gemm_qkvg", GemmTile::Narrow),
    ("gemm_gu", GemmTile::Narrow),
    ("gemm_wo", GemmTile::Narrow),
    ("gemm_down", GemmTile::Wide),
];

/// The tile `stem` runs on; any GEMM not in [`GEMM_TILES`] is narrow.
pub fn gemm_tile(stem: &str) -> GemmTile {
    GEMM_TILES
        .iter()
        .find(|(name, _)| *name == stem)
        .map_or(GemmTile::Narrow, |&(_, tile)| tile)
}

/// The kernel source of an int8 GEMM family on `tile`. Only the residual
/// family has a 256x256 sibling; the others would gain nothing from it.
pub fn gemm_source(family: &'static str, tile: GemmTile) -> &'static str {
    match (family, tile) {
        ("gemm_i8_resid_256", GemmTile::Wide) => "gemm_i8_resid_256x256",
        _ => family,
    }
}

/// m-tiles per raster group of the int8 GEMMs.
pub const GEMM_M_GROUP: usize = 4;

/// Operand row pitch in int8 elements, which are also bytes.
///
/// A row of a multiple of 8192 bytes makes the rows a GEMM step touches alias
/// in the cache; one extra k step of padding took the down projection (K =
/// 16384) from 61 to 77 TOPS. Rows of 6144 bytes showed no such effect, so they
/// stay dense.
pub const fn gemm_pitch(k: usize) -> usize {
    if k.is_multiple_of(8192) { k + 64 } else { k }
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
    fn attention_tiles_fit_every_supported_capacity() {
        assert_eq!(attention_rows(8191), 16);
        assert_eq!(attention_rows(8192), 32);
        for tokens in TOKENS {
            let rows = attention_rows(tokens);
            assert!(capacity(tokens) >= tokens.div_ceil(rows) * rows);
            assert_eq!(attention_source(tokens).ends_with("_q32"), rows == 32);
        }
    }

    #[test]
    fn the_grid_is_the_tiles_themselves() {
        assert_eq!(gemm_grid_rows(4115), 17, "4115 / 256 rounded up");
        assert_eq!(gemm_grid_rows(256), 1);
        assert_eq!(gemm_grid_rows(16), 1);
    }
}
