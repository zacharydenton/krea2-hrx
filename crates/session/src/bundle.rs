//! The compiled kernel bundle a session runs on, and the metadata contract
//! that says it was built by the same rules this host derives.
use std::path::Path;

use hrx::Kernel;
use loom::shape;

use crate::{Error, Result, HIDDEN, INTER};

/// What `launch.txt` records. Version 4 is the bf16 residual stream; version 5
/// adds the fp16 query tile count. Version 3 and earlier expect an fp16 stream
/// and are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    pub version: u32,
    pub tokens: u32,
    pub gemm_rows: u32,
    pub m_group: u32,
    pub capacity: usize,
    pub attention_waves: u32,
    pub pitch_hidden: u32,
    pub pitch_inter: u32,
    pub attention_bits: u32,
    pub gemm_bits: u32,
    pub fp16_query_tiles: u32,
}

/// Buffer capacity for a sequence: `tokens + 16` of headroom, rounded to whole
/// 64-key blocks, and wider still when a query tile spans 32 rows.
#[allow(clippy::manual_div_ceil)] // the first branch floors on purpose
pub fn capacity(tokens: usize, fp16_query_tiles: u32) -> usize {
    if fp16_query_tiles == 2 {
        // Deliberately a floor: tokens + 79 rounded *down* to whole 64-key
        // blocks, which is 16 rows of headroom past the 32-row query tile.
        (tokens + 79) / 64 * 64
    } else {
        std::cmp::max((tokens + 16).div_ceil(32) * 32, tokens.div_ceil(64) * 64)
    }
}

impl Metadata {
    /// Parses the single line and checks every field against what this host
    /// derives for the same sequence, so a bundle from other rules is refused
    /// before any weights are read.
    pub fn parse(text: &str, tokens: usize) -> Result<Metadata> {
        let bad = || {
            Error::invalid(
                "invalid kernel launch metadata; rebuild with scripts/build_kernels.py",
            )
        };
        let mut fields = text.split_whitespace().map(str::parse::<u64>);
        let mut next =
            || -> Result<u64> { fields.next().transpose().ok().flatten().ok_or_else(bad) };
        let metadata = Metadata {
            version: next()? as u32,
            tokens: next()? as u32,
            gemm_rows: next()? as u32,
            m_group: next()? as u32,
            capacity: next()? as usize,
            attention_waves: next()? as u32,
            pitch_hidden: next()? as u32,
            pitch_inter: next()? as u32,
            attention_bits: next()? as u32,
            gemm_bits: next()? as u32,
            fp16_query_tiles: 1,
        };
        let metadata = match metadata.version {
            4 => metadata,
            5 => Metadata { fp16_query_tiles: next()? as u32, ..metadata },
            _ => return Err(bad()),
        };
        let expected_tiles = if metadata.version == 5 && metadata.attention_bits == 16 {
            shape::fp16_query_tiles(tokens as i32) as u32
        } else {
            1
        };
        let bits = metadata.gemm_bits as i32;
        let derived = metadata.tokens == tokens as u32
            && metadata.fp16_query_tiles == expected_tiles
            && matches!(metadata.attention_bits, 4 | 8 | 16)
            && matches!(metadata.gemm_bits, 4 | 8)
            && metadata.capacity == capacity(tokens, metadata.fp16_query_tiles)
            && metadata.gemm_rows == shape::gemm_rows(tokens as i32, bits) as u32
            && metadata.m_group
                == shape::gemm_m_group(tokens as i32, metadata.gemm_rows as i32) as u32
            && metadata.attention_waves == if tokens < 8192 { 8 } else { 4 }
            && metadata.pitch_hidden == shape::gemm_pitch(HIDDEN, bits) as u32
            && metadata.pitch_inter == shape::gemm_pitch(INTER, bits) as u32;
        if !derived {
            return Err(bad());
        }
        Ok(metadata)
    }

    /// The GEMM operand width as a suffix: `i4` or `i8`.
    pub fn width(&self) -> String {
        format!("i{}", self.gemm_bits)
    }

    /// The 256-row kernels carry a suffix; the 128-row ones do not.
    pub fn tile(&self) -> &'static str {
        if self.gemm_rows == 256 {
            "_256"
        } else {
            ""
        }
    }

    /// The attention kernel this bundle was built with. 4 and 8 are the
    /// smoothed Sage kernels, which take the preparation pass; 16 is fp16 QK
    /// and PV straight from the RoPE outputs.
    pub fn attention_symbol(&self) -> String {
        let mut name = match (self.attention_bits, self.fp16_query_tiles) {
            (16, 2) => "krea2_attention_query32".to_string(),
            (16, _) => "krea2_attention_gqa_lds_f16_wmma".to_string(),
            (4, _) => "krea2_attention_sage_i4_fast".to_string(),
            (_, _) => "krea2_attention_sage_i8_fast".to_string(),
        };
        if self.attention_bits != 16 && self.attention_waves != 8 {
            name.push_str("_prefetch");
        }
        name
    }
}

/// Every kernel one block needs, loaded from a bundle directory.
pub struct Kernels {
    pub prepare_norm: Kernel,
    pub prepare_gated: Kernel,
    pub prepare_swiglu: Kernel,
    pub gemm_qkvg: Kernel,
    pub gemm_gu: Kernel,
    pub gemm_wo: Kernel,
    pub gemm_down: Kernel,
    pub rope: Kernel,
    pub attention: Kernel,
    pub attention_transpose: Option<Kernel>,
}

impl Kernels {
    pub fn load(directory: &Path, metadata: &Metadata) -> Result<Kernels> {
        let load = |stem: &str, symbol: &str| -> Result<Kernel> {
            Kernel::load(&directory.join(format!("{stem}.hsaco")), symbol).map_err(Error::from)
        };
        let width = metadata.width();
        let tile = metadata.tile();
        Ok(Kernels {
            prepare_norm: load(
                &format!("prepare_norm_{width}"),
                &format!("krea2_prepare_norm_{width}"),
            )?,
            prepare_gated: load(
                &format!("prepare_gated_{width}"),
                &format!("krea2_prepare_gated_{width}"),
            )?,
            prepare_swiglu: load(
                &format!("prepare_plain_{width}"),
                &format!("krea2_prepare_plain_{width}"),
            )?,
            gemm_qkvg: load("gemm_qkvg", &format!("krea2_gemm_{width}{tile}"))?,
            gemm_gu: load("gemm_gu", &format!("krea2_gemm_{width}_swiglu{tile}"))?,
            gemm_wo: load("gemm_wo", &format!("krea2_gemm_{width}_resid{tile}"))?,
            gemm_down: load("gemm_down", &format!("krea2_gemm_{width}_resid{tile}"))?,
            rope: load("rope_qknorm", "krea2_rope_qknorm_f16")?,
            attention: load("attention", &metadata.attention_symbol())?,
            attention_transpose: match metadata.fp16_query_tiles {
                2 => Some(load("attention_transpose", "krea2_sage_transpose")?),
                _ => None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line `tests/test_runtime.py` pins for 4115 tokens of int8 weights.
    const PINNED: &str = "5 4115 256 4 4160 8 6144 16448 16 8 1\n";

    #[test]
    fn the_pinned_line_parses_and_agrees_with_the_derived_shapes() {
        let metadata = Metadata::parse(PINNED, 4115).expect("the pinned metadata");
        assert_eq!(metadata.version, 5);
        assert_eq!(metadata.capacity, 4160);
        assert_eq!(metadata.gemm_rows, 256);
        assert_eq!(metadata.width(), "i8");
        assert_eq!(metadata.tile(), "_256");
        assert_eq!(metadata.attention_symbol(), "krea2_attention_gqa_lds_f16_wmma");
    }

    #[test]
    fn a_bundle_for_another_sequence_or_another_rule_is_refused() {
        for (line, why) in [
            ("3 4115 256 4 4160 8 6144 16448 16 8 1", "version 3 expects an fp16 stream"),
            ("5 4096 256 4 4160 8 6144 16448 16 8 1", "compiled for other tokens"),
            ("5 4115 128 4 4160 8 6144 16448 16 8 1", "wrong tile rows"),
            ("5 4115 256 3 4160 8 6144 16448 16 8 1", "wrong raster group"),
            ("5 4115 256 4 4128 8 6144 16448 16 8 1", "wrong capacity"),
            ("5 4115 256 4 4160 4 6144 16448 16 8 1", "wrong wave count"),
            ("5 4115 256 4 4160 8 6144 16512 16 8 1", "int4 pitch with int8 weights"),
            ("5 4115 256 4 4160 8 6144 16448 12 8 1", "no such attention width"),
            ("5 4115 256 4 4160 8 6144 16448 16 6 1", "no such operand width"),
            ("5 4115 256 4 4160 8 6144 16448 16 8 2", "query tiles the host does not use"),
            ("5 4115 256 4 4160 8 6144 16448 16 8", "version 5 without its tile count"),
            ("", "no metadata at all"),
        ] {
            let error = Metadata::parse(line, 4115).unwrap_err();
            assert!(error.message.contains("invalid kernel launch metadata"), "{why}: {error}");
        }
    }

    #[test]
    fn capacity_leaves_headroom_and_whole_key_blocks() {
        assert_eq!(capacity(4115, 1), 4160);
        assert_eq!(capacity(16, 1), 64);
        assert_eq!(capacity(4115, 2), 4160);
        assert_eq!(capacity(2048, 1), 2080);
    }
}
