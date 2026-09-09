//! The compiled kernel bundle a session runs on, and the metadata contract
//! that says it was built by the same rules this host derives.

use hrx::Kernel;
use loom::{shape, Shape};

use crate::{Error, Result, HIDDEN, INTER};

/// Launch dimensions derived from the prepared artifact shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
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

impl From<&Shape> for Metadata {
    fn from(shape: &Shape) -> Self {
        Self {
            tokens: shape.tokens as u32,
            gemm_rows: shape.rows as u32,
            m_group: shape.m_group as u32,
            capacity: shape.capacity as usize,
            attention_waves: shape.attention_waves as u32,
            pitch_hidden: shape::gemm_pitch(HIDDEN, shape.gemm_bits) as u32,
            pitch_inter: shape::gemm_pitch(INTER, shape.gemm_bits) as u32,
            attention_bits: shape.attention_bits as u32,
            gemm_bits: shape.gemm_bits as u32,
            fp16_query_tiles: shape.query_tiles as u32,
        }
    }
}

impl Metadata {
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

/// Every kernel one block needs, loaded from verified artifact bytes.
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
    pub fn load(
        stream: &hrx::Stream,
        bundle: &loom::PreparedBundle,
        metadata: &Metadata,
    ) -> Result<Kernels> {
        Self::load_with(metadata, |stem, symbol| {
            let artifact = bundle
                .artifact(stem)
                .ok_or_else(|| Error::invalid(format!("missing prepared kernel: {stem}")))?;
            if artifact.symbol() != symbol {
                return Err(Error::invalid(format!("unexpected export for {stem}")));
            }
            // Safety: prepare compiles the embedded model sources for this shape.
            unsafe { stream.load_artifact(artifact) }.map_err(Error::from)
        })
    }

    fn load_with(
        metadata: &Metadata,
        load: impl Fn(&str, &str) -> Result<Kernel>,
    ) -> Result<Kernels> {
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

    #[test]
    fn capacity_leaves_headroom_and_whole_key_blocks() {
        assert_eq!(capacity(4115, 1), 4160);
        assert_eq!(capacity(16, 1), 64);
        assert_eq!(capacity(4115, 2), 4160);
        assert_eq!(capacity(2048, 1), 2080);
    }
}
