//! The compiled kernels a session runs on, loaded from a prepared bundle and
//! checked against the exports the host expects for its shape.

use crate::kernels::PreparedBundle;
use hrx::Kernel;

use super::{Error, Result};

/// Every kernel one block needs, loaded from verified artifact bytes.
pub struct Kernels {
    pub(crate) prepare_norm: Kernel,
    pub(crate) prepare_gated: Kernel,
    pub(crate) prepare_swiglu: Kernel,
    pub(crate) gemm_qkvg: Kernel,
    pub(crate) gemm_gu: Kernel,
    pub(crate) gemm_wo: Kernel,
    pub(crate) gemm_down: Kernel,
    pub(crate) rope: Kernel,
    pub(crate) attention: Kernel,
}

impl Kernels {
    /// Loads the bundle's exports, refusing one whose symbol is not the kernel
    /// this host launches under that name.
    pub fn load(stream: &hrx::Stream, bundle: &PreparedBundle) -> Result<Kernels> {
        let load = |stem: &str, symbol: &str| {
            let artifact = bundle
                .artifact(stem)
                .ok_or_else(|| Error::internal(format!("missing prepared kernel: {stem}")))?;
            if artifact.symbol() != symbol {
                return Err(Error::internal(format!("unexpected export for {stem}")));
            }
            // Safety: prepare compiles the embedded model sources for this shape.
            unsafe { stream.load_artifact(artifact) }.map_err(Error::from)
        };
        let attention = format!("krea2_{}", bundle.shape().attention_source());
        Ok(Kernels {
            prepare_norm: load("prepare_norm_i8", "krea2_prepare_norm_i8")?,
            prepare_gated: load("prepare_gated_i8", "krea2_prepare_gated_i8")?,
            prepare_swiglu: load("prepare_plain_i8", "krea2_prepare_plain_i8")?,
            gemm_qkvg: load("gemm_qkvg", "krea2_gemm_i8_256")?,
            gemm_gu: load("gemm_gu", "krea2_gemm_i8_swiglu_256")?,
            gemm_wo: load("gemm_wo", "krea2_gemm_i8_resid_256")?,
            gemm_down: load("gemm_down", "krea2_gemm_i8_resid_256")?,
            rope: load("rope_qknorm", "krea2_rope_qknorm_f16")?,
            attention: load("attention", &attention)?,
        })
    }
}
