//! Backend selection for the first text-fusion block's up projection.
use crate::ops::{Ops, Tensor, Weight};
use hrx::Stream;
use std::path::Path;

const RETIRED: &str = "the legacy Chess NPU fusion pilot is retired in HRX 0.8; use auto or gpu until a native Loom implementation passes latency and quality qualification";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum FusionBackend {
    /// Use the qualified GPU implementation.
    #[default]
    Auto,
    /// Always execute the existing GPU operation.
    Gpu,
    /// Report that the legacy NPU pilot is retired.
    Npu,
}

pub(crate) struct Fusion {
    backend: FusionBackend,
}
impl Fusion {
    pub fn new(_checkpoint: &Path) -> Self {
        Self { backend: FusionBackend::Auto }
    }
    pub fn set_backend(&mut self, backend: FusionBackend) {
        self.backend = backend;
    }
    pub fn reason(&self) -> String {
        match self.backend {
            FusionBackend::Gpu => "GPU: explicitly selected",
            FusionBackend::Auto => "GPU: no qualified native NPU fusion implementation",
            FusionBackend::Npu => RETIRED,
        }
        .into()
    }
    fn validate(&self) -> crate::ops::Result<()> {
        if self.backend == FusionBackend::Npu {
            return Err(crate::ops::Error(RETIRED.into()));
        }
        Ok(())
    }
    pub fn linear(
        &self,
        _stream: &mut Stream,
        _ops: &Ops,
        _x: &Tensor,
        _w: &Weight,
        _bias: Option<hrx::View<'_>>,
    ) -> crate::ops::Result<Option<Tensor>> {
        self.validate()?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_modes_work_and_forced_legacy_npu_fails_before_device_access() {
        let mut fusion = Fusion::new(Path::new("unused"));
        assert!(fusion.validate().is_ok());
        fusion.set_backend(FusionBackend::Gpu);
        assert!(fusion.validate().is_ok());
        fusion.set_backend(FusionBackend::Npu);
        assert!(fusion.validate().unwrap_err().to_string().contains("retired"));
    }
}
