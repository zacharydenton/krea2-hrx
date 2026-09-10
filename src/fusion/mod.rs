//! Backend selection for the first text-fusion block's up projection.
use crate::ops::{Ops, Tensor, Weight};
use hrx::Stream;
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

#[cfg(feature = "npu")]
pub mod npu;
#[cfg(feature = "npu")]
pub mod qualification;

/// Selection applies only to the qualified fusion projection; other work stays on GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum FusionBackend {
    /// Use an NPU only when a matching saved qualification passes every gate.
    #[default]
    Auto,
    /// Always execute the existing GPU operation.
    Gpu,
    /// Require a verified NPU artifact for the selected projection.
    Npu,
}

pub(crate) struct Fusion {
    backend: FusionBackend,
    checkpoint: PathBuf,
    #[cfg(feature = "npu")]
    state: Mutex<npu::Cache>,
    reason: Mutex<String>,
}
impl Fusion {
    pub fn new(checkpoint: &Path) -> Self {
        Self {
            backend: FusionBackend::Auto,
            checkpoint: checkpoint.into(),
            #[cfg(feature = "npu")]
            state: Mutex::new(npu::Cache::default()),
            reason: Mutex::new("GPU: no qualified NPU selection".into()),
        }
    }
    pub fn set_backend(&mut self, backend: FusionBackend) {
        self.backend = backend;
    }
    pub fn reason(&self) -> String {
        self.reason.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn linear(
        &self,
        stream: &mut Stream,
        ops: &Ops,
        x: &Tensor,
        w: &Weight,
        bias: Option<hrx::View<'_>>,
    ) -> crate::ops::Result<Option<Tensor>> {
        #[cfg(feature = "npu")]
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| crate::ops::Error("fusion cache poisoned".into()))?;
            let result =
                state.linear((self.backend, &self.checkpoint), stream, ops, x, w, bias);
            *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = state.reason.clone();
            result
        }
        #[cfg(not(feature = "npu"))]
        {
            let _ = (&self.checkpoint, stream, ops, x, w, bias);
            if self.backend == FusionBackend::Npu {
                return Err(crate::ops::Error("NPU support was not compiled in".into()));
            }
            Ok(None)
        }
    }
}
