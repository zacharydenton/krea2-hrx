//! Backend selection for the first text-fusion block's up projection.
#[cfg(feature = "npu")]
pub mod npu;
use crate::ops::{Ops, Tensor, Weight};
use hrx::Stream;
use std::path::Path;
#[cfg(feature = "npu")]
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum FusionBackend {
    /// Use GPU until native NPU latency and full quality qualification passes.
    #[default]
    Auto,
    /// Always execute the existing GPU operation.
    Gpu,
    /// Run the experimental native Loom/XDNA BF16 projection (requires `npu`).
    Npu,
}

#[cfg(feature = "npu")]
struct Cached {
    shape: npu::Shape,
    projection: npu::Projection,
    // Retain the immutable model weight allocation so its identity cannot be reused.
    weight: Tensor,
    bias_identity: Option<(usize, usize)>,
}

pub(crate) struct Fusion {
    backend: FusionBackend,
    #[cfg(feature = "npu")]
    cached: Mutex<Option<Cached>>,
}
impl Fusion {
    pub fn new(_checkpoint: &Path) -> Self {
        Self {
            backend: FusionBackend::Auto,
            #[cfg(feature = "npu")]
            cached: Mutex::new(None),
        }
    }
    pub fn set_backend(&mut self, backend: FusionBackend) {
        self.backend = backend;
    }
    pub fn reason(&self) -> String {
        match self.backend {
            FusionBackend::Gpu => "GPU: explicitly selected",
            FusionBackend::Auto => "GPU: native NPU latency and full quality are not qualified",
            #[cfg(feature = "npu")]
            FusionBackend::Npu => {
                "NPU: experimental native Loom/XDNA BF16 projection, explicitly selected"
            }
            #[cfg(not(feature = "npu"))]
            FusionBackend::Npu => "NPU requires rebuilding with --features npu",
        }
        .into()
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
        capture(stream, x, w, bias)?;
        if self.backend != FusionBackend::Npu {
            return Ok(None);
        }
        #[cfg(not(feature = "npu"))]
        {
            let _ = (stream, ops, x, w, bias);
            Err(crate::ops::Error(self.reason()))
        }
        #[cfg(feature = "npu")]
        {
            use crate::ops::{Error, Layout};
            let shape = npu::Shape {
                m: x.rows(),
                k: x.cols(),
                n: w.shape.first().copied().unwrap_or(0),
            };
            shape.storage_bytes()?;
            if w.shape != [shape.n, shape.k]
                || w.layout() != Layout::RowMajor
                || w.count != shape.n * shape.k
                || bias.is_some_and(|b| b.len() != shape.n * 2)
            {
                return Err(Error("native fusion requires row-major BF16 [M,K], [N,K] weights and optional [N] bias".into()));
            }
            // Model weights and bias are immutable for the lifetime of this Fusion.
            // Hold the lock across execution to prevent concurrent activation writes.
            let mut cached =
                self.cached.lock().map_err(|_| Error("native fusion cache poisoned".into()))?;
            let weight_identity = identity(w.values()?);
            let bias_identity = bias.map(identity);
            let reusable = cached.as_ref().is_some_and(|c| {
                c.shape == shape
                    && identity(c.weight.binding().expect("retained valid weight"))
                        == weight_identity
                    && c.bias_identity == bias_identity
            });
            if !reusable {
                // Evict before allocating so shape changes do not double the bound.
                *cached = None;
                let mut weights = vec![0u16; shape.n * shape.k];
                stream.read_blocking(w.values()?, bytemuck::cast_slice_mut(&mut weights))?;
                let mut biases = bias.map(|_| vec![0u16; shape.n]);
                if let (Some(view), Some(values)) = (bias, biases.as_mut()) {
                    stream.read_blocking(view, bytemuck::cast_slice_mut(values))?;
                }
                let projection = npu::Projection::new(
                    shape,
                    &weights,
                    biases.as_deref(),
                    stream,
                    ops.compiler(),
                )?;
                *cached = Some(Cached {
                    shape,
                    projection,
                    weight: w.tensor(shape.n, shape.k)?,
                    bias_identity,
                });
            }
            let output = ops.tensor(stream, shape.m, shape.n)?;
            cached.as_mut().expect("initialized projection").projection.execute(
                stream,
                x.binding()?,
                output.binding()?,
            )?;
            Ok(Some(output))
        }
    }
}
#[cfg(feature = "npu")]
fn identity(view: hrx::View<'_>) -> (usize, usize) {
    (view.owner() as *const hrx::Buffer as usize, view.offset())
}

/// Opt-in input capture for reproducing a stage benchmark, never a quality baseline.
#[cfg(feature = "npu")]
fn capture(
    stream: &mut Stream,
    x: &Tensor,
    w: &Weight,
    bias: Option<hrx::View<'_>>,
) -> crate::ops::Result<()> {
    use crate::ops::Error;
    let Some(root) = std::env::var_os("KREA2_FUSION_CAPTURE") else {
        return Ok(());
    };
    let result = (|| -> anyhow::Result<()> {
        let shape =
            npu::Shape { m: x.rows(), k: x.cols(), n: w.shape.first().copied().unwrap_or(0) };
        shape.storage_bytes()?;
        let directory = Path::new(&root).join(format!("{}x{}x{}", shape.m, shape.k, shape.n));
        if directory.join("case.json").exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&directory)?;
        let input = x.download(stream)?;
        let mut weights = vec![0u16; w.count];
        stream.read_blocking(w.values()?, bytemuck::cast_slice_mut(&mut weights))?;
        let mut biases = vec![0u16; bias.map_or(0, |b| b.len() / 2)];
        if let Some(b) = bias {
            stream.read_blocking(b, bytemuck::cast_slice_mut(&mut biases))?;
        }
        let mut hashes = serde_json::Map::new();
        for (name, values) in
            [("input.bf16", input), ("weights.bf16", weights), ("bias.bf16", biases)]
        {
            use std::io::Write;
            let bytes = bytemuck::cast_slice::<u16, u8>(&values);
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(name))?
                .write_all(bytes)?;
            hashes.insert(name.into(), hrx::bundle::digest(bytes).into());
        }
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("case.json"))?
            .write_all(
                serde_json::to_string_pretty(
                    &serde_json::json!({"shape":shape,"digests":hashes} ),
                )?
                .as_bytes(),
            )?;
        Ok(())
    })();
    result.map_err(|e| Error(format!("fusion input capture: {e:#}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn auto_remains_gpu_without_new_qualification() {
        let mut fusion = Fusion::new(Path::new("unused"));
        assert!(fusion.reason().starts_with("GPU:"));
        fusion.set_backend(FusionBackend::Gpu);
        assert_eq!(fusion.reason(), "GPU: explicitly selected");
        fusion.set_backend(FusionBackend::Npu);
        #[cfg(feature = "npu")]
        assert!(fusion.reason().contains("experimental native"));
        #[cfg(not(feature = "npu"))]
        assert!(fusion.reason().contains("--features npu"));
    }
}
