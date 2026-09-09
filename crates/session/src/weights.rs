//! The checkpoint on the device: one allocation, filled row by row.
//!
//! The plan says where every row goes ([`krea2_checkpoint::Plan`]); this walks
//! it, assembling each operand at its device pitch through a staging buffer so
//! the 13 GB never lands in host memory at once. Padding stays zero — the
//! kernels never read it.
use std::path::Path;

use hrx::{device, Buffer, DevicePtr};
use krea2_checkpoint::{Checkpoint, Plan};

use crate::{Error, Result};

const STAGING_BYTES: usize = 16 << 20;

/// One checkpoint, resident. Sessions of different sequence lengths share it.
pub struct Weights {
    plan: Plan,
    storage: Buffer,
    pub(crate) stream: std::sync::Arc<hrx::Device>,
}

impl Weights {
    /// Reads the checkpoint and uploads it.
    pub fn load(path: &Path) -> Result<Weights> {
        let file = Checkpoint::open(path)?;
        let plan = Plan::for_checkpoint(&file)?;
        let stream = hrx::Device::current_or_new()?;
        let _scope = stream.enter();
        let storage = stream.allocate(plan.total_bytes)?;
        let mut staging = Vec::new();
        for span in plan.spans.values() {
            let base = storage.ptr().offset(span.device_offset);
            if !span.host.is_empty() {
                device().copy_from_host(base, &span.host)?;
                continue;
            }
            let pitch = if span.rows > 0 { span.device_row_bytes } else { span.row_bytes };
            let width = span.row_bytes;
            let rows_per_chunk = std::cmp::max(1, STAGING_BYTES / pitch);
            staging.clear();
            staging.resize(rows_per_chunk * pitch, 0);
            let mut staged = 0;
            let mut written = 0;
            for segment in &span.segments {
                let tensor = file.get(&segment.tensor)?;
                for row in segment.rows.clone() {
                    let source = &tensor.bytes[row * width..(row + 1) * width];
                    staging[staged * pitch..staged * pitch + width].copy_from_slice(source);
                    staged += 1;
                    if staged == rows_per_chunk {
                        device().copy_from_host(
                            base.offset(written * pitch),
                            &staging[..staged * pitch],
                        )?;
                        written += staged;
                        staged = 0;
                    }
                }
            }
            if staged > 0 {
                device()
                    .copy_from_host(base.offset(written * pitch), &staging[..staged * pitch])?;
            }
        }
        Ok(Weights { plan, storage, stream: stream.clone() })
    }

    pub fn layers(&self) -> usize {
        self.plan.layers
    }

    pub fn bits(&self) -> u32 {
        self.plan.bits
    }

    /// The device address of one span, checked against the size the session
    /// expects so a mismatched checkpoint is caught before any launch.
    pub fn at(&self, name: &str, bytes: usize) -> Result<DevicePtr> {
        let span = self
            .plan
            .spans
            .get(name)
            .ok_or_else(|| Error::failed(format!("missing tensor {name}")))?;
        if span.file_bytes() != bytes {
            return Err(Error::failed(format!(
                "tensor {name} has {} bytes, expected {bytes}",
                span.file_bytes()
            )));
        }
        Ok(self.storage.ptr().offset(span.device_offset))
    }
}
