//! The checkpoint on the device: one allocation, filled row by row.
//!
//! The plan says where every row goes ([`krea2_checkpoint::Plan`]); this walks
//! it, assembling each operand at its device pitch through a staging buffer so
//! the 13 GB never lands in host memory at once. Padding stays zero — the
//! kernels never read it.
use std::path::Path;

use hrx::{Buffer, Stream, View};
use krea2_checkpoint::{Checkpoint, Plan};

use crate::{Error, Result};

const STAGING_BYTES: usize = 16 << 20;

/// One checkpoint, resident. Sessions of different sequence lengths share it.
pub struct Weights {
    plan: Plan,
    storage: Buffer,
}

/// Where one operand sits in the resident checkpoint. Resolved once, when a
/// session is built, and turned into a binding at each dispatch.
#[derive(Clone, Copy)]
pub struct At {
    offset: usize,
    bytes: usize,
}

impl Weights {
    /// Reads the checkpoint's header and derives the layout, which is every
    /// check a malformed checkpoint fails. No device is touched, so a rejection
    /// costs nothing and can be tested without a GPU.
    pub fn plan(path: &Path) -> Result<(Checkpoint, Plan)> {
        let file = Checkpoint::open(path)?;
        let plan = Plan::for_checkpoint(&file)?;
        Ok((file, plan))
    }

    /// Reads the checkpoint and uploads it.
    pub fn load(stream: &mut Stream, path: &Path) -> Result<Weights> {
        let (file, plan) = Self::plan(path)?;
        Self::upload(stream, &file, plan)
    }

    /// Walks a derived plan, filling one allocation row by row.
    pub fn upload(stream: &mut Stream, file: &Checkpoint, plan: Plan) -> Result<Weights> {
        let storage = stream.allocate(plan.total_bytes)?;
        let mut staging = Vec::new();
        for span in plan.spans.values() {
            let base = storage.slice(span.device_offset, span.device_bytes);
            if !span.host.is_empty() {
                stream.upload(base.slice(0, span.host.len())?, &span.host)?;
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
                        let chunk = base.slice(written * pitch, staged * pitch)?;
                        stream.upload(chunk, &staging[..staged * pitch])?;
                        written += staged;
                        staged = 0;
                    }
                }
            }
            if staged > 0 {
                let chunk = base.slice(written * pitch, staged * pitch)?;
                stream.upload(chunk, &staging[..staged * pitch])?;
            }
        }
        // Not what bounds host memory: Stream::upload copies into runtime
        // staging and stops referencing the caller's slice, and it submits and
        // waits on its own once staging passes 64 MB. Draining here is so a
        // failed upload surfaces at load rather than at the first dispatch.
        stream.synchronize()?;
        Ok(Weights { plan, storage })
    }

    pub fn layers(&self) -> usize {
        self.plan.layers
    }

    pub fn bits(&self) -> u32 {
        self.plan.bits
    }

    /// The device address of one span, checked against the size the session
    /// expects so a mismatched checkpoint is caught before any launch.
    /// The device span of an operand located earlier. `locate` did the checking.
    pub fn view(&self, at: At) -> View<'_> {
        self.storage.slice(at.offset, at.bytes)
    }

    /// Where `name` lives, checked against the size the caller expects. The
    /// span covers the operand's *device* bytes, which for a padded operand is
    /// wider than the file's: that padding is what the kernels stride over.
    pub fn locate(&self, name: &str, bytes: usize) -> Result<At> {
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
        Ok(At { offset: span.device_offset, bytes: span.device_bytes })
    }
}
