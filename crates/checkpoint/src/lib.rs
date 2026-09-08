//! ComfyUI's int8 ConvRot checkpoint, read as it is.
//!
//! Nothing here is converted or re-quantized: the file is mapped read-only and
//! this crate works out *where every row goes* on the device. The GEMMs want
//! wq, wk, wv and the attention gate as one operand, the MLP gate and up rows
//! interleaved in 16-row groups for the SwiGLU epilogue, and every row at the
//! padded pitch its kernel was compiled for. The result is a [`Plan`]: a list
//! of destinations and the source rows that fill them, which the session
//! uploads without ever materializing the whole thing in host memory.
// Mapping the checkpoint is the one unsafe operation here; everything else is
// slices and arithmetic.
#![deny(unsafe_code)]

pub mod file;
pub mod plan;

pub use file::{Checkpoint, Tensor};
pub use plan::{Plan, Segment, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
