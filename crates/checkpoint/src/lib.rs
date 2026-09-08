//! Safetensors parsing and device upload plans for ConvRot block weights.
//!
//! [`Plan`] arranges Q/K/V/gate rows, interleaves MLP gate/up rows in groups of 16,
//! and applies the operand pitches required by the compiled GEMMs. Uploads stream
//! from the mapped checkpoint without materializing a full host copy.
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
