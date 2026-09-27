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

pub use file::{Checkpoint, DType, Tensor};
pub use plan::{Plan, Segment, Span};

pub use crate::{Error, Result};
