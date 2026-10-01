//! Full-transformer training with BF16 parameters and blockwise optimizer state.
pub mod artifacts;
pub(crate) mod checkpoint;
pub mod numerics;
pub(crate) mod parameters;
pub mod spec;
