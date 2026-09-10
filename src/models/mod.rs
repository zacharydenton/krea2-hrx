//! Model discovery, dense weight loading, text encoder and VAE graphs.
//! Transformer block execution is provided by [`crate::session`].
#![deny(unsafe_op_in_unsafe_fn)]

pub mod files;
pub mod graph;
pub mod hub;
pub mod weights;

pub use files::{Files, Request};
pub use graph::{Models, MODULATION_ELEMENTS};
pub use weights::Weights;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<hrx::Error> for Error {
    fn from(error: hrx::Error) -> Self {
        Error(error.to_string())
    }
}

impl From<crate::checkpoint::Error> for Error {
    fn from(error: crate::checkpoint::Error) -> Self {
        Error(error.to_string())
    }
}

impl From<crate::ops::Error> for Error {
    fn from(error: crate::ops::Error) -> Self {
        Error(error.to_string())
    }
}

impl From<crate::tokenizer::Error> for Error {
    fn from(error: crate::tokenizer::Error) -> Self {
        Error(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
