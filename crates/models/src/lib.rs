//! Model discovery, dense weight loading, text encoder and VAE graphs.
//! Transformer block execution is provided by `krea2-session`.
#![deny(unsafe_code)]

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
        Error(error.0)
    }
}

impl From<krea2_checkpoint::Error> for Error {
    fn from(error: krea2_checkpoint::Error) -> Self {
        Error(error.0)
    }
}

impl From<krea2_ops::Error> for Error {
    fn from(error: krea2_ops::Error) -> Self {
        Error(error.0)
    }
}

impl From<krea2_tokenizer::Error> for Error {
    fn from(error: krea2_tokenizer::Error) -> Self {
        Error(error.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
