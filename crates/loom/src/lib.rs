//! Compiling and caching Loom kernels for gfx1151.
//!
//! Auxiliary kernels are keyed by source and configuration; transformer bundles
//! also track compiler identity. Both caches verify artifact hashes and serialize
//! publication with process locks. Sources are embedded from `kernels/` at build time.
pub mod blocks;
pub mod cache;
pub mod compile;
pub mod shape;
pub mod sources;

pub use blocks::{prepare, Shape};
pub use cache::{auxiliary_kernel, cache_root};
pub use compile::{compiler, user_cache_directory};

/// Anything the compiler, the cache or the runtime rejects.
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

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error(message)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Named kernel configuration values, sorted for stable cache serialization.
pub type Config = std::collections::BTreeMap<String, u64>;

/// A compiler invocation's `--config` values, which are not all counts.
pub type Settings = std::collections::BTreeMap<String, String>;

/// Convenience for the common `[("tokens", 4115), ...]` literal.
pub fn config<const N: usize>(entries: [(&str, u64); N]) -> Config {
    entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect()
}
