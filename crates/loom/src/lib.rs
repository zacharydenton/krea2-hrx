//! Compiling and caching Loom kernels for gfx1151.
//!
//! Two caches, as the C++ host had: auxiliary kernels keyed by the hash of
//! source plus configuration, and (later) whole block bundles keyed by a
//! signature over every job. Both publish atomically under a lock file, so
//! several processes can share one cache directory, and both verify a cached
//! artifact's hash before loading it.
//!
//! The kernel sources themselves are embedded by `build.rs` from `kernels/`,
//! which is the only copy — there is no generated header to keep in step.
pub mod blocks;
pub mod cache;
pub mod compile;
pub mod shape;
pub mod sources;

pub use blocks::{prepare, Shape};
pub use cache::{auxiliary_kernel, cache_root};
pub use compile::{compiler, user_cache_directory, Compilation};

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
        Error(error.0)
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error(message)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A kernel's configuration: named `size_t` values, ordered as the C++
/// `std::map` ordered them so cache keys stay byte-identical.
pub type Config = std::collections::BTreeMap<String, u64>;

/// A compiler invocation's `--config` values, which are not all counts.
pub type Settings = std::collections::BTreeMap<String, String>;

/// Convenience for the common `[("tokens", 4115), ...]` literal.
pub fn config<const N: usize>(entries: [(&str, u64); N]) -> Config {
    entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect()
}
