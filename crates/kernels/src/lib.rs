//! The Krea 2 kernel catalogue: which Loom kernel the model wants at which
//! shape, and the source it is compiled from.
//!
//! Compilation itself belongs to `hrx::loom`; this crate chooses the
//! specialization, keys the cache on it and holds the tiling rules the shape
//! follows. Sources are embedded from `kernels/` at build time.
pub mod blocks;
pub mod cache;
pub mod shape;
pub mod sources;

use hrx::{Constants, Kernel};

pub use blocks::{prepare, prepare_for_target, PreparedBundle, Shape};
pub use cache::{auxiliary_kernel, cache_root, compiler, digest};

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

/// Whether to ask the compiler for its report on every kernel it builds.
/// Off by default: the reports are large and only wanted when tuning.
pub fn kernel_reports() -> bool {
    std::env::var_os("KREA2_KERNEL_REPORT").is_some_and(|v| v == "1")
}

/// A kernel's scalar arguments: Loom packs the leading indices, then the floats.
///
/// Loom picks each index's width by range analysis, so the host cannot know it
/// from the source. The export declares the total constant size, which is what
/// recovers the width here — 4 or 8 bytes per index once the floats are taken off.
#[derive(Clone, Copy, Default)]
pub struct Scalars {
    indices: [u64; 4],
    index_count: usize,
    floats: [f32; 2],
    float_count: usize,
}

impl Scalars {
    pub fn new() -> Scalars {
        Scalars::default()
    }

    pub fn index(mut self, value: usize) -> Scalars {
        self.indices[self.index_count] = value as u64;
        self.index_count += 1;
        self
    }

    pub fn float(mut self, value: f32) -> Scalars {
        self.floats[self.float_count] = value;
        self.float_count += 1;
        self
    }

    pub fn pack(&self, name: &str, kernel: &Kernel) -> Result<Constants> {
        let declared = kernel.info().constant_byte_length as usize;
        let floats = self.float_count * 4;
        let width = match declared.checked_sub(floats) {
            Some(0) if self.index_count == 0 => 0,
            Some(rest) if self.index_count > 0 && rest % self.index_count == 0 => {
                rest / self.index_count
            }
            _ => return Err(Error(format!("{name}: cannot fit scalars in {declared} bytes"))),
        };
        let mut constants = Constants::new();
        for index in &self.indices[..self.index_count] {
            match width {
                4 => constants.push(*index as u32),
                8 => constants.push(*index),
                _ => return Err(Error(format!("{name}: odd index width {width}"))),
            }
            .map_err(|e| Error(e.to_string()))?;
        }
        for value in &self.floats[..self.float_count] {
            constants.push(*value).map_err(|e| Error(e.to_string()))?;
        }
        Ok(constants)
    }
}
