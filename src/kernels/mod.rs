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

pub use crate::{Error, Result};
pub use blocks::{Attention, PreparedBundle, Shape, prepare_for_target};
pub use cache::{PreparedKernels, compiler};

/// Named kernel configuration values, sorted for stable cache serialization.
/// The names are the kernels' own `config` spellings, so they are literals.
pub type Config = std::collections::BTreeMap<&'static str, u64>;

/// A compiler invocation's `--config` values, which are not all counts.
pub type Settings = std::collections::BTreeMap<String, String>;

/// Convenience for the common `[("tokens", 4115), ...]` literal.
pub fn config<const N: usize>(entries: [(&'static str, u64); N]) -> Config {
    entries.into_iter().collect()
}

/// Whether to ask the compiler for its report on every kernel it builds.
/// Off by default: the reports are large and only wanted when tuning. Read
/// once per process, since every dispatch's cache key includes it.
pub fn kernel_reports() -> bool {
    static REPORTS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("KREA2_KERNEL_REPORT").is_some_and(|v| v == "1")
    });
    *REPORTS
}

/// Whether `KREA2_NATIVE_PROFILE=1` asks for synchronized stage timings on
/// stderr: the pipeline's per-stage times, and each block session's per-kernel
/// breakdown. Read when asked, so a long-lived process can switch it.
pub fn native_profile() -> bool {
    std::env::var_os("KREA2_NATIVE_PROFILE").is_some_and(|value| value == "1")
}

/// What the compiler said about one kernel, on stderr.
///
/// Warnings and errors always print. Notes do not, unless `KREA2_KERNEL_REPORT`
/// asked for them: the backend emits one per spilled value, and a dozen of those
/// per kernel are what hid the register pressure fixed in `b1aeb55` until
/// someone read past them. Tests that assert on spills read `diagnostics()`
/// themselves and are unaffected by this.
pub fn report(name: &str, artifact: &hrx::loom::Artifact) {
    let verbose = kernel_reports();
    for diagnostic in artifact.diagnostics() {
        if !verbose && diagnostic.severity < hrx::loom::Severity::Warning {
            continue;
        }
        // Line zero means the compiler had no source position for it.
        let at = match diagnostic.line {
            0 => String::new(),
            line => format!(" at {line}:{}", diagnostic.column),
        };
        eprintln!(
            "krea2 kernel {name}: {:?} {}{at}: {}",
            diagnostic.severity, diagnostic.code, diagnostic.message
        );
    }
    if let Some(report) = artifact.report() {
        eprintln!("krea2 kernel {name} report: {}", report.json());
    }
}

/// A two-dimensional launch grid in the runtime's `u32` workgroup counts,
/// refusing one that does not fit rather than truncating it.
pub fn grid(stage: &str, x: usize, y: usize) -> Result<[u32; 3]> {
    let fit = |count: usize| {
        u32::try_from(count).map_err(|_| {
            Error::internal(format!("{stage}: {count} workgroups exceed the grid"))
        })
    };
    Ok([fit(x)?, fit(y)?, 1])
}

/// The block size a kernel was compiled for, checked against the one the
/// host is about to launch it with. The runtime validates the block against
/// the compiled size too; disagreeing here names the stage instead of the
/// export, and a mismatch is always a host bug.
pub fn workgroup(stage: &str, kernel: &Kernel, threads: u32) -> Result<[u32; 3]> {
    let compiled = kernel.info().workgroup_size;
    if compiled != [threads, 1, 1] {
        return Err(Error::internal(format!(
            "{stage}: compiled for workgroup {compiled:?} but the host asked for {threads}"
        )));
    }
    Ok(compiled)
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

    /// Appends an index. Kernels take at most four.
    pub fn index(mut self, value: usize) -> Scalars {
        assert!(self.index_count < self.indices.len(), "a kernel takes at most four indices");
        self.indices[self.index_count] = value as u64;
        self.index_count += 1;
        self
    }

    /// Appends a float. Kernels take at most two.
    pub fn float(mut self, value: f32) -> Scalars {
        assert!(self.float_count < self.floats.len(), "a kernel takes at most two floats");
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
            _ => {
                return Err(Error::internal(format!(
                    "{name}: cannot fit scalars in {declared} bytes"
                )));
            }
        };
        let mut constants = Constants::new();
        for index in &self.indices[..self.index_count] {
            match width {
                4 => constants.push(u32::try_from(*index).map_err(|_| {
                    Error::internal(format!("{name}: index {index} exceeds its 32-bit slot"))
                })?),
                8 => constants.push(*index),
                _ => return Err(Error::internal(format!("{name}: odd index width {width}"))),
            }?;
        }
        for value in &self.floats[..self.float_count] {
            constants.push(*value)?;
        }
        Ok(constants)
    }
}
