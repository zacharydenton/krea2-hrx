//! Which auxiliary kernel is wanted at which shape.
//!
//! HRX owns compilation, the artifact cache, the loaded-kernel cache and the
//! compiler's own memoization. What is left here is the model's half: the
//! embedded source for a name, the `krea2_` symbol and `krea2.<name>.` config
//! spellings, and the grid that joins the configuration for every kernel but
//! one.
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Mutex;

use hrx::{Kernel, Stream};

use super::{sources, Config, Error, Result};

/// Reuse each selected compiler's pinned identity across model configurations.
/// Failed resolution is retryable and retains HRX's provisioning diagnostic.
pub fn compiler(override_path: Option<&str>) -> Result<hrx::loom::Compiler> {
    compiler_for_target(override_path, &hrx::Target::default())
}

/// Reuse a compiler for the architecture of the stream that will load its output.
pub fn compiler_for_target(
    override_path: Option<&str>,
    target: &hrx::Target,
) -> Result<hrx::loom::Compiler> {
    // The defaults are tuned for a small consumer: four workers, and room for
    // 64 source modules with arbitrary eviction. One shape needs about fifty
    // modules and guided sampling prepares two shapes, so raise the cache above
    // what a run can hold and let the compiler use the cores this box has.
    //
    // `workers` sizes HRX's own batch pool. The compiler behind it is memoized
    // by HRX, so KREA2_COMPILE_WORKERS is read once per (library, target). The
    // cap stays: past eight, the per-specialization cache locks dominate.
    let workers = match std::env::var("KREA2_COMPILE_WORKERS").ok().and_then(|v| v.parse().ok())
    {
        Some(n) => NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN),
        None => std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::MIN)
            .min(NonZeroUsize::new(8).expect("nonzero")),
    };
    let options = hrx::loom::CompilerOptions {
        workers,
        module_cache_capacity: 256,
        target: target.clone(),
    };
    Ok(hrx::loom::Compiler::shared(override_path.map(Path::new), options)?)
}

/// Loaded operations owned by one model instance, released with it.
///
/// The cache itself is HRX's, keyed by the artifact each specialization
/// compiles to. This holds the compiler selection until a stream names the
/// target to build for.
#[derive(Default)]
pub struct PreparedKernels {
    loaded: Mutex<Option<hrx::loom::KeyedKernels<RequestKey>>>,
    compiler: Option<String>,
}

/// The model inputs that determine an auxiliary specialization. HRX indexes
/// these directly; expanding their strings and hashing source is miss-only.
#[derive(Hash, PartialEq, Eq)]
struct RequestKey {
    name: String,
    config: Config,
    grid: (u32, u32),
    report: bool,
}

impl PreparedKernels {
    pub fn new(compiler: Option<&str>) -> Self {
        Self { loaded: Mutex::default(), compiler: compiler.map(str::to_owned) }
    }

    pub fn get(
        &self,
        stream: &Stream,
        name: &str,
        config: Config,
        grid: (u32, u32),
    ) -> Result<Kernel> {
        let source = auxiliary_source(name)?;
        let key = RequestKey {
            name: name.to_owned(),
            config,
            grid: if name == "sage_transpose" { (0, 0) } else { grid },
            report: super::kernel_reports(),
        };
        let mut loaded =
            self.loaded.lock().map_err(|_| Error("operation cache poisoned".into()))?;
        let kernels = match &*loaded {
            Some(kernels) => kernels,
            None => loaded.insert(
                hrx::loom::Kernels::new(compiler_for_target(
                    self.compiler.as_deref(),
                    stream.target(),
                )?)
                .reporting(|artifact| super::report(artifact.symbol(), artifact))
                .keyed(),
            ),
        };
        // Safety: source is embedded and immutable; the key includes every
        // input used to select the source, specialization and report setting.
        Ok(unsafe {
            kernels.get_or_insert_with(stream, key, |key| {
                Ok((source, specialization(&key.name, &key.config, key.grid, key.report)))
            })
        }?)
    }
}

/// The embedded source for an auxiliary kernel, named in the error when there
/// is none. Separate from the compile so the lookup can be tested without a
/// device: naming a kernel now takes a stream, resolving it does not.
fn auxiliary_source(name: &str) -> Result<&'static str> {
    sources::auxiliary(name).ok_or_else(|| Error(format!("no auxiliary kernel named {name}")))
}

/// The export and configuration one auxiliary kernel compiles from.
///
/// `grid_x` and `grid_y` join the configuration for every kernel but
/// `sage_transpose`, which is written for any grid.
fn specialization(
    name: &str,
    config: &Config,
    grid: (u32, u32),
    report: bool,
) -> hrx::loom::Specialization {
    let mut config = config.clone();
    if name != "sage_transpose" {
        config.insert("grid_x".into(), u64::from(grid.0));
        config.insert("grid_y".into(), u64::from(grid.1));
    }
    let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
    request.replace_config(
        config.iter().map(|(k, v)| (format!("krea2.{name}.{k}"), v.to_string())).collect(),
    );
    request.set_report(report);
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_kernel_is_named_in_the_error() {
        let error = auxiliary_source("no_such_kernel").unwrap_err();
        assert_eq!(error.0, "no auxiliary kernel named no_such_kernel");
    }

    #[test]
    fn the_grid_joins_the_configuration_except_where_the_kernel_takes_any() {
        let config = super::super::config([("tokens", 4115)]);
        let joined = specialization("unary_one", &config, (7, 3), false);
        assert_eq!(
            joined.configuration().get("krea2.unary_one.grid_x").map(String::as_str),
            Some("7")
        );
        assert_eq!(joined.symbol(), "krea2_unary_one");
        let any = specialization("sage_transpose", &config, (7, 3), false);
        assert!(!any.configuration().contains_key("krea2.sage_transpose.grid_x"));
    }
}
