//! Auxiliary kernel cache, keyed by source and configuration, and the compiler
//! selection it caches behind. HRX owns hashing, locking and artifact
//! publication; this owns which kernel is wanted at which shape.
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use hrx::{Kernel, Stream};

use crate::{sources, Config, Error, Result};

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
    type CompilerKey = (Option<String>, String);
    static COMPILERS: Mutex<Option<HashMap<CompilerKey, hrx::loom::Compiler>>> =
        Mutex::new(None);
    let selected =
        override_path.map(str::to_owned).or_else(|| std::env::var("HRX_LOOM_LIBRARY").ok());
    let mut cache = COMPILERS.lock().map_err(|_| Error("compiler cache poisoned".into()))?;
    let cache = cache.get_or_insert_with(HashMap::new);
    let key = (selected.clone(), target.as_str().to_owned());
    if let Some(compiler) = cache.get(&key) {
        return Ok(compiler.clone());
    }
    // The defaults are tuned for a small consumer: four workers, and room for
    // 64 source modules with arbitrary eviction. One shape needs about fifty
    // modules and guided sampling prepares two shapes, so raise the cache above
    // what a run can hold and let the compiler use the cores this box has.
    //
    // `workers` now sizes HRX's own batch pool rather than one we ran, so
    // KREA2_COMPILE_WORKERS is read once per (library, target) instead of once
    // per prepare -- the compiler behind it is memoized. The cap stays: past
    // eight, the per-specialization cache locks dominate.
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
    let compiler =
        hrx::loom::Compiler::with_options(selected.as_deref().map(Path::new), options)?;
    cache.insert(key, compiler.clone());
    Ok(compiler)
}

pub use hrx::bundle::digest;

type Shapes = BTreeMap<Config, Kernel>;
type Grids = BTreeMap<(u32, u32), Shapes>;

/// Loaded operations owned by one model instance. Cache hits compare the
/// existing configuration directly: no compiler lookup or signature formatting.
#[derive(Default)]
pub struct PreparedKernels {
    loaded: Mutex<HashMap<String, Grids>>,
    compiler: Option<String>,
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
        let mut loaded =
            self.loaded.lock().map_err(|_| Error("operation cache poisoned".into()))?;
        if let Some(kernel) = loaded
            .get(name)
            .and_then(|grids| grids.get(&grid))
            .and_then(|shapes| shapes.get(&config))
        {
            return Ok(kernel.clone());
        }
        let kernel = auxiliary_kernel(stream, name, &config, grid, self.compiler.as_deref())?;
        loaded
            .entry(name.into())
            .or_default()
            .entry(grid)
            .or_default()
            .insert(config, kernel.clone());
        Ok(kernel)
    }
}

pub fn cache_root() -> Result<PathBuf> {
    Ok(hrx::bundle::cache_root()?.join("kernels"))
}

/// The embedded source for an auxiliary kernel, named in the error when there
/// is none. Separate from the compile so the lookup can be tested without a
/// device: naming a kernel now takes a stream, resolving it does not.
fn auxiliary_source(name: &str) -> Result<&'static str> {
    sources::auxiliary(name).ok_or_else(|| Error(format!("no auxiliary kernel named {name}")))
}

/// Compiles `name` for `config` if needed and returns the loaded kernel.
///
/// `grid_x` and `grid_y` join the configuration for every kernel but
/// `sage_transpose`, which is written for any grid.
pub fn auxiliary_kernel(
    stream: &Stream,
    name: &str,
    config: &Config,
    grid: (u32, u32),
    compiler_path: Option<&str>,
) -> Result<Kernel> {
    let mut config = config.clone();
    if name != "sage_transpose" {
        config.insert("grid_x".into(), u64::from(grid.0));
        config.insert("grid_y".into(), u64::from(grid.1));
    }
    let source = auxiliary_source(name)?;
    let compiler = compiler_for_target(compiler_path, stream.target())?;
    let symbol = format!("krea2_{name}");
    let mut request = hrx::loom::Specialization::new(&symbol);
    request.config =
        config.iter().map(|(k, v)| (format!("krea2.{name}.{k}"), v.to_string())).collect();
    request.report = crate::kernel_reports();
    let artifact = compiler.module(source).compile(&request, &cache_root()?)?;
    crate::report(name, &artifact);
    // Safety: the shared compiler produced this export from embedded model source.
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_kernel_is_named_in_the_error() {
        let error = auxiliary_source("no_such_kernel").unwrap_err();
        assert_eq!(error.0, "no auxiliary kernel named no_such_kernel");
    }
}
