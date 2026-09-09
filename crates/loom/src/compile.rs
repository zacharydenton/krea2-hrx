//! Model compiler selection; HRX owns hashing, locking and artifact publication.
use crate::{Error, Result};
use std::{collections::HashMap, num::NonZeroUsize, path::Path, sync::Mutex};

/// Reuse each selected compiler's pinned identity across model configurations.
/// Failed resolution is retryable and retains HRX's provisioning diagnostic.
pub fn compiler(override_path: Option<&str>) -> Result<hrx::loom::Compiler> {
    static COMPILERS: Mutex<Option<HashMap<Option<String>, hrx::loom::Compiler>>> =
        Mutex::new(None);
    let selected =
        override_path.map(str::to_owned).or_else(|| std::env::var("HRX_LOOM_LIBRARY").ok());
    let mut cache = COMPILERS.lock().map_err(|_| Error("compiler cache poisoned".into()))?;
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(compiler) = cache.get(&selected) {
        return Ok(compiler.clone());
    }
    // The defaults are tuned for a small consumer: four workers, and room for
    // 64 source modules with arbitrary eviction. One shape needs about fifty
    // modules and guided sampling prepares two shapes, so raise the cache above
    // what a run can hold and let the compiler use the cores this box has.
    let options = hrx::loom::CompilerOptions {
        workers: std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        module_cache_capacity: 256,
        ..Default::default()
    };
    let compiler =
        hrx::loom::Compiler::with_options(selected.as_deref().map(Path::new), options)?;
    cache.insert(selected, compiler.clone());
    Ok(compiler)
}

pub use hrx::bundle::digest;
