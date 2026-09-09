//! Model compiler selection; HRX owns hashing, locking and artifact publication.
use crate::{Error, Result};
use std::{collections::HashMap, path::Path, sync::Mutex};

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
    let compiler = hrx::loom::Compiler::resolve(selected.as_deref().map(Path::new))?;
    cache.insert(selected, compiler.clone());
    Ok(compiler)
}

pub use hrx::bundle::digest;
