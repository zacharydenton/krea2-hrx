//! Model compiler selection; HRX owns hashing, locking and artifact publication.
use crate::{Error, Result};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
};

/// Reuse each selected compiler's pinned identity across model configurations.
/// Failed resolution is retryable and retains HRX's provisioning diagnostic.
pub fn compiler(override_path: Option<&str>) -> Result<hrx::loom::Compiler> {
    static COMPILERS: Mutex<Option<HashMap<Option<String>, hrx::loom::Compiler>>> =
        Mutex::new(None);
    let selected =
        override_path.map(str::to_owned).or_else(|| std::env::var("LOOM_COMPILE").ok());
    let mut cache = COMPILERS.lock().map_err(|_| Error("compiler cache poisoned".into()))?;
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(compiler) = cache.get(&selected) {
        return Ok(compiler.clone());
    }
    let compiler = hrx::loom::Compiler::resolve(selected.as_deref().map(Path::new))?;
    cache.insert(selected, compiler.clone());
    Ok(compiler)
}

/// Model metadata lives under HRX's configured per-user cache.
pub fn user_cache_directory(leaf: &str) -> Result<PathBuf> {
    let path = hrx::bundle::cache_root()?.join("krea2").join(leaf);
    std::fs::create_dir_all(&path).map_err(|e| Error(e.to_string()))?;
    Ok(path)
}

pub use hrx::bundle::digest;
