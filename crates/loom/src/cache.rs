//! Auxiliary kernel cache, keyed by source and configuration.
//! Compilation is locked across processes; artifact hashes are verified on load.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use hrx::Kernel;

use crate::compile::{digest, user_cache_directory, Compilation, Lock, Scratch};
use crate::{compiler, sources, Config, Error, Result};

const ROOT: &str = "native-gfx1151-v1";

static LOADED: Mutex<Option<HashMap<String, Kernel>>> = Mutex::new(None);

pub fn cache_root() -> Result<PathBuf> {
    user_cache_directory(ROOT)
}

/// The signature a kernel is cached under: its name, then its configuration in
/// key order, one `key=value` per line.
fn signature(name: &str, config: &Config) -> String {
    let mut text = format!("{name}\n");
    for (key, value) in config {
        text.push_str(&format!("{key}={value}\n"));
    }
    text
}

/// Compiles `name` for `config` if needed and returns the loaded kernel.
///
/// `grid_x` and `grid_y` join the configuration for every kernel but
/// `sage_transpose`, which is written for any grid.
pub fn auxiliary_kernel(
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
    let signature = signature(name, &config);
    let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
    let loaded = loaded.get_or_insert_with(HashMap::new);
    if let Some(kernel) = loaded.get(&signature) {
        return Ok(kernel.clone());
    }
    let source = sources::auxiliary(name)
        .ok_or_else(|| Error(format!("no auxiliary kernel named {name}")))?;
    let key = digest(format!("{source}{signature}").as_bytes());
    let root = cache_root()?;
    let path = root.join(format!("{key}.hsaco"));
    let hash_path = root.join(format!("{key}.sha256"));

    // A populated cache is usable without write access: check before locking.
    if !(path.exists() && hash_path.exists()) {
        let _lock = Lock::acquire(&root, "auxiliary kernel cache")?;
        if !(path.exists() && hash_path.exists()) {
            let staging = root.join(format!("{key}.tmp"));
            let _scratch = Scratch(vec![
                root.join(format!("{name}.loom")),
                root.join(format!("{name}.log")),
                staging.clone(),
            ]);
            let compiler = compiler(compiler_path);
            let settings =
                config.iter().map(|(key, value)| (key.clone(), value.to_string())).collect();
            Compilation { compiler: &compiler, name, source, config: &settings }
                .run(&root, &staging)?;
            let compiled = std::fs::read(&staging)
                .map_err(|e| Error(format!("cannot read {}: {e}", staging.display())))?;
            std::fs::write(&hash_path, digest(&compiled))
                .map_err(|e| Error(format!("cannot write {}: {e}", hash_path.display())))?;
            std::fs::rename(&staging, &path)
                .map_err(|e| Error(format!("cannot publish {}: {e}", path.display())))?;
        }
    }
    let compiled = std::fs::read(&path)
        .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
    let recorded = std::fs::read_to_string(&hash_path)
        .map_err(|e| Error(format!("cannot read {}: {e}", hash_path.display())))?;
    if recorded != digest(&compiled) {
        return Err(Error(format!("corrupt auxiliary kernel: {}", path.display())));
    }
    let kernel = Kernel::load(&path, &format!("krea2_{name}"))?;
    loaded.insert(signature, kernel.clone());
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_signature_is_the_name_then_the_configuration_in_key_order() {
        let config = crate::config([("tokens", 4115), ("cols", 6144)]);
        assert_eq!(signature("unary_one", &config), "unary_one\ncols=6144\ntokens=4115\n");
    }

    #[test]
    fn an_unknown_kernel_is_named_in_the_error() {
        let error =
            auxiliary_kernel("no_such_kernel", &Config::new(), (1, 1), None).unwrap_err();
        assert_eq!(error.0, "no auxiliary kernel named no_such_kernel");
    }
}
