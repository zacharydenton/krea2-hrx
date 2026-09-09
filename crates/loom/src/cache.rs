//! Auxiliary kernel cache, keyed by source and configuration.
//! Compilation is locked across processes; artifact hashes are verified on load.
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use hrx::Kernel;

use crate::compile::compiler;
use crate::{sources, Config, Error, Result};

static LOADED: Mutex<Option<HashMap<String, Kernel>>> = Mutex::new(None);

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

    pub fn get(&self, name: &str, config: Config, grid: (u32, u32)) -> Result<Kernel> {
        let mut loaded =
            self.loaded.lock().map_err(|_| Error("operation cache poisoned".into()))?;
        if let Some(kernel) = loaded
            .get(name)
            .and_then(|grids| grids.get(&grid))
            .and_then(|shapes| shapes.get(&config))
        {
            return Ok(kernel.clone());
        }
        let kernel = auxiliary_kernel(name, &config, grid, self.compiler.as_deref())?;
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
    let source = sources::auxiliary(name)
        .ok_or_else(|| Error(format!("no auxiliary kernel named {name}")))?;
    let compiler = compiler(compiler_path)?;
    let signature = format!("{}\n{}", compiler.identity(), signature(name, &config));
    {
        let loaded = LOADED.lock().map_err(|_| Error("kernel cache poisoned".into()))?;
        if let Some(kernel) = loaded.as_ref().and_then(|m| m.get(&signature)) {
            return Ok(kernel.clone());
        }
    }
    let symbol = format!("krea2_{name}");
    let mut request = hrx::loom::Specialization::new(&symbol);
    request.config =
        config.iter().map(|(k, v)| (format!("krea2.{name}.{k}"), v.to_string())).collect();
    let path = compiler.module(source).compile(&request, &cache_root()?)?;
    // Safety: the shared compiler produced this export from embedded model source.
    let kernel = unsafe { Kernel::load_artifact(&path)? };
    let mut loaded = LOADED.lock().map_err(|_| Error("kernel cache poisoned".into()))?;
    Ok(loaded.get_or_insert_with(HashMap::new).entry(signature).or_insert(kernel).clone())
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
