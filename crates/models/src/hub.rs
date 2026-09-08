//! ComfyUI's Krea 2 files from the Hugging Face hub.
//!
//! The hub keeps the same cache Python's `huggingface_hub` keeps — `$HF_HOME`
//! or `~/.cache/huggingface`, blobs shared by hash — so a file some other tool
//! already pulled is used where it lies, with no copy and no second download.
//! The cache is consulted before the network, which is what makes a box with
//! no route to the internet and a warm cache behave like an online one.
use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;

use crate::{Error, Result};

/// ComfyUI's Krea 2 repository, whose layout is the models directory's:
/// `diffusion_models/`, `text_encoders/`, `vae/`.
pub const REPO: &str = "Comfy-Org/Krea-2";

/// The path a file already has in the cache, without touching the network.
pub fn cached(repo: &str, name: &str) -> Option<PathBuf> {
    Cache::from_env().model(repo.to_string()).get(name)
}

/// The cached file, downloaded into the cache if it is not there yet.
///
/// `offline`, or `HF_HUB_OFFLINE` in the environment, makes a cache miss an
/// error instead of a download.
pub fn file(repo: &str, name: &str, offline: bool) -> Result<PathBuf> {
    if let Some(path) = cached(repo, name) {
        return Ok(path);
    }
    if offline || offline_by_environment() {
        return Err(Error(format!(
            "{repo}/{name} is not in the Hugging Face cache, and downloading is off"
        )));
    }
    let api = ApiBuilder::from_env()
        .with_progress(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .build()
        .map_err(|e| Error(format!("cannot reach the Hugging Face hub: {e}")))?;
    api.model(repo.to_string())
        .get(name)
        .map_err(|e| Error(format!("cannot fetch {repo}/{name}: {e}")))
}

fn offline_by_environment() -> bool {
    std::env::var_os("HF_HUB_OFFLINE").is_some_and(|value| value != "0" && !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_miss_offline_is_an_error_and_not_a_download() {
        let error =
            file(REPO, "diffusion_models/not-a-real-file.safetensors", true).unwrap_err();
        assert!(error.0.contains("is not in the Hugging Face cache"), "{error}");
    }
}
