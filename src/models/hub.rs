//! Pinned model downloads through the standard Hugging Face cache.

use hrx::artifacts::hf::{HubFile, Repository, Resolver};
use std::path::PathBuf;

use super::{Error, Result};

/// The upstream quantized checkpoints; repository paths stay inside the HF cache.
pub const REPO: (&str, &str) = ("Comfy-Org", "Krea-2");

/// Immutable upstream snapshot shared by all default model components.
pub const REVISION: &str = "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96";

/// Resolve a cached file, downloading it on a cache miss unless offline.
pub fn file(name: &str, offline: bool) -> Result<PathBuf> {
    Resolver::new(Repository::new(REPO.0, REPO.1).at(REVISION))
        .offline(offline)
        .resolve(&HubFile::new(name))
        .map_err(|error| Error(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_miss_offline_is_an_error_and_not_a_download() {
        let error = file("diffusion_models/not-a-real-file.safetensors", true).unwrap_err();
        assert!(error.0.contains("downloading is off"), "{error}");
    }
}
