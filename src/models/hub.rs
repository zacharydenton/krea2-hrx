//! Model and tokenizer downloads through the Hugging Face hub cache.
//! Cached files are reused before network access; offline mode refuses downloads.
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{HFClientSync, HFError};

use super::{Error, Result};

/// ComfyUI's Krea 2 repository, whose layout is the models directory's:
/// `diffusion_models/`, `text_encoders/`, `vae/`.
pub const REPO: (&str, &str) = ("Comfy-Org", "Krea-2");

/// The text encoder's own repository, which is where its tokenizer lives.
pub const TOKENIZER_REPO: (&str, &str) = ("Qwen", "Qwen3-VL-4B-Instruct");

/// The path a file already has in the cache, without touching the network.
pub fn cached(repo: (&str, &str), name: &str) -> Option<PathBuf> {
    client()
        .ok()?
        .model(repo.0, repo.1)
        .download_file()
        .filename(name)
        .local_files_only(true)
        .send()
        .ok()
}

/// The cached file, downloaded into the cache if it is not there yet.
///
/// `offline`, or `HF_HUB_OFFLINE` in the environment, makes a cache miss an
/// error instead of a download.
pub fn file(repo: (&str, &str), name: &str, offline: bool) -> Result<PathBuf> {
    let client = client()?;
    let repository = client.model(repo.0, repo.1);
    let local = repository.download_file().filename(name).local_files_only(true).send();
    match local {
        Ok(path) => return Ok(path),
        Err(HFError::LocalEntryNotFound { .. }) => {}
        Err(error) => return Err(named(repo, name, error)),
    }
    if offline || offline_by_environment() {
        return Err(Error(format!(
            "{}/{}/{name} is not in the Hugging Face cache, and downloading is off",
            repo.0, repo.1
        )));
    }
    // A silent ten-minute pause on a multi-gigabyte fetch is not feedback.
    repository
        .download_file()
        .filename(name)
        .maybe_progress(std::io::stderr().is_terminal().then(Bar::default))
        .send()
        .map_err(|error| named(repo, name, error))
}

/// The client owns a tokio runtime thread, so it is made once and shared.
fn client() -> Result<HFClientSync> {
    static CLIENT: OnceLock<std::result::Result<HFClientSync, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| HFClientSync::new().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| Error(format!("cannot reach the Hugging Face hub: {e}")))
}

fn named(repo: (&str, &str), name: &str, error: HFError) -> Error {
    Error(format!("cannot fetch {}/{}/{name}: {error}", repo.0, repo.1))
}

fn offline_by_environment() -> bool {
    std::env::var_os("HF_HUB_OFFLINE").is_some_and(|value| value != "0" && !value.is_empty())
}

/// A one-line percentage on stderr, redrawn in place. These files are measured
/// in gigabytes; a silent ten-minute pause is not acceptable feedback.
#[derive(Default)]
struct Bar {
    total: Mutex<u64>,
}

impl ProgressHandler for Bar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let mut stderr = std::io::stderr().lock();
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                *self.total.lock().unwrap_or_else(|e| e.into_inner()) = *total_bytes;
            }
            DownloadEvent::AggregateProgress { bytes_completed, total_bytes, .. } => {
                let total = match total_bytes {
                    0 => *self.total.lock().unwrap_or_else(|e| e.into_inner()),
                    other => *other,
                };
                if total > 0 {
                    let _ = write!(
                        stderr,
                        "\rfetching {:5.1}% of {:.2} GB",
                        100.0 * *bytes_completed as f64 / total as f64,
                        total as f64 / 1e9
                    );
                    let _ = stderr.flush();
                }
            }
            DownloadEvent::Complete => {
                let _ = writeln!(stderr, "\rfetched                    ");
            }
            DownloadEvent::Progress { .. } => {}
        }
    }
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

    /// The prompt encoding is a contract, so the hub's tokenizer and the one
    /// compiled in must be the same bytes. Skipped when the hub has not been
    /// asked for it yet, so an offline box does not fail on a missing file.
    #[test]
    fn the_hub_tokenizer_is_the_embedded_one() {
        let Some(path) = cached(TOKENIZER_REPO, "tokenizer.json") else {
            return;
        };
        let bytes = std::fs::read(&path).expect("the cached tokenizer");
        assert_eq!(
            bytes.len(),
            crate::tokenizer::EMBEDDED.len(),
            "{}/{} tokenizer.json is no longer the embedded one",
            TOKENIZER_REPO.0,
            TOKENIZER_REPO.1
        );
        assert!(bytes == crate::tokenizer::EMBEDDED, "the hub tokenizer changed under us");
    }
}
