//! Pinned model downloads through the standard Hugging Face hub cache.
//! Cached files are reused before network access; offline mode refuses downloads.
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{HFClientSync, HFError};

use super::{Error, Result};

/// The upstream quantized checkpoints; repository paths stay inside the HF cache.
pub const REPO: (&str, &str) = ("Comfy-Org", "Krea-2");

/// Immutable upstream snapshot shared by all default model components.
pub const REVISION: &str = "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96";

/// The cached file, downloaded into the cache if it is not there yet.
///
/// `offline`, or `HF_HUB_OFFLINE` in the environment, makes a cache miss an
/// error instead of a download.
pub fn file(name: &str, offline: bool) -> Result<PathBuf> {
    let client = client()?;
    let repository = client.model(REPO.0, REPO.1);
    let local = repository
        .download_file()
        .filename(name)
        .revision(REVISION)
        .local_files_only(true)
        .send();
    match local {
        Ok(path) => return Ok(path),
        Err(HFError::LocalEntryNotFound { .. }) => {}
        Err(error) => return Err(named(name, error)),
    }
    if offline || offline_by_environment() {
        return Err(Error(format!(
            "{}/{}/{name} is not in the Hugging Face cache, and downloading is off",
            REPO.0, REPO.1
        )));
    }
    // A silent ten-minute pause on a multi-gigabyte fetch is not feedback.
    repository
        .download_file()
        .filename(name)
        .revision(REVISION)
        .maybe_progress(
            (std::io::stderr().is_terminal()
                && !std::env::var("HF_HUB_DISABLE_PROGRESS_BARS")
                    .is_ok_and(|value| true_value(&value)))
            .then(Bar::default),
        )
        .send()
        .map_err(|error| named(name, error))
}

/// The client owns a tokio runtime thread, so it is made once and shared.
fn client() -> Result<HFClientSync> {
    static CLIENT: OnceLock<std::result::Result<HFClientSync, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| HFClientSync::new().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| Error(format!("cannot reach the Hugging Face hub: {e}")))
}

fn named(name: &str, error: HFError) -> Error {
    Error(format!("cannot fetch {}/{}/{name}: {error}", REPO.0, REPO.1))
}

fn offline_by_environment() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| true_value(&value))
}

// Match huggingface_hub's documented boolean environment-variable semantics.
fn true_value(value: &str) -> bool {
    ["1", "ON", "YES", "TRUE"].iter().any(|truth| value.eq_ignore_ascii_case(truth))
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
        let error = file("diffusion_models/not-a-real-file.safetensors", true).unwrap_err();
        assert!(error.0.contains("is not in the Hugging Face cache"), "{error}");
    }

    #[test]
    fn hf_boolean_values_follow_the_documented_convention() {
        for value in ["1", "ON", "on", "YES", "yes", "TRUE", "true", "True"] {
            assert!(true_value(value), "{value}");
        }
        for value in ["", "0", "false", "FALSE", "off", "no", "anything", " true "] {
            assert!(!true_value(value), "{value}");
        }
    }
}
