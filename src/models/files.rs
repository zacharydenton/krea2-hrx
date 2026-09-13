//! Model discovery through the standard Hugging Face cache or explicit files.
//! The Qwen tokenizer is resolved separately, with an embedded fallback.
use std::path::{Path, PathBuf};

use super::hub;
use super::{Error, Result};

/// The three files, and which sampler the checkpoint wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Files {
    pub checkpoint: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    /// Qwen's `tokenizer.json`, when the hub could supply it. `None` means the
    /// embedded copy, which is the same bytes.
    pub tokenizer: Option<PathBuf>,
    /// Turbo: a fixed timestep shift and no guidance. Raw: neither.
    pub distilled: bool,
}

impl Files {
    /// Resolves `checkpoint`, which is either a path to a file or the name of
    /// one in ComfyUI's Krea 2 repository (`krea2_turbo_int8_convrot`).
    pub fn of(checkpoint: &Path) -> Request<'_> {
        Request { checkpoint, text_encoder: None, vae: None, distilled: None, offline: false }
    }
}

/// What to resolve, and where it may be looked for.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    checkpoint: &'a Path,
    text_encoder: Option<&'a Path>,
    vae: Option<&'a Path>,
    distilled: Option<bool>,
    offline: bool,
}

impl<'a> Request<'a> {
    /// Overrides the text encoder, which is otherwise the bf16 one where both
    /// it and the fp8 one are present.
    pub fn text_encoder(mut self, path: Option<&'a Path>) -> Request<'a> {
        self.text_encoder = path;
        self
    }

    pub fn vae(mut self, path: Option<&'a Path>) -> Request<'a> {
        self.vae = path;
        self
    }

    /// Overrides the sampler, which the checkpoint's file name otherwise
    /// decides: "raw" anywhere in it means Raw.
    pub fn distilled(mut self, distilled: Option<bool>) -> Request<'a> {
        self.distilled = distilled;
        self
    }

    /// Local files and the Hugging Face cache only: never the network.
    pub fn offline(mut self, offline: bool) -> Request<'a> {
        self.offline = offline;
        self
    }

    pub fn resolve(self) -> Result<Files> {
        let checkpoint =
            match std::path::absolute(self.checkpoint).ok().filter(|path| path.is_file()) {
                Some(path) => path,
                None => hub::file(hub::REPO, &self.repository_name()?, self.offline)?,
            };
        let text_encoder = match self.text_encoder {
            Some(path) => path.to_path_buf(),
            None => self.find(&[
                "text_encoders/qwen3vl_4b_bf16.safetensors",
                "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            ])?,
        };
        let vae = match self.vae {
            Some(path) => path.to_path_buf(),
            None => self.find(&["vae/qwen_image_vae.safetensors"])?,
        };
        let distilled = self.distilled.unwrap_or_else(|| {
            !checkpoint
                .file_name()
                .map(|name| name.to_string_lossy().to_lowercase().contains("raw"))
                .unwrap_or(false)
        });
        // Fall back to the embedded tokenizer when the hub is unavailable.
        let tokenizer = hub::file(hub::TOKENIZER_REPO, "tokenizer.json", self.offline).ok();
        Ok(Files { checkpoint, text_encoder, vae, tokenizer, distilled })
    }

    /// The checkpoint as a path in the repository: a bare name is a diffusion
    /// model, and a name given without one gets the extension.
    fn repository_name(&self) -> Result<String> {
        let name = self
            .checkpoint
            .to_str()
            .ok_or_else(|| Error(format!("{} is not a name", self.checkpoint.display())))?;
        if name.is_empty() || name.contains('/') || name.contains("..") {
            return Err(Error(format!("cannot read {name}")));
        }
        let name = match name.ends_with(".safetensors") {
            true => name.to_string(),
            false => format!("{name}.safetensors"),
        };
        Ok(format!("diffusion_models/{name}"))
    }

    /// Reuse the first cached variant, otherwise download the preferred one.
    fn find(&self, names: &[&str]) -> Result<PathBuf> {
        if let Some(path) = names.iter().find_map(|name| hub::cached(hub::REPO, name)) {
            return Ok(path);
        }
        let wanted = names.first().ok_or_else(|| Error("nothing to look for".into()))?;
        hub::file(hub::REPO, wanted, self.offline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_files_need_no_directory_layout_and_preserve_sampler_selection() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("custom_raw.safetensors");
        let encoder = directory.path().join("encoder.safetensors");
        let vae = directory.path().join("decoder.safetensors");
        for path in [&checkpoint, &encoder, &vae] {
            std::fs::write(path, b"").unwrap();
        }
        let request =
            Files::of(&checkpoint).text_encoder(Some(&encoder)).vae(Some(&vae)).offline(true);
        let files = request.clone().resolve().unwrap();
        assert_eq!(files.checkpoint, checkpoint);
        assert_eq!(files.text_encoder, encoder);
        assert_eq!(files.vae, vae);
        assert!(!files.distilled);
        assert!(request.distilled(Some(true)).resolve().unwrap().distilled);
    }

    #[test]
    fn missing_explicit_paths_are_errors_instead_of_hub_names() {
        let directory = tempfile::tempdir().unwrap();
        let absolute = directory.path().join("missing.safetensors");
        for path in [absolute.as_path(), Path::new("./missing.safetensors")] {
            let error = Files::of(path).offline(true).resolve().unwrap_err();
            assert!(error.0.contains("cannot read"), "{error}");
        }
    }
}
