//! Model discovery through the standard Hugging Face cache or explicit files.
//! The default text encoder is BF16 and the tokenizer is embedded.
use std::path::{Path, PathBuf};

use super::hub;
use super::{Error, Result};

/// The three files, and which sampler the checkpoint wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Files {
    pub checkpoint: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    /// Optional explicit tokenizer override. `None` uses the bundled tokenizer.
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
    /// Overrides the default BF16 text encoder with an explicit local file.
    pub fn text_encoder(mut self, path: Option<&'a Path>) -> Request<'a> {
        self.text_encoder = path;
        self
    }

    pub fn vae(mut self, path: Option<&'a Path>) -> Request<'a> {
        self.vae = path;
        self
    }

    /// Selects Turbo (`true`) or Raw (`false`); required for custom checkpoints.
    pub fn distilled(mut self, distilled: Option<bool>) -> Request<'a> {
        self.distilled = distilled;
        self
    }

    /// Local files and the Hugging Face cache only: never the network.
    pub fn offline(mut self, offline: bool) -> Request<'a> {
        self.offline = offline;
        self
    }

    /// Known model identifiers have a defined sampler. Custom files need an
    /// explicit choice; renaming a file must not silently change its sampler.
    pub fn is_distilled(&self) -> Result<bool> {
        if let Some(value) = self.distilled {
            return Ok(value);
        }
        match self.checkpoint.to_str() {
            Some("krea2_turbo_int8_convrot" | "krea2_turbo_int8_convrot.safetensors") => {
                Ok(true)
            }
            Some("krea2_raw_int8_convrot" | "krea2_raw_int8_convrot.safetensors") => Ok(false),
            _ => {
                Err(Error("custom checkpoints require an explicit Turbo or Raw sampler".into()))
            }
        }
    }

    pub fn resolve(self) -> Result<Files> {
        let distilled = self.is_distilled()?;
        let checkpoint =
            match std::path::absolute(self.checkpoint).ok().filter(|path| path.is_file()) {
                Some(path) => path,
                None => hub::file(&self.repository_name()?, self.offline)?,
            };
        let text_encoder = match self.text_encoder {
            Some(path) => path.to_path_buf(),
            None => hub::file("text_encoders/qwen3vl_4b_bf16.safetensors", self.offline)?,
        };
        let vae = match self.vae {
            Some(path) => path.to_path_buf(),
            None => hub::file("vae/qwen_image_vae.safetensors", self.offline)?,
        };
        let tokenizer = None;
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
        let request = Files::of(&checkpoint)
            .text_encoder(Some(&encoder))
            .vae(Some(&vae))
            .distilled(Some(false))
            .offline(true);
        let files = request.clone().resolve().unwrap();
        assert_eq!(files.checkpoint, checkpoint);
        assert_eq!(files.text_encoder, encoder);
        assert_eq!(files.vae, vae);
        assert!(!files.distilled);
        assert!(request.distilled(Some(true)).resolve().unwrap().distilled);
    }

    #[test]
    fn custom_names_never_guess_a_sampler() {
        for name in [
            "redraw.safetensors",
            "raw.safetensors",
            "turbo.safetensors",
            "./krea2_raw_int8_convrot.safetensors",
        ] {
            let request = Files::of(Path::new(name));
            assert!(request.is_distilled().is_err(), "{name}");
            assert!(request.clone().distilled(Some(true)).is_distilled().unwrap());
            assert!(!request.distilled(Some(false)).is_distilled().unwrap());
        }
        assert!(Files::of(Path::new("krea2_turbo_int8_convrot")).is_distilled().unwrap());
        assert!(!Files::of(Path::new("krea2_raw_int8_convrot")).is_distilled().unwrap());
    }

    #[test]
    fn missing_explicit_paths_are_errors_instead_of_hub_names() {
        let directory = tempfile::tempdir().unwrap();
        let absolute = directory.path().join("missing.safetensors");
        for path in [absolute.as_path(), Path::new("./missing.safetensors")] {
            let error =
                Files::of(path).distilled(Some(true)).offline(true).resolve().unwrap_err();
            assert!(error.0.contains("cannot read"), "{error}");
        }
    }
}
