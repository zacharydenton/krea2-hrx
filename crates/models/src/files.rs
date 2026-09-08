//! Finding ComfyUI's three files.
//!
//! A local models directory first — `<models>/diffusion_models/<checkpoint>`
//! beside `<models>/text_encoders/qwen3vl_4b_{bf16,fp8_scaled}.safetensors` and
//! `<models>/vae/qwen_image_vae.safetensors` — and then the Hugging Face hub,
//! whose Krea 2 repository has that same layout. So a checkpoint can be named
//! by path or just by name, and a file another tool already downloaded is used
//! where it lies.
use std::path::{Path, PathBuf};

use crate::hub;
use crate::{Error, Result};

/// The three files, and which sampler the checkpoint wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Files {
    pub checkpoint: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
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
        let local = std::path::absolute(self.checkpoint).ok().filter(|path| path.is_file());
        // A checkpoint that is not a file on disk names one in the repository,
        // whose diffusion models are all .safetensors. Naming a model is how
        // you ask for it to be fetched: giving a path means that directory, so
        // a file missing from it is an error and not an eight-gigabyte
        // download nobody asked for.
        let (checkpoint, root) = match local {
            Some(path) => {
                let root = path.parent().and_then(Path::parent).map(Path::to_path_buf);
                (path, root)
            }
            None => (hub::file(hub::REPO, &self.repository_name()?, self.offline)?, None),
        };
        let named = root.is_none();
        let text_encoder = match self.text_encoder {
            Some(path) => path.to_path_buf(),
            None => self.find(
                root.as_deref(),
                &[
                    "text_encoders/qwen3vl_4b_bf16.safetensors",
                    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                ],
                named,
                "text encoder",
                "<models>/text_encoders/qwen3vl_4b_{bf16,fp8_scaled}.safetensors",
            )?,
        };
        let vae = match self.vae {
            Some(path) => path.to_path_buf(),
            None => self.find(
                root.as_deref(),
                &["vae/qwen_image_vae.safetensors"],
                named,
                "VAE",
                "<models>/vae/qwen_image_vae.safetensors",
            )?,
        };
        let distilled = self.distilled.unwrap_or_else(|| {
            !checkpoint
                .file_name()
                .map(|name| name.to_string_lossy().to_lowercase().contains("raw"))
                .unwrap_or(false)
        });
        Ok(Files { checkpoint, text_encoder, vae, distilled })
    }

    /// The checkpoint as a path in the repository: a bare name is a diffusion
    /// model, and a name given without one gets the extension.
    fn repository_name(&self) -> Result<String> {
        let name = self
            .checkpoint
            .to_str()
            .ok_or_else(|| Error(format!("{} is not a name", self.checkpoint.display())))?;
        if name.is_empty() || name.starts_with('/') || name.contains("..") {
            return Err(Error(format!("cannot read {name}")));
        }
        let name = match name.ends_with(".safetensors") {
            true => name.to_string(),
            false => format!("{name}.safetensors"),
        };
        Ok(match name.contains('/') {
            true => name,
            false => format!("diffusion_models/{name}"),
        })
    }

    /// The first of `names` beside the checkpoint or in the Hugging Face
    /// cache. `fetch` allows the network for a file neither one has.
    fn find(
        &self,
        root: Option<&Path>,
        names: &[&str],
        fetch: bool,
        what: &str,
        expected: &str,
    ) -> Result<PathBuf> {
        if let Some(root) = root {
            if let Some(path) = names.iter().map(|name| root.join(name)).find(|p| p.is_file()) {
                return Ok(path);
            }
        }
        if let Some(path) = names.iter().find_map(|name| hub::cached(hub::REPO, name)) {
            return Ok(path);
        }
        let wanted = names.first().ok_or_else(|| Error("nothing to look for".into()))?;
        if fetch && !self.offline {
            return hub::file(hub::REPO, wanted, false);
        }
        Err(Error(format!(
            "the {what} was not found beside {} or in the Hugging Face cache \
             (expected {expected}; pass it explicitly, or name a checkpoint to fetch \
             the set from {})",
            self.checkpoint.display(),
            hub::REPO
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A models directory with empty files, enough for path resolution.
    fn layout(name: &str, text_encoder: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("krea2-files-{}-{name}", std::process::id()));
        for directory in ["diffusion_models", "text_encoders", "vae"] {
            std::fs::create_dir_all(root.join(directory)).expect("the layout");
        }
        std::fs::write(root.join("diffusion_models").join(name), b"").expect("the checkpoint");
        std::fs::write(root.join("text_encoders").join(text_encoder), b"")
            .expect("the encoder");
        std::fs::write(root.join("vae/qwen_image_vae.safetensors"), b"").expect("the vae");
        root
    }

    #[test]
    fn the_siblings_are_found_and_the_name_says_which_sampler() {
        let root =
            layout("krea2_turbo_int8_convrot.safetensors", "qwen3vl_4b_bf16.safetensors");
        let checkpoint = root.join("diffusion_models/krea2_turbo_int8_convrot.safetensors");
        let files = Files::of(&checkpoint).resolve().expect("resolution");
        assert_eq!(files.text_encoder, root.join("text_encoders/qwen3vl_4b_bf16.safetensors"));
        assert_eq!(files.vae, root.join("vae/qwen_image_vae.safetensors"));
        assert!(files.distilled, "a turbo checkpoint is distilled");

        let raw = root.join("diffusion_models/krea2_raw_int8_convrot.safetensors");
        std::fs::write(&raw, b"").expect("a raw checkpoint");
        assert!(!Files::of(&raw).resolve().expect("resolution").distilled);
        // An explicit choice wins over the name.
        assert!(Files::of(&raw).distilled(Some(true)).resolve().expect("resolution").distilled);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn the_fp8_encoder_is_the_fallback_and_a_missing_one_says_where_it_looked() {
        let root = layout("krea2_turbo.safetensors", "qwen3vl_4b_fp8_scaled.safetensors");
        let checkpoint = root.join("diffusion_models/krea2_turbo.safetensors");
        let files = Files::of(&checkpoint).resolve().expect("resolution");
        assert!(files.text_encoder.ends_with("qwen3vl_4b_fp8_scaled.safetensors"));

        // A path was given, so a missing sibling is an error and never a
        // download: the message names what is expected where.
        std::fs::remove_file(&files.text_encoder).expect("removing the encoder");
        let error = Files::of(&checkpoint).resolve().unwrap_err();
        assert!(error.0.contains("the text encoder was not found beside"), "{error}");
        assert!(error.0.contains("text_encoders/qwen3vl_4b_{bf16,fp8_scaled}"), "{error}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_checkpoint_that_is_not_a_file_is_looked_for_in_the_repository() {
        // Offline and uncached, so this reports the miss rather than fetching.
        let error = Files::of(Path::new("krea2_turbo_int8_convrot"))
            .offline(true)
            .resolve()
            .unwrap_err();
        assert!(
            error.0.contains(
                "Comfy-Org/Krea-2/diffusion_models/krea2_turbo_int8_convrot.safetensors"
            ),
            "{error}"
        );
    }
}
