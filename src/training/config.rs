use crate::lora::io;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Reproducible character-training settings; paths are relative to the config file.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    /// Original-basis RAW BF16 safetensors checkpoint.
    pub model: PathBuf,
    /// BF16 text encoder override; otherwise use the existing pinned HF cache.
    pub text_encoder: Option<PathBuf>,
    /// Qwen VAE override; otherwise use the existing pinned HF cache.
    pub vae: Option<PathBuf>,
    /// Image and caption directory, scanned recursively.
    pub dataset: PathBuf,
    /// Character trigger, required in every caption.
    pub trigger: String,
    /// Run directory, holding cache, checkpoints and evaluation output.
    pub output: PathBuf,
    /// Target square area: 512, 768 or 1024.
    pub resolution: usize,
    /// Adapter rank, 1 through 128.
    pub rank: usize,
    /// Adapter alpha; effective scale is alpha/rank.
    pub alpha: f32,
    /// Number of optimizer updates.
    pub steps: usize,
    /// Microbatches per optimizer update; physical batch size is one.
    pub accumulation: usize,
    /// Constant AdamW learning rate.
    pub learning_rate: f32,
    /// AdamW decay on adapter parameters only.
    pub weight_decay: f32,
    /// Adam first moment coefficient.
    pub beta1: f32,
    /// Adam second moment coefficient.
    pub beta2: f32,
    /// Adam denominator epsilon.
    pub epsilon: f32,
    /// Global gradient norm cap, after accumulation averaging.
    pub max_grad_norm: f32,
    /// Optimizer updates between checkpoints.
    pub save_every: usize,
    /// Number of complete checkpoints retained.
    pub keep_checkpoints: usize,
    /// Seed for initialization, shuffling, posterior sampling, timesteps and noise.
    pub seed: u64,
    /// Maximum planned GPU allocation in GiB.
    pub memory_gib: usize,
    /// Prevent model downloads on a cache miss.
    pub offline: bool,
    /// Fixed prompts for Turbo checkpoint evaluation.
    pub validation_prompts: Vec<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            text_encoder: None,
            vae: None,
            dataset: PathBuf::new(),
            trigger: String::new(),
            output: PathBuf::new(),
            resolution: 1024,
            rank: 32,
            alpha: 32.0,
            steps: 1500,
            accumulation: 1,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_grad_norm: 1.0,
            save_every: 250,
            keep_checkpoints: 4,
            seed: 37,
            memory_gib: 96,
            offline: false,
            validation_prompts: Vec::new(),
        }
    }
}

impl TrainConfig {
    /// Read and validate a configuration before any GPU work or model download.
    pub fn read(path: &Path) -> Result<Self> {
        let mut config: Self =
            serde_json::from_slice(&std::fs::read(path).map_err(io)?).map_err(io)?;
        config.validate()?;
        let root = std::path::absolute(path)
            .map_err(io)?
            .parent()
            .expect("absolute file path")
            .to_owned();
        for path in [&mut config.model, &mut config.dataset, &mut config.output] {
            if path.is_relative() {
                *path = root.join(&*path);
            }
        }
        for path in [&mut config.text_encoder, &mut config.vae].into_iter().flatten() {
            if path.is_relative() {
                *path = root.join(&*path);
            }
        }
        Ok(config)
    }

    /// Refuse unsupported settings and nonfinite hyperparameters.
    pub fn validate(&self) -> Result<()> {
        if self.model.as_os_str().is_empty()
            || self.dataset.as_os_str().is_empty()
            || self.output.as_os_str().is_empty()
            || self.trigger.trim().is_empty()
        {
            return Err(Error::invalid("model, dataset, output and trigger are required"));
        }
        if ![512, 768, 1024].contains(&self.resolution)
            || !(1..=128).contains(&self.rank)
            || self.steps == 0
            || self.accumulation == 0
            || self.save_every == 0
            || self.keep_checkpoints == 0
            || self.memory_gib == 0
            || self.memory_gib > 128
        {
            return Err(Error::invalid(
                "invalid resolution, rank, step count, checkpoint interval or memory budget",
            ));
        }
        for value in [self.alpha, self.learning_rate, self.epsilon, self.max_grad_norm] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::invalid("training scales must be finite and positive"));
            }
        }
        if !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
        {
            return Err(Error::invalid("invalid AdamW coefficients"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_typos_and_nonfinite_or_empty_training_settings() {
        assert!(serde_json::from_str::<TrainConfig>(r#"{"train_text_encoder":true}"#).is_err());
        assert!(TrainConfig::default().validate().is_err());
        let mut c = TrainConfig {
            model: "raw.safetensors".into(),
            dataset: "images".into(),
            output: "run".into(),
            trigger: "bluej".into(),
            ..Default::default()
        };
        c.validate().unwrap();
        c.beta1 = f32::NAN;
        assert!(c.validate().is_err());
    }
}
