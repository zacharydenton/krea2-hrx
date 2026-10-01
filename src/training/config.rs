use crate::lora::io;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Which transformer parameters are updated; Qwen and the VAE stay frozen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// Train low-rank adapters with FP32 AdamW.
    #[default]
    Lora,
    /// Train every RAW transformer parameter with bounded-memory AdamW.
    Full,
}

/// Reproducible character-training settings; paths are relative to the config file.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    /// Parameter selection and optimizer representation.
    pub mode: TrainingMode,
    /// Original-basis RAW BF16 safetensors checkpoint.
    pub model: PathBuf,
    /// BF16 text encoder override; otherwise use the existing pinned HF cache.
    pub text_encoder: Option<PathBuf>,
    /// Qwen VAE override; otherwise use the existing pinned HF cache.
    pub vae: Option<PathBuf>,
    /// Image and caption directory, scanned recursively.
    pub dataset: PathBuf,
    /// Character trigger, required in every LoRA caption; ignored in full mode.
    pub trigger: String,
    /// Run directory, holding cache, checkpoints and evaluation output.
    pub output: PathBuf,
    /// Target square area: 512, 768 or 1024.
    pub resolution: usize,
    /// Adapter rank, 1 through 128.
    pub rank: usize,
    /// Adapter targets: `main_blocks` (224) or `all` (264).
    pub targets: crate::lora::Targets,
    /// Adapter alpha; effective scale is alpha/rank.
    pub alpha: f32,
    /// Number of optimizer updates.
    pub steps: usize,
    /// Microbatches per optimizer update; physical batch size is one.
    pub accumulation: usize,
    /// Recompute block activations during backward to reduce memory use.
    /// Disable to retain tapes when the allocation budget permits it.
    pub gradient_checkpointing: bool,
    /// Constant AdamW learning rate.
    pub learning_rate: f32,
    /// AdamW decay on trained parameters.
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
    /// Full-mode model-only snapshot interval; absent disables automatic snapshots.
    pub snapshot_every: Option<usize>,
    /// Number of unpinned model-only snapshots retained separately from resume state.
    pub keep_snapshots: usize,
    /// Seed for initialization, shuffling, posterior sampling, timesteps and noise.
    pub seed: u64,
    /// Maximum planned GPU allocation in GiB.
    pub memory_gib: usize,
    /// Maximum free device scratch retained between operations, in MiB.
    pub scratch_pool_mib: usize,
    /// Prevent model downloads on a cache miss.
    pub offline: bool,
    /// Fixed prompts for Turbo LoRA or RAW full-checkpoint evaluation.
    pub validation_prompts: Vec<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::Lora,
            model: PathBuf::new(),
            text_encoder: None,
            vae: None,
            dataset: PathBuf::new(),
            trigger: String::new(),
            output: PathBuf::new(),
            resolution: 1024,
            rank: 32,
            targets: crate::lora::Targets::MainBlocks,
            alpha: 32.0,
            steps: 1500,
            accumulation: 1,
            gradient_checkpointing: true,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_grad_norm: 1.0,
            save_every: 250,
            keep_checkpoints: 4,
            snapshot_every: None,
            keep_snapshots: 2,
            seed: 37,
            memory_gib: 96,
            scratch_pool_mib: 2048,
            offline: false,
            validation_prompts: Vec::new(),
        }
    }
}

impl TrainConfig {
    /// Full-tuning starting point for 128 GiB Strix Halo; paths remain required.
    pub fn full_preset() -> Self {
        Self {
            mode: TrainingMode::Full,
            resolution: 512,
            learning_rate: 1e-5,
            gradient_checkpointing: false,
            memory_gib: 112,
            scratch_pool_mib: 8192,
            keep_checkpoints: 1,
            snapshot_every: Some(250),
            ..Self::default()
        }
    }
    /// Whether the conditioning towers are trainable and need cached Qwen taps.
    pub(crate) fn trains_conditioning(&self) -> bool {
        self.mode == TrainingMode::Full || self.targets == crate::lora::Targets::All
    }
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
            || (self.mode == TrainingMode::Lora && self.trigger.trim().is_empty())
        {
            return Err(Error::invalid("model, dataset, output and trigger are required"));
        }
        if ![512, 768, 1024].contains(&self.resolution)
            || (self.mode == TrainingMode::Lora && !(1..=128).contains(&self.rank))
            || self.steps == 0
            || self.accumulation == 0
            || self.save_every == 0
            || self.keep_checkpoints == 0
            || self.snapshot_every == Some(0)
            || (self.snapshot_every.is_some()
                && (self.mode != TrainingMode::Full || self.keep_snapshots == 0))
            || self.memory_gib == 0
            || self.memory_gib > 128
            || self.scratch_pool_mib > 16384
        {
            return Err(Error::invalid(
                "invalid resolution, rank, step count, checkpoint interval or memory budget",
            ));
        }
        for value in [
            if self.mode == TrainingMode::Full { 1.0 } else { self.alpha },
            self.learning_rate,
            self.epsilon,
            self.max_grad_norm,
        ] {
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
    fn full_preset_ignores_adapter_settings_and_keeps_lora_defaults() {
        let c = TrainConfig {
            model: "raw.safetensors".into(),
            dataset: "images".into(),
            output: "run".into(),
            rank: 0,
            alpha: 0.0,
            ..TrainConfig::full_preset()
        };
        c.validate().unwrap();
        assert!(c.trains_conditioning());
        assert_eq!(c.resolution, 512);
        assert!(!c.gradient_checkpointing);
        assert_eq!(c.scratch_pool_mib, 8192);
        assert_eq!(c.snapshot_every, Some(250));
        assert_eq!(c.keep_snapshots, 2);
        assert_eq!(TrainConfig::default().mode, TrainingMode::Lora);
    }
    #[test]
    fn rejects_typos_and_nonfinite_or_empty_training_settings() {
        assert_eq!(
            serde_json::from_str::<TrainConfig>(r#"{}"#).unwrap().targets,
            crate::lora::Targets::MainBlocks
        );
        assert_eq!(
            serde_json::from_str::<TrainConfig>(r#"{"targets":"all"}"#).unwrap().targets,
            crate::lora::Targets::All
        );
        assert!(serde_json::from_str::<TrainConfig>(r#"{"targets":"typo"}"#).is_err());
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
        c.snapshot_every = Some(250);
        assert!(c.validate().is_err());
        c.mode = TrainingMode::Full;
        c.validate().unwrap();
        c.snapshot_every = Some(0);
        assert!(c.validate().is_err());
        c.snapshot_every = Some(1);
        c.keep_snapshots = 0;
        assert!(c.validate().is_err());
        c.snapshot_every = None;
        c.validate().unwrap();
        c.beta1 = f32::NAN;
        assert!(c.validate().is_err());
    }
}
