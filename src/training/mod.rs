//! RAW LoRA and full-transformer training with explicit Loom backward operations.
pub(crate) mod auxiliary;
mod config;
pub mod dataset;
pub mod full;
mod memory;
pub mod model;
pub mod ops;
pub mod optimizer;
pub mod prepare;
mod quantized;
mod trainer;
pub mod vae;
pub use config::{TrainConfig, TrainingMode};
pub use dataset::PreparedDataset;
pub use trainer::{MemoryEstimate, StepStats, Trainer, sample_latent, sample_sigma};
