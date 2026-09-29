//! Character LoRA training with explicit Loom backward operations.
mod config;
pub mod dataset;
pub mod model;
pub mod ops;
pub mod optimizer;
pub mod prepare;
mod quantized;
mod trainer;
pub mod vae;
pub use config::TrainConfig;
pub use dataset::PreparedDataset;
pub use trainer::{MemoryEstimate, StepStats, Trainer, sample_latent, sample_sigma};
