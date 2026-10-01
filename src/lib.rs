//! Krea 2 image generation on AMD Strix Halo, powered by Loom kernels and HRX.
//!
//! The layering is a chain, and each module depends only on the ones above it
//! and on the one [`Error`] they all return: [`numerics`] and [`tokenizer`] are
//! self-contained; [`kernels`] holds the
//! kernel catalogue and its embedded Loom sources; [`checkpoint`] maps
//! safetensors onto the device layout the kernels expect; [`ops`] provides
//! device tensors and the auxiliary operations; [`session`] runs the 28
//! transformer blocks; [`models`] adds the text encoder, the outer graph and
//! the VAE; [`pipeline`] turns a prompt into pixels.
//!
//! Most callers need only [`pipeline::Pipeline`]:
//!
//! ```no_run
//! use krea2::pipeline::{Files, Pipeline, Request};
//!
//! # fn main() -> krea2::Result<()> {
//! let files = Files::of("krea2_turbo_int8_convrot".as_ref()).resolve()?;
//! let pipeline = Pipeline::open(files, None)?;
//! let rgb = pipeline.generate(&Request::new("a red fox in the snow"), None)?;
//! assert_eq!(rgb.len(), 1024 * 1024 * 3);
//! # Ok(())
//! # }
//! ```

pub mod checkpoint;
mod context;
mod error;
pub mod kernels;
pub mod lora;
pub mod models;
pub mod numerics;
pub mod ops;
pub mod pipeline;
pub mod session;
pub mod tokenizer;
pub mod training;

pub use error::{Error, Result};
