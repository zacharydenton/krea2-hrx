//! Krea 2 image generation on AMD Strix Halo, powered by Loom kernels and HRX.
//!
//! The layering is a chain, and each module depends only on the ones above it:
//! [`numerics`] and [`tokenizer`] are self-contained; [`kernels`] holds the
//! kernel catalogue and its embedded Loom sources; [`checkpoint`] maps
//! safetensors onto the device layout the kernels expect; [`ops`] provides
//! device tensors and the auxiliary operations; [`session`] runs the 28
//! transformer blocks; [`models`] adds the text encoder, the outer graph and
//! the VAE; [`pipeline`] turns a prompt into pixels.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod checkpoint;
pub mod fusion;
pub mod kernels;
pub mod models;
pub mod numerics;
pub mod ops;
pub mod pipeline;
pub mod session;
pub mod tokenizer;
