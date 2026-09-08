//! Krea 2 end to end: prompt in, RGB out.
//!
//! This is the C++ `native_pipeline.cpp` — the graph in `krea2-models`, the
//! blocks in `krea2-session`, and the sampler that drives them. One image is
//! one `generate`; the models stay resident and the block session is rebuilt
//! only when the sequence length changes, which is when the image size does.
#![deny(unsafe_code)]

pub mod noise;
pub mod profile;
pub mod schedule;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use hrx::device;
use krea2_models::Models;
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::Tensor;
use krea2_session::{Session, Weights};

use crate::profile::Profile;

pub use krea2_models::{hub, Files};

/// The transformer's residual width, and the patch size in pixels.
const WIDTH: usize = 6144;
const PATCH: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

macro_rules! from_error {
    ($($type:path),*) => {$(
        impl From<$type> for Error {
            fn from(error: $type) -> Self {
                Error(error.to_string())
            }
        }
    )*};
}

from_error!(
    hrx::Error,
    loom::Error,
    krea2_models::Error,
    krea2_ops::Error,
    krea2_session::Error
);

pub type Result<T> = std::result::Result<T, Error>;

/// Called after each sampling step with the seconds spent so far. Returning
/// false abandons the image.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, f64) -> bool;

/// What one image asks for. `guidance` and `steps` of `None` take the
/// checkpoint's defaults: Turbo 0 and 8, Raw 3.5 and 52.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub width: usize,
    pub height: usize,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Float32 packed `[h/16 * w/16][64]`, in place of the seeded noise.
    pub initial_latents: Option<&'a [f32]>,
}

impl<'a> Request<'a> {
    pub fn new(prompt: &'a str) -> Request<'a> {
        Request {
            prompt,
            negative_prompt: "",
            width: 1024,
            height: 1024,
            steps: None,
            guidance: None,
            seed: 0,
            initial_latents: None,
        }
    }
}

/// The rope tables for one image geometry, which change only with it.
#[derive(Default)]
struct Rope {
    width: usize,
    height: usize,
    text_tokens: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// The block session for one sequence length, rebuilt when that changes.
struct Blocks {
    session: Session,
    tokens: usize,
}

/// Everything resident: the three models, the block weights, and whichever
/// block session the last image needed.
pub struct Pipeline {
    files: Files,
    compiler: Option<String>,
    cache: PathBuf,
    models: Models,
    /// Calls are serialized: they share the models' buffer pool.
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    weights: Option<Arc<Weights>>,
    blocks: Option<Blocks>,
    rope: Rope,
}

impl Pipeline {
    /// `compiler` of `None` takes `LOOM_COMPILE`, else `loom-compile` on PATH.
    pub fn open(files: Files, compiler: Option<&str>) -> Result<Pipeline> {
        loom::set_compiler(compiler);
        Ok(Pipeline {
            models: Models::open(&files)?,
            files,
            compiler: compiler.map(str::to_string),
            cache: loom::user_cache_directory("blocks-gfx1151-v1")?,
            state: Mutex::new(State::default()),
        })
    }

    /// Turbo: a fixed timestep shift and no guidance. Raw: neither.
    pub fn distilled(&self) -> bool {
        self.files.distilled
    }

    pub fn models(&self) -> &Models {
        &self.models
    }

    /// The prompt's token ids, with no chat template.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self.models.tokenizer.encode(text).map_err(krea2_models::Error::from)?)
    }

    /// The conditioning tokens a prompt encodes to, without running the
    /// encoder: the template's 34-token prefix is not part of them.
    pub fn prompt_tokens(&self, prompt: &str) -> Result<usize> {
        let ids = self.models.tokenizer.prompt(prompt).map_err(krea2_models::Error::from)?;
        Ok(ids.len() - 34)
    }

    /// A prompt encoded through Krea's template: float32 `[tokens][12][2560]`.
    pub fn encode(&self, prompt: &str) -> Result<(usize, Vec<f32>)> {
        let _serialized = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let ids = self.models.tokenizer.prompt(prompt).map_err(krea2_models::Error::from)?;
        let taps = self.models.encode(&ids)?;
        device().synchronize()?;
        Ok((ids.len() - 34, downloaded(&taps)?))
    }

    /// Latents to RGB8 HWC.
    pub fn decode(&self, latents: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        dimensions(width, height)?;
        let _serialized = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let packed = self.upload(latents, width / PATCH * (height / PATCH), 64)?;
        let rgb = self.models.decode(&packed, height, width)?;
        device().synchronize()?;
        Ok(rgb)
    }

    /// One transformer forward: packed latents and tapped text states in,
    /// float32 packed velocity out. `text` is `[text_tokens][12][2560]`.
    pub fn transformer(
        &self,
        text: &[f32],
        text_tokens: usize,
        latents: &[f32],
        width: usize,
        height: usize,
        timestep: f32,
    ) -> Result<Vec<f32>> {
        dimensions(width, height)?;
        if !(1..=512).contains(&text_tokens) || !(0.0..=1.0).contains(&timestep) {
            return Err(Error("invalid transformer arguments".into()));
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let taps = self.upload(text, text_tokens * 12, 2560)?;
        let conditioning = self.models.text_fusion(&taps)?;
        let packed = self.upload(latents, width / PATCH * (height / PATCH), 64)?;
        let velocity =
            self.forward(&mut state, &packed, &conditioning, timestep, width, height)?;
        device().synchronize()?;
        downloaded(&velocity)
    }

    /// One image. `progress` is called after each step and may cancel.
    pub fn generate(
        &self,
        request: &Request,
        mut progress: Option<Progress>,
    ) -> Result<Vec<u8>> {
        let (width, height) = (request.width, request.height);
        dimensions(width, height)?;
        let guidance = request.guidance.unwrap_or(if self.distilled() { 0.0 } else { 3.5 });
        let steps = request.steps.unwrap_or(if self.distilled() { 8 } else { 52 });
        if !(1..=100).contains(&steps)
            || !guidance.is_finite()
            || !(0.0..=100.0).contains(&guidance)
        {
            return Err(Error("invalid generation arguments".into()));
        }
        let image_tokens = width / PATCH * (height / PATCH);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let began = std::time::Instant::now();
        let mut timing = Profile::new("generate");

        let latents = match request.initial_latents {
            Some(values) => self.upload(values, image_tokens, 64)?,
            None => {
                let values = noise::latents(request.seed, image_tokens * 64);
                self.upload(&values, image_tokens, 64)?
            }
        };
        let text = self.conditioning(request.prompt)?;
        let guided = guidance > 0.0;
        let uncond =
            if guided { Some(self.conditioning(request.negative_prompt)?) } else { None };
        timing.mark("encode and text fusion")?;

        // Turbo's shift is fixed; Raw's follows the image's token count.
        let mu = if self.distilled() { 1.15 } else { schedule::dynamic_mu(image_tokens) };
        for step in 0..steps {
            let sigma = schedule::sigma(step, steps, mu);
            let next = schedule::sigma(step + 1, steps, mu);
            let velocity = self.forward(&mut state, &latents, &text, sigma, width, height)?;
            if let Some(uncond) = &uncond {
                let unguided =
                    self.forward(&mut state, &latents, uncond, sigma, width, height)?;
                self.models.ops.guidance(&velocity, &unguided, guidance)?;
            }
            self.models.ops.euler_step(&latents, &velocity, next - sigma)?;
            if let Some(progress) = progress.as_deref_mut() {
                device().synchronize()?;
                if !progress(step + 1, steps, began.elapsed().as_secs_f64()) {
                    return Err(Error("cancelled".into()));
                }
            }
        }
        timing.mark("denoise")?;
        let rgb = self.models.decode(&latents, height, width)?;
        device().synchronize()?;
        timing.mark("VAE decode")?;
        Ok(rgb)
    }

    /// A prompt through the tokenizer, the encoder and the fusion tower.
    fn conditioning(&self, prompt: &str) -> Result<Tensor> {
        let ids = self.models.tokenizer.prompt(prompt).map_err(krea2_models::Error::from)?;
        let taps = self.models.encode(&ids)?;
        Ok(self.models.text_fusion(&taps)?)
    }

    /// Text and image tokens through the 28 blocks and the final layer.
    fn forward(
        &self,
        state: &mut State,
        latents: &Tensor,
        text: &Tensor,
        timestep: f32,
        width: usize,
        height: usize,
    ) -> Result<Tensor> {
        let mut timing = Profile::new("forward");
        let image_tokens = width / PATCH * (height / PATCH);
        let tokens = text.rows() + image_tokens;
        let (embedding, modulation) = self.models.time(timestep)?;
        let image = self.models.image_in(latents)?;

        // The residual stream is the conditioning followed by the image.
        let x = self.models.ops.tensor(tokens, WIDTH)?;
        device().copy_device_to_device(x.ptr(), text.ptr(), text.size() * 2)?;
        device().copy_device_to_device(
            x.ptr().offset(text.size() * 2),
            image.ptr(),
            image.size() * 2,
        )?;

        timing.mark("embeddings")?;
        self.prepare(state, tokens)?;
        timing.mark("prepare session")?;
        let mods = self.models.modulation(&modulation)?;
        self.rope(state, width, height, text.rows());
        let blocks = state.blocks.as_ref().expect("just prepared");
        timing.mark("modulation and rope")?;
        blocks.session.run_device(x.ptr(), mods.ptr(), &state.rope.cos, &state.rope.sin)?;
        timing.mark("blocks")?;

        let output = x.view(image_tokens, WIDTH, text.rows() * WIDTH)?;
        let velocity = self.models.last(&output, &embedding)?;
        timing.mark("final layer")?;
        Ok(velocity)
    }

    /// The block session for `tokens`, built if the last image had another
    /// sequence length. The weights outlive it and are loaded once.
    fn prepare(&self, state: &mut State, tokens: usize) -> Result<()> {
        if state.blocks.as_ref().is_some_and(|blocks| blocks.tokens == tokens) {
            return Ok(());
        }
        let weights = match &state.weights {
            Some(weights) => Arc::clone(weights),
            None => {
                let loaded = Arc::new(Weights::load(&self.files.checkpoint)?);
                state.weights = Some(Arc::clone(&loaded));
                loaded
            }
        };
        // Dropped before the new one is built, so two sessions' scratch buffers
        // are never resident at once.
        state.blocks = None;
        let shape = loom::Shape::from_environment(tokens as i32, weights.bits() as i32)?;
        let bundle = loom::prepare(&self.cache, self.compiler.as_deref(), &shape)?;
        let session = Session::with_weights(weights, &bundle, tokens, 28)?;
        state.blocks = Some(Blocks { session, tokens });
        Ok(())
    }

    /// The rope tables, rebuilt when the geometry changes. Geometry, not just
    /// the token count: rectangular grids have different phases.
    fn rope(&self, state: &mut State, width: usize, height: usize, text_tokens: usize) {
        let rope = &mut state.rope;
        if rope.width == width && rope.height == height && rope.text_tokens == text_tokens {
            return;
        }
        let (columns, rows) = (width / PATCH, height / PATCH);
        let tokens = text_tokens + columns * rows;
        rope.cos = vec![0.0; tokens * 128];
        rope.sin = vec![0.0; tokens * 128];
        for token in 0..tokens {
            let mut offset = 0;
            // Three axes over 128 channels: 32 for the frame, 48 each for the
            // row and the column. Text tokens sit at position zero on all three.
            for axis in 0..3 {
                let width_of_axis = if axis == 0 { 32 } else { 48 };
                let position = match (axis, token >= text_tokens) {
                    (1, true) => (token - text_tokens) / columns,
                    (2, true) => (token - text_tokens) % columns,
                    _ => 0,
                };
                for channel in 0..width_of_axis {
                    let phase = position as f64
                        * 1000f64.powf(-2.0 * (channel / 2) as f64 / width_of_axis as f64);
                    rope.cos[token * 128 + offset + channel] = phase.cos() as f32;
                    rope.sin[token * 128 + offset + channel] = phase.sin() as f32;
                }
                offset += width_of_axis;
            }
        }
        rope.width = width;
        rope.height = height;
        rope.text_tokens = text_tokens;
    }

    /// Float32 in, bf16 on the device, refusing what bf16 cannot hold.
    fn upload(&self, values: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
        if values.len() != rows * cols {
            return Err(Error("wrong input buffer size".into()));
        }
        let mut bits = Vec::with_capacity(values.len());
        for &value in values {
            if !value.is_finite() {
                return Err(Error("nonfinite input".into()));
            }
            let rounded = from_f32(value);
            if !to_f32(rounded).is_finite() {
                return Err(Error("input exceeds bf16 range".into()));
            }
            bits.push(rounded);
        }
        Ok(Tensor::from_slice(self.models.ops.pool(), &bits, rows, cols)?)
    }
}

fn downloaded(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.download()?.into_iter().map(to_f32).collect())
}

fn dimensions(width: usize, height: usize) -> Result<()> {
    if !(64..=2048).contains(&width)
        || !(64..=2048).contains(&height)
        || !width.is_multiple_of(PATCH)
        || !height.is_multiple_of(PATCH)
    {
        return Err(Error("dimensions must be multiples of 16 in 64..2048".into()));
    }
    Ok(())
}

/// Convenience: resolve ComfyUI's files and open a pipeline over them.
pub fn open(checkpoint: &Path, compiler: Option<&str>) -> Result<Pipeline> {
    Pipeline::open(Files::of(checkpoint).resolve()?, compiler)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dimensions_the_patching_cannot_serve_are_refused() {
        assert!(dimensions(1024, 1024).is_ok());
        for (width, height) in [(1020, 1024), (1024, 2064), (32, 64), (64, 63)] {
            let error = dimensions(width, height).unwrap_err();
            assert!(error.0.contains("multiples of 16"), "{width}x{height}: {error}");
        }
    }
}
