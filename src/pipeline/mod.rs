//! Krea 2 generation from prompts to RGB.
//! Models remain resident across calls. Two recent block shapes are retained so
//! guided generation can alternate conditioning lengths without rebuilding.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod noise;
pub mod profile;
pub mod schedule;
mod shared;

use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::models::Models;
use crate::numerics::{from_f32, to_f32};
use crate::ops::Tensor;
use crate::session::{Session, Weights};
use hrx::inference::ModelContext;
use hrx::Stream;
use shared::{native_stream, BlockCache, BlockShape, Blocks, Bridge};

use self::profile::Profile;

pub use crate::models::{hub, Files};

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
    crate::kernels::Error,
    crate::models::Error,
    crate::ops::Error,
    crate::session::Error
);

pub type Result<T> = std::result::Result<T, Error>;

/// Called after each sampling step with the seconds spent so far. Returning
/// false abandons the image.
///
/// It runs on the calling thread while the pipeline's lock is held, so it must
/// not call back into the same pipeline: `encode`, `decode`, `transformer` and
/// `generate` would all deadlock on a lock this closure is already inside.
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

/// Everything resident: the three models, the block weights, and whichever
/// two most recently used block sessions.
pub struct Pipeline {
    context: ModelContext,
    bridge: Arc<Bridge>,
    files: Files,
    compiler: Option<String>,
    models: Models,
    /// Calls are serialized: they share the models' buffer pool.
    state: Mutex<State>,
}

/// The stream lives here rather than beside the buffers: it is `!Sync` and
/// almost every operation on it needs `&mut`, and this is the lock that already
/// serialized calls because they share the models' pool.
struct State {
    stream: Stream,
    weights: Option<Arc<Weights>>,
    blocks: BlockCache,
}

/// Optional execution policy. Auto uses only saved, passing NPU qualifications.
#[derive(Clone, Debug, Default)]
pub struct PipelineOptions {
    pub fusion_backend: crate::fusion::FusionBackend,
}

impl Pipeline {
    /// `compiler` of `None` takes `HRX_LOOM_LIBRARY` or the pinned bundle.
    pub fn open(files: Files, compiler: Option<&str>) -> Result<Pipeline> {
        Self::with_options(files, compiler, PipelineOptions::default())
    }

    /// Open with an explicit policy for the fusion projection.
    pub fn with_options(
        files: Files,
        compiler: Option<&str>,
        options: PipelineOptions,
    ) -> Result<Pipeline> {
        Self::open_in(files, &ModelContext::new(Default::default())?, compiler, options)
    }

    /// Share block execution, tensors and dependency tracking with other clients.
    /// Auxiliary model operations keep their private native stream and are
    /// drained at the explicit device-copy boundary; no pixels/latents read back.
    pub fn open_in(
        files: Files,
        context: &ModelContext,
        compiler: Option<&str>,
        options: PipelineOptions,
    ) -> Result<Pipeline> {
        // Built before the state so the models allocate on the stream that will
        // later dispatch them; allocation only needs a shared borrow.
        let mut stream = native_stream(context)?;
        let mut models = Models::open(&mut stream, &files, compiler)?;
        models.fusion.set_backend(options.fusion_backend);
        Ok(Pipeline {
            context: context.clone(),
            bridge: Arc::new(Bridge::new(native_stream(context)?)),
            models,
            files,
            compiler: compiler.map(str::to_string),
            state: Mutex::new(State {
                stream,
                weights: None,
                blocks: BlockCache::new(2, |blocks| blocks.plan.is_idle())?,
            }),
        })
    }

    /// Shared scheduler statistics cover coordinated tensors and block work;
    /// native model weights and auxiliary pools are accounted separately.
    pub fn context(&self) -> &ModelContext {
        &self.context
    }

    /// Backend and reason selected for the most recent fusion projection.
    pub fn fusion_selection(&self) -> String {
        self.models.fusion.reason()
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
        Ok(self.models.tokenizer.encode(text).map_err(crate::models::Error::from)?)
    }

    /// The conditioning tokens a prompt encodes to, without running the
    /// encoder: the template's 34-token prefix is not part of them.
    pub fn prompt_tokens(&self, prompt: &str) -> Result<usize> {
        let ids = self.models.tokenizer.prompt(prompt).map_err(crate::models::Error::from)?;
        Ok(ids.len() - 34)
    }

    /// A prompt encoded through Krea's template: float32 `[tokens][12][2560]`.
    pub fn encode(&self, prompt: &str) -> Result<(usize, Vec<f32>)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error("pipeline is poisoned; create a new pipeline".into()))?;
        let State { stream, .. } = &mut *state;
        self.bridge.check()?;
        let ids = self.models.tokenizer.prompt(prompt).map_err(crate::models::Error::from)?;
        let taps = self.models.encode(stream, &ids)?;
        stream.synchronize()?;
        Ok((ids.len() - 34, downloaded(stream, &taps)?))
    }

    /// Latents to RGB8 HWC.
    pub fn decode(&self, latents: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        dimensions(width, height)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error("pipeline is poisoned; create a new pipeline".into()))?;
        let State { stream, .. } = &mut *state;
        self.bridge.check()?;
        let packed = self.upload(stream, latents, width / PATCH * (height / PATCH), 64)?;
        let rgb = self.models.decode(stream, &packed, height, width)?;
        stream.synchronize()?;
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
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error("pipeline is poisoned; create a new pipeline".into()))?;
        let State { stream, weights, blocks } = &mut *state;
        self.bridge.check()?;
        let taps = self.upload(stream, text, text_tokens * 12, 2560)?;
        let conditioning = self.models.text_fusion(stream, &taps)?;
        let packed = self.upload(stream, latents, width / PATCH * (height / PATCH), 64)?;
        let velocity = self.forward(
            stream,
            weights,
            blocks,
            &packed,
            &conditioning,
            timestep,
            width,
            height,
        )?;
        stream.synchronize()?;
        downloaded(stream, &velocity)
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
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error("pipeline is poisoned; create a new pipeline".into()))?;
        let State { stream, weights, blocks } = &mut *state;
        self.bridge.check()?;
        let began = std::time::Instant::now();
        let mut timing = Profile::new(stream, "generate");

        let latents = match request.initial_latents {
            Some(values) => self.upload(stream, values, image_tokens, 64)?,
            None => {
                let values = noise::latents(request.seed, image_tokens * 64);
                self.upload(stream, &values, image_tokens, 64)?
            }
        };
        let text = self.conditioning(stream, request.prompt)?;
        let guided = guidance > 0.0;
        let uncond = if guided {
            Some(self.conditioning(stream, request.negative_prompt)?)
        } else {
            None
        };
        timing.mark(stream, "encode and text fusion")?;

        // Turbo's shift is fixed; Raw's follows the image's token count.
        let mu = if self.distilled() { 1.15 } else { schedule::dynamic_mu(image_tokens) };
        for step in 0..steps {
            let sigma = schedule::sigma(step, steps, mu);
            let next = schedule::sigma(step + 1, steps, mu);
            let velocity =
                self.forward(stream, weights, blocks, &latents, &text, sigma, width, height)?;
            if let Some(uncond) = &uncond {
                let unguided = self
                    .forward(stream, weights, blocks, &latents, uncond, sigma, width, height)?;
                self.models.ops.guidance(stream, &velocity, &unguided, guidance)?;
            }
            self.models.ops.euler_step(stream, &latents, &velocity, next - sigma)?;
            if let Some(progress) = progress.as_deref_mut() {
                stream.synchronize()?;
                if !progress(step + 1, steps, began.elapsed().as_secs_f64()) {
                    return Err(Error("cancelled".into()));
                }
            }
        }
        timing.mark(stream, "denoise")?;
        let rgb = self.models.decode(stream, &latents, height, width)?;
        stream.synchronize()?;
        timing.mark(stream, "VAE decode")?;
        Ok(rgb)
    }

    /// A prompt through the tokenizer, the encoder and the fusion tower.
    fn conditioning(&self, stream: &mut Stream, prompt: &str) -> Result<Tensor> {
        let ids = self.models.tokenizer.prompt(prompt).map_err(crate::models::Error::from)?;
        let taps = self.models.encode(stream, &ids)?;
        Ok(self.models.text_fusion(stream, &taps)?)
    }

    /// Text and image tokens through the 28 blocks and the final layer.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        stream: &mut Stream,
        weights: &mut Option<Arc<Weights>>,
        cache: &BlockCache,
        latents: &Tensor,
        text: &Tensor,
        timestep: f32,
        width: usize,
        height: usize,
    ) -> Result<Tensor> {
        let mut timing = Profile::new(stream, "forward");
        let image_tokens = width / PATCH * (height / PATCH);
        let tokens = text.rows() + image_tokens;
        let (embedding, modulation) = self.models.time(stream, timestep)?;
        let image = self.models.image_in(stream, latents)?;

        // The residual stream is the conditioning followed by the image.
        let x = self.models.ops.tensor(stream, tokens, WIDTH)?;
        stream.copy(x.binding()?.slice(0, text.size() * 2)?, text.binding()?)?;
        stream
            .copy(x.binding()?.slice(text.size() * 2, image.size() * 2)?, image.binding()?)?;

        timing.mark(stream, "embeddings")?;
        let mods = self.models.modulation(stream, &modulation)?;
        let blocks = self.prepare(
            stream,
            weights,
            cache,
            BlockShape { width, height, text_tokens: text.rows() },
        )?;
        timing.mark(stream, "prepare session")?;
        self.run_blocks(stream, &blocks, &x, mods)?;
        timing.mark(stream, "blocks")?;

        let output = x.view(image_tokens, WIDTH, text.rows() * WIDTH)?;
        let velocity = self.models.last(stream, &output, &embedding)?;
        timing.mark(stream, "final layer")?;
        Ok(velocity)
    }

    /// Reuse either guidance shape. Weights are shared by both workspaces.
    fn prepare(
        &self,
        stream: &mut Stream,
        weights: &mut Option<Arc<Weights>>,
        cache: &BlockCache,
        shape: BlockShape,
    ) -> Result<Arc<Blocks>> {
        let resident = match &*weights {
            Some(weights) => Arc::clone(weights),
            None => {
                let loaded = Arc::new(Weights::load(stream, &self.files.checkpoint)?);
                *weights = Some(Arc::clone(&loaded));
                loaded
            }
        };
        let build = || -> Result<Blocks> {
            let mut private = native_stream(&self.context)?;
            let session = Session::with_weights(
                &mut private,
                resident,
                shape.tokens(),
                28,
                self.compiler.as_deref(),
            )?;
            let mut rope = Rope::default();
            self.rope(&mut rope, shape.width, shape.height, shape.text_tokens);
            let plan = session.into_prepared(&self.context, private, rope.cos, rope.sin)?;
            Blocks::new(&self.context, plan, shape.tokens())
        };
        Ok(cache.get_or_prepare(shape, || {
            build().map_err(|e| hrx::Error::Message(e.to_string()))
        })?)
    }

    /// The rope tables, rebuilt when the geometry changes. Geometry, not just
    /// the token count: rectangular grids have different phases.
    fn rope(&self, rope: &mut Rope, width: usize, height: usize, text_tokens: usize) {
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
    fn upload(
        &self,
        stream: &mut Stream,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
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
        Ok(Tensor::from_slice(self.models.ops.pool(), stream, &bits, rows, cols)?)
    }
}

fn downloaded(stream: &mut Stream, tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.download(stream)?.into_iter().map(to_f32).collect())
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
    fn geometry_keys_distinguish_equal_sequence_lengths() {
        let portrait = BlockShape { width: 64, height: 128, text_tokens: 3 };
        let landscape = BlockShape { width: 128, height: 64, text_tokens: 3 };
        assert_eq!(portrait.tokens(), landscape.tokens());
        assert_ne!(portrait, landscape);
        let cache = hrx::plan_cache::PlanCache::new(2, |_: &usize| true).unwrap();
        let mut builds = 0;
        for shape in [portrait, landscape, portrait, landscape] {
            let result = cache
                .get_or_prepare(shape, || {
                    builds += 1;
                    Ok(shape.tokens())
                })
                .unwrap();
            assert_eq!(*result, shape.tokens());
        }
        assert_eq!(builds, 2);
    }

    #[test]
    fn the_dimensions_the_patching_cannot_serve_are_refused() {
        assert!(dimensions(1024, 1024).is_ok());
        for (width, height) in [(1020, 1024), (1024, 2064), (32, 64), (64, 63)] {
            let error = dimensions(width, height).unwrap_err();
            assert!(error.0.contains("multiples of 16"), "{width}x{height}: {error}");
        }
    }
}
