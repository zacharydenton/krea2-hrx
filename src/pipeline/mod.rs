//! Krea 2 generation from prompts to RGB.
//! Models remain resident across calls. Two recent block shapes are retained so
//! guided generation can alternate conditioning lengths without rebuilding.

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
use hrx::Stream;
use hrx::inference::ModelContext;
use shared::{BlockCache, BlockShape, Blocks, Bridge, native_stream};

use self::profile::Profile;

pub use crate::kernels::Attention;
pub use crate::models::{Files, hub};
pub use crate::{Error, Result};

/// The transformer's residual width, and the patch size in pixels.
const WIDTH: usize = 6144;
const PATCH: usize = 16;

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

/// One sampling step's inputs to the blocks, shared by both guidance branches.
struct StepInputs {
    embedding: Tensor,
    image: Tensor,
    /// Float32 `[28][6][6144]`. Shared because each block handoff retains the
    /// source until its copy drains.
    modulation: Arc<hrx::PooledBuffer>,
}

/// Everything resident: the three models, the block weights, and whichever
/// two most recently used block sessions.
pub struct Pipeline {
    context: ModelContext,
    bridge: Arc<Bridge>,
    files: Files,
    compiler: Option<String>,
    attention: Attention,
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

/// Optional execution policy.
#[derive(Clone, Debug, Default)]
pub struct PipelineOptions {
    /// Backend for the first text-fusion up projection. `Auto` is the GPU.
    pub fusion_backend: crate::fusion::FusionBackend,
    /// Attention kernels for every block session. `None` takes
    /// [`Attention::from_environment`] when the pipeline opens.
    pub attention: Option<Attention>,
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
        // The models allocate on the stream that will later dispatch them, which
        // then moves into the state lock.
        let attention = match options.attention {
            Some(attention) => attention,
            None => Attention::from_environment()?,
        };
        let mut stream = native_stream(context)?;
        let mut models = Models::open(&mut stream, &files, compiler)?;
        models.fusion.set_backend(options.fusion_backend);
        Ok(Pipeline {
            context: context.clone(),
            bridge: Arc::new(Bridge::new(native_stream(context)?)),
            models,
            files,
            compiler: compiler.map(str::to_string),
            attention,
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
        self.models.tokenizer.encode(text)
    }

    /// The conditioning tokens a prompt encodes to, without running the
    /// encoder: the template's 34-token prefix is not part of them.
    pub fn prompt_tokens(&self, prompt: &str) -> Result<usize> {
        let ids = self.models.tokenizer.prompt(prompt)?;
        Ok(ids.len() - 34)
    }

    /// A prompt encoded through Krea's template: float32 `[tokens][12][2560]`.
    pub fn encode(&self, prompt: &str) -> Result<(usize, Vec<f32>)> {
        let mut state = self.state.lock().map_err(|_| Error::Poisoned("pipeline"))?;
        let State { stream, .. } = &mut *state;
        self.bridge.check()?;
        let ids = self.models.tokenizer.prompt(prompt)?;
        let taps = self.models.encode(stream, &ids)?;
        stream.synchronize()?;
        Ok((ids.len() - 34, downloaded(stream, &taps)?))
    }

    /// Latents to RGB8 HWC.
    pub fn decode(&self, latents: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        dimensions(width, height)?;
        let mut state = self.state.lock().map_err(|_| Error::Poisoned("pipeline"))?;
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
            return Err(Error::invalid("invalid transformer arguments"));
        }
        let mut state = self.state.lock().map_err(|_| Error::Poisoned("pipeline"))?;
        let State { stream, weights, blocks } = &mut *state;
        self.bridge.check()?;
        let taps = self.upload(stream, text, text_tokens * 12, 2560)?;
        let conditioning = self.models.text_fusion(stream, &taps)?;
        let packed = self.upload(stream, latents, width / PATCH * (height / PATCH), 64)?;
        let inputs = self.step_inputs(stream, &packed, timestep)?;
        let velocity =
            self.forward(stream, weights, blocks, &inputs, &conditioning, width, height)?;
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
            return Err(Error::invalid("invalid generation arguments"));
        }
        let image_tokens = width / PATCH * (height / PATCH);
        let mut state = self.state.lock().map_err(|_| Error::Poisoned("pipeline"))?;
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
            // Both guidance branches see the same latents at the same timestep,
            // so they share the embeddings and the modulation tables.
            let inputs = self.step_inputs(stream, &latents, sigma)?;
            let velocity =
                self.forward(stream, weights, blocks, &inputs, &text, width, height)?;
            if let Some(uncond) = &uncond {
                let unguided =
                    self.forward(stream, weights, blocks, &inputs, uncond, width, height)?;
                self.models.ops.guidance(stream, &velocity, &unguided, guidance)?;
            }
            self.models.ops.euler_step(stream, &latents, &velocity, next - sigma)?;
            if let Some(progress) = progress.as_deref_mut() {
                stream.synchronize()?;
                if !progress(step + 1, steps, began.elapsed().as_secs_f64()) {
                    return Err(Error::Cancelled);
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
        let ids = self.models.tokenizer.prompt(prompt)?;
        let taps = self.models.encode(stream, &ids)?;
        self.models.text_fusion(stream, &taps)
    }

    /// What one step's forwards share: the timestep embedding, the image
    /// tokens, and the blocks' modulation tables.
    fn step_inputs(
        &self,
        stream: &mut Stream,
        latents: &Tensor,
        timestep: f32,
    ) -> Result<StepInputs> {
        let mut timing = Profile::new(stream, "step");
        let (embedding, modulation) = self.models.time(stream, timestep)?;
        let image = self.models.image_in(stream, latents)?;
        let modulation = Arc::new(self.models.modulation(stream, &modulation)?);
        timing.mark(stream, "embeddings")?;
        Ok(StepInputs { embedding, image, modulation })
    }

    /// Text and image tokens through the 28 blocks and the final layer.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        stream: &mut Stream,
        weights: &mut Option<Arc<Weights>>,
        cache: &BlockCache,
        inputs: &StepInputs,
        text: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor> {
        let mut timing = Profile::new(stream, "forward");
        let blocks = self.prepare(
            stream,
            weights,
            cache,
            BlockShape { width, height, text_tokens: text.rows() },
        )?;
        timing.mark(stream, "prepare session")?;
        // The residual stream is the conditioning followed by the image; only
        // the image rows come back.
        let output =
            self.run_blocks(stream, &blocks, text, &inputs.image, &inputs.modulation)?;
        timing.mark(stream, "blocks")?;
        let velocity = self.models.last(stream, &output, &inputs.embedding)?;
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
                self.attention,
                self.compiler.as_deref(),
            )?;
            let (cos, sin) = rope(shape);
            let plan = session.into_prepared(&self.context, private, cos, sin)?;
            Blocks::new(&self.context, plan, shape.tokens())
        };
        Ok(cache.get_or_prepare(shape, || build().map_err(hrx::Error::from))?)
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
            return Err(Error::invalid("wrong input buffer size"));
        }
        let mut bits = Vec::with_capacity(values.len());
        for &value in values {
            if !value.is_finite() {
                return Err(Error::invalid("nonfinite input"));
            }
            let rounded = from_f32(value);
            if !to_f32(rounded).is_finite() {
                return Err(Error::invalid("input exceeds bf16 range"));
            }
            bits.push(rounded);
        }
        Tensor::from_slice(self.models.ops.pool(), stream, &bits, rows, cols)
    }
}

/// The rotary tables for one image geometry: float32 `[tokens][128]` cosines
/// and sines. Geometry, not just the token count: rectangular grids have
/// different phases. Built once per prepared block shape.
fn rope(shape: BlockShape) -> (Vec<f32>, Vec<f32>) {
    let columns = shape.width / PATCH;
    let tokens = shape.tokens();
    // Three axes over 128 channels: 32 for the frame, 48 each for the row and
    // the column. Each channel pair shares one frequency.
    let axes = [32usize, 48, 48];
    let frequencies: Vec<f64> = axes
        .iter()
        .flat_map(|&width| {
            (0..width)
                .map(move |channel| 1000f64.powf(-2.0 * (channel / 2) as f64 / width as f64))
        })
        .collect();
    let mut cos = vec![0.0; tokens * 128];
    let mut sin = vec![0.0; tokens * 128];
    // Text tokens sit at position zero on all three axes, so theirs stay 0/1.
    for token in 0..shape.text_tokens {
        cos[token * 128..(token + 1) * 128].fill(1.0);
    }
    for token in shape.text_tokens..tokens {
        let patch = token - shape.text_tokens;
        let positions = [0, patch / columns, patch % columns];
        let mut channel = 0;
        for (axis, &width) in axes.iter().enumerate() {
            for _ in 0..width {
                let phase = positions[axis] as f64 * frequencies[channel];
                cos[token * 128 + channel] = phase.cos() as f32;
                sin[token * 128 + channel] = phase.sin() as f32;
                channel += 1;
            }
        }
    }
    (cos, sin)
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
        return Err(Error::invalid("dimensions must be multiples of 16 in 64..2048"));
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
    fn rope_tables_follow_the_three_axis_positions() {
        let shape = BlockShape { width: 64, height: 48, text_tokens: 2 };
        let (cos, sin) = rope(shape);
        assert_eq!(cos.len(), shape.tokens() * 128);
        // The per-token formula the hoisted frequencies must reproduce exactly.
        for token in 0..shape.tokens() {
            let mut offset = 0;
            for (axis, width) in [32usize, 48, 48].into_iter().enumerate() {
                let position = match (axis, token >= 2) {
                    (1, true) => (token - 2) / 4,
                    (2, true) => (token - 2) % 4,
                    _ => 0,
                };
                for channel in 0..width {
                    let phase = position as f64
                        * 1000f64.powf(-2.0 * (channel / 2) as f64 / width as f64);
                    let at = token * 128 + offset + channel;
                    assert_eq!(cos[at], phase.cos() as f32, "cos {token} {axis} {channel}");
                    assert_eq!(sin[at], phase.sin() as f32, "sin {token} {axis} {channel}");
                }
                offset += width;
            }
        }
    }

    #[test]
    fn the_dimensions_the_patching_cannot_serve_are_refused() {
        assert!(dimensions(1024, 1024).is_ok());
        for (width, height) in [(1020, 1024), (1024, 2064), (32, 64), (64, 63)] {
            let error = dimensions(width, height).unwrap_err();
            assert!(error.to_string().contains("multiples of 16"), "{width}x{height}: {error}");
        }
    }
}
