//! Checkpointed main-block training, deterministic sampling and resumable state.
use super::{
    PreparedDataset, TrainConfig,
    model::Transformer,
    ops::{self, FloatTensor},
    optimizer, prepare,
};
use crate::checkpoint::Checkpoint;
use crate::lora::{Adapter, Factors, PROJECTIONS, SavedTensor, floats, io, save_tensors};
use crate::models::Models;
use crate::models::graph::{LATENT_MEAN, LATENT_STDDEV};
use crate::numerics::{from_f32, to_f32};
use crate::ops::{Binary, Tensor};
use crate::{Error, Result};
use hrx::Stream;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Serialize, Deserialize)]
struct State {
    config: TrainConfig,
    step: usize,
    epoch: u64,
    cursor: usize,
    rng_word: String,
    fingerprint: String,
    software: String,
    compiler: String,
    target: String,
}

/// Conservative device allocation estimate before opening a training stream.
#[derive(Debug, Serialize)]
pub struct MemoryEstimate {
    /// Frozen checkpoint bytes, including preserved FP32 norm copies.
    pub frozen: usize,
    /// FP32 masters, gradients and two moments, plus BF16 factors.
    pub adapters: usize,
    /// Saved residual input at every block boundary.
    pub activations: usize,
    /// Recomputed block, transpose buffers, temporary gradients and pool headroom.
    pub scratch: usize,
    /// Sum of the planned device allocations.
    pub total: usize,
}

impl MemoryEstimate {
    /// Compute an upper estimate from validated caches and the actual checkpoint.
    pub fn for_run(c: &TrainConfig, data: &PreparedDataset) -> Result<Self> {
        let file = Checkpoint::open(&c.model)?;
        let mut frozen = 0usize;
        for name in file.names() {
            let t = file.get(name)?;
            frozen = frozen
                .checked_add(t.bytes.len() * 2)
                .ok_or_else(|| Error::invalid("checkpoint size overflow"))?;
        }
        // Weight storage is BF16, except small norm vectors retained in both formats.
        // A 2x file bound also covers loader staging and an individual transpose.
        let parameters: usize = PROJECTIONS.iter().map(|(_, o, i)| (o + i) * c.rank * 28).sum();
        let adapters = parameters * 18;
        let mut tokens = 0;
        for sample in &data.samples {
            let f = Checkpoint::open(&prepare::cache_path(c, sample, "text"))?;
            let text = f.get("conditioning")?.shape[0];
            tokens = tokens.max(sample.width / 16 * (sample.height / 16) + text);
        }
        let activations = 29 * tokens * 6144 * 2;
        let scratch = tokens * (6144 * 24 + 16384 * 10) * 2 + (4usize << 30);
        let total = frozen + adapters + activations + scratch;
        Ok(Self { frozen, adapters, activations, scratch, total })
    }
}

/// One resident training run, with a single stream and update-boundary checkpoints.
pub struct Trainer {
    config: TrainConfig,
    data: PreparedDataset,
    models: Models,
    model: Transformer,
    stream: Stream,
    state: State,
    rng: ChaCha8Rng,
    order: Vec<usize>,
    failed: bool,
    _budget: hrx::residency::ResidencyManager,
}

/// Measurements for one completed optimizer update, excluding checkpoint I/O.
#[derive(Clone, Copy, Debug)]
pub struct StepStats {
    /// Number of completed optimizer updates.
    pub step: usize,
    /// Mean flow loss across the accumulated microbatches.
    pub loss: f64,
    /// Averaged gradient norm before clipping.
    pub gradient_norm: f64,
    /// Wall time including device synchronization.
    pub seconds: f64,
}

fn order(count: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(
        seed ^ epoch.wrapping_mul(0x9e3779b97f4a7c15) ^ 0xa5a5a5a55a5a5a5a,
    );
    order.shuffle(&mut rng);
    order
}

impl Trainer {
    /// Open a prepared run; use `prepare` first. No text encoder or VAE stays resident.
    pub fn open(config: TrainConfig) -> Result<Self> {
        Self::build(config, None)
    }

    /// Resume exact FP32 master/optimizer state at a complete update boundary.
    pub fn resume(checkpoint: &Path) -> Result<Self> {
        let state: State =
            serde_json::from_slice(&std::fs::read(checkpoint.join("state.json")).map_err(io)?)
                .map_err(io)?;
        Self::build(state.config.clone(), Some((checkpoint, state)))
    }

    fn build(config: TrainConfig, resume: Option<(&Path, State)>) -> Result<Self> {
        config.validate()?;
        let data = prepare::load(&config)?;
        let memory = MemoryEstimate::for_run(&config, &data)?;
        eprintln!(
            "planned training allocations: {:.2} GiB (budget {} GiB)",
            memory.total as f64 / (1u64 << 30) as f64,
            config.memory_gib
        );
        if memory.total > config.memory_gib * (1usize << 30) {
            return Err(Error::invalid("training allocation estimate exceeds memory_gib"));
        }
        let budget = hrx::residency::ResidencyManager::new(config.memory_gib * (1usize << 30))?;
        let mut stream = Stream::open()?.with_memory_budget(budget.budget());
        let target = stream.target().as_str().to_owned();
        let mut state = State {
            config: config.clone(),
            step: 0,
            epoch: 0,
            cursor: 0,
            rng_word: "0".into(),
            fingerprint: data.fingerprint.clone(),
            software: prepare::software_hash(),
            compiler: crate::kernels::cache::compiler_for_target(None, stream.target())?
                .identity()
                .to_owned(),
            target,
        };
        if let Some((_, saved)) = &resume {
            if saved.fingerprint != data.fingerprint
                || saved.software != state.software
                || saved.compiler != state.compiler
                || saved.target != state.target
                || saved.cursor > data.samples.len()
                || saved.step > config.steps
            {
                return Err(Error::invalid(
                    "resume dataset, software, device or cursor mismatch",
                ));
            }
            state = saved.clone();
        }
        let adapter = match &resume {
            Some((path, _)) => {
                read_master_adapter(&path.join("optimizer.safetensors"), &config)?
            }
            None => Adapter::initialize(config.rank, config.alpha, config.seed)?,
        };
        let models = Models::load_parts(&mut stream, &config.model, None, None, None, None)?;
        let model = Transformer::load(&models.ops, &mut stream, &config.model, Some(&adapter))?;
        drop(adapter);
        if let Some((path, _)) = &resume {
            let file = Checkpoint::open(&path.join("optimizer.safetensors"))?;
            for (name, p) in &model.adapters {
                for (part, p) in [("a", &p.a), ("b", &p.b)] {
                    for (kind, tensor) in [("first", &p.first), ("second", &p.second)] {
                        let t = file.get(&format!("{name}.{part}.{kind}"))?;
                        if t.dtype != crate::checkpoint::DType::F32
                            || t.shape != [tensor.rows(), tensor.cols()]
                        {
                            return Err(Error::invalid("optimizer state shape/dtype mismatch"));
                        }
                        let values = floats(t)?;
                        if values
                            .iter()
                            .any(|v| !v.is_finite() || (kind == "second" && *v < 0.0))
                        {
                            return Err(Error::invalid("invalid optimizer moments"));
                        }
                        stream.upload(tensor.binding(), t.bytes)?;
                    }
                }
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ 0x1234_5678_abcd_ef01);
        rng.set_word_pos(state.rng_word.parse().map_err(io)?);
        let order = order(data.samples.len(), config.seed, state.epoch);
        Ok(Self {
            config,
            data,
            models,
            model,
            stream,
            state,
            rng,
            order,
            failed: false,
            _budget: budget,
        })
    }

    /// Immutable settings used to construct this run and its checkpoints.
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Completed optimizer updates.
    pub fn step(&self) -> usize {
        self.state.step
    }

    /// Complete one update without saving. Returns `None` at the configured limit.
    ///
    /// After a failed update, discard this trainer and resume a complete checkpoint;
    /// partially accumulated gradients and device writes cannot be saved or retried.
    pub fn train_step(&mut self) -> Result<Option<StepStats>> {
        if self.failed {
            return Err(Error::invalid("training update failed; resume a complete checkpoint"));
        }
        if self.state.step == self.config.steps {
            return Ok(None);
        }
        self.failed = true;
        let started = Instant::now();
        let mut loss = 0.0;
        for _ in 0..self.config.accumulation {
            if self.state.cursor == self.order.len() {
                self.state.epoch += 1;
                self.state.cursor = 0;
                self.order = order(self.data.samples.len(), self.config.seed, self.state.epoch);
            }
            let index = self.order[self.state.cursor];
            loss += self.microbatch(index)?;
            self.state.cursor += 1;
        }
        let next = self.state.step + 1;
        let norm = optimizer::update(
            &self.models.ops,
            &mut self.stream,
            &self.model,
            &self.config,
            next,
        )?;
        self.stream.synchronize()?;
        self.state.step = next;
        self.state.rng_word = self.rng.get_word_pos().to_string();
        self.failed = false;
        Ok(Some(StepStats {
            step: next,
            loss: loss / self.config.accumulation as f64,
            gradient_norm: norm,
            seconds: started.elapsed().as_secs_f64(),
        }))
    }

    /// Train to the configured final step. A false callback result saves and stops.
    pub fn run(
        &mut self,
        mut progress: impl FnMut(usize, f64, f64, f64) -> bool,
    ) -> Result<()> {
        while let Some(stats) = self.train_step()? {
            let next = stats.step;
            let keep_going = progress(next, stats.loss, stats.gradient_norm, stats.seconds);
            if next.is_multiple_of(self.config.save_every)
                || next == self.config.steps
                || !keep_going
            {
                self.save()?;
            }
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    fn microbatch(&mut self, index: usize) -> Result<f64> {
        let sample = &self.data.samples[index];
        let posterior =
            Checkpoint::open(&prepare::cache_path(&self.config, sample, "posterior"))?;
        let moments = floats(posterior.get("posterior")?)?;
        let (h, w) = (sample.height / 8, sample.width / 8);
        let clean = sample_latent(&moments, h, w, &mut self.rng)?;
        let sigma = sample_sigma(w / 2 * (h / 2), &mut self.rng);
        let (noisy, target) = flow_sample(&clean, sigma, &mut self.rng);
        let ops = &self.models.ops;
        let stream = &mut self.stream;
        let noisy = Tensor::from_slice(ops.pool(), stream, &noisy, h / 2 * (w / 2), 64)?;
        let textfile = Checkpoint::open(&prepare::cache_path(&self.config, sample, "text"))?;
        let text = textfile.get("conditioning")?;
        let textbits: Vec<u16> =
            text.bytes.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect();
        let text = Tensor::from_slice(ops.pool(), stream, &textbits, text.shape[0], 6144)?;
        let (embedding, modvec) = self.models.time(stream, sigma)?;
        let image = self.models.image_in(stream, &noisy)?;
        let mods_f32 = self.models.modulation(stream, &modvec)?;
        let mods = ops::cast_view(ops, stream, mods_f32.binding(), 28 * 6, 6144)?;
        let tokens = text.rows() + image.rows();
        let initial = ops.tensor(stream, tokens, 6144)?;
        stream.copy(initial.binding()?.slice(0, text.size() * 2)?, text.binding()?)?;
        stream.copy(
            initial.binding()?.slice(text.size() * 2, image.size() * 2)?,
            image.binding()?,
        )?;
        let (cos, sin) = crate::pipeline::rope_tables(sample.width, sample.height, text.rows());
        let cos = FloatTensor::from_slice(stream, tokens, 128, &cos)?;
        let sin = FloatTensor::from_slice(stream, tokens, 128, &sin)?;
        let mut boundaries = vec![initial];
        for block in 0..28 {
            let m = mods.view(6, 6144, block * 6 * 6144)?;
            let tape = self.model.block(
                ops,
                stream,
                block,
                boundaries.last().expect("input"),
                &m,
                &cos,
                &sin,
                1.0,
            )?;
            boundaries.push(tape.output);
        }
        let last = boundaries.last().expect("output").view(image.rows(), 6144, text.size())?;
        let prediction = self.models.last(stream, &last, &embedding)?;
        let target = FloatTensor::from_slice(stream, image.rows(), 64, &target)?;
        let (loss, gradient) = ops::flow_loss(ops, stream, &prediction, &target)?;
        let image_grad = last_backward(&self.models, stream, &last, &embedding, &gradient)?;
        let mut grad = ops.tensor(stream, tokens, 6144)?;
        grad.zero(stream)?;
        stream.copy(
            grad.binding()?.slice(text.size() * 2, image.size() * 2)?,
            image_grad.binding()?,
        )?;
        for block in (0..28).rev() {
            boundaries.pop();
            let input = boundaries.last().expect("saved block input");
            let m = mods.view(6, 6144, block * 6 * 6144)?;
            let tape = self.model.block(ops, stream, block, input, &m, &cos, &sin, 1.0)?;
            grad = self.model.backward(ops, stream, block, tape, &m, &cos, &sin, &grad)?;
        }
        Ok(loss)
    }

    /// Atomically save resumable FP32 state and a separate portable BF16 adapter.
    pub fn save(&mut self) -> Result<PathBuf> {
        if self.failed {
            return Err(Error::invalid("cannot checkpoint a failed training update"));
        }
        self.stream.synchronize()?;
        let root = self.config.output.join("checkpoints");
        std::fs::create_dir_all(&root).map_err(io)?;
        let path = root.join(format!("step-{:06}", self.state.step));
        if path.exists() {
            return Err(Error::invalid(format!(
                "checkpoint already exists: {}",
                path.display()
            )));
        }
        let temporary =
            root.join(format!(".step-{:06}-{}.tmp", self.state.step, std::process::id()));
        std::fs::create_dir(&temporary).map_err(io)?;
        let result: Result<()> = (|| {
            let adapter = self.model.adapter(&mut self.stream)?;
            let metadata = [
                ("base_model".into(), self.config.model.display().to_string()),
                ("trigger".into(), self.config.trigger.clone()),
                ("step".into(), self.state.step.to_string()),
            ]
            .into();
            adapter.save(&temporary.join("adapter.safetensors"), metadata)?;
            let mut tensors = BTreeMap::new();
            for (name, p) in &self.model.adapters {
                for (part, p) in [("a", &p.a), ("b", &p.b)] {
                    for (kind, t) in
                        [("master", &p.master), ("first", &p.first), ("second", &p.second)]
                    {
                        let values = t.download(&mut self.stream)?;
                        tensors.insert(
                            format!("{name}.{part}.{kind}"),
                            SavedTensor {
                                dtype: "F32",
                                shape: vec![t.rows(), t.cols()],
                                bytes: values.into_iter().flat_map(f32::to_le_bytes).collect(),
                            },
                        );
                    }
                }
            }
            save_tensors(&temporary.join("optimizer.safetensors"), tensors, BTreeMap::new())?;
            self.state.rng_word = self.rng.get_word_pos().to_string();
            prepare::write_json(&temporary.join("state.json"), &self.state)?;
            std::fs::rename(&temporary, &path).map_err(io)?;
            prepare::write_json(&self.config.output.join("run.json"), &self.config)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&temporary);
        }
        result?;
        let mut complete = std::fs::read_dir(&root)
            .map_err(io)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("step-"))
                    && p.join("state.json").is_file()
            })
            .collect::<Vec<_>>();
        complete.sort();
        let remove = complete.len().saturating_sub(self.config.keep_checkpoints);
        for old in complete.into_iter().take(remove) {
            std::fs::remove_dir_all(old).map_err(io)?;
        }
        Ok(path)
    }
}

fn read_master_adapter(path: &Path, c: &TrainConfig) -> Result<Adapter> {
    let file = Checkpoint::open(path)?;
    let mut layers = BTreeMap::new();
    for block in 0..28 {
        for (name, outputs, inputs) in PROJECTIONS {
            let key = format!("blocks.{block}.{name}");
            let a = file.get(&format!("{key}.a.master"))?;
            let b = file.get(&format!("{key}.b.master"))?;
            if a.dtype != crate::checkpoint::DType::F32
                || b.dtype != a.dtype
                || a.shape != [c.rank, inputs]
                || b.shape != [outputs, c.rank]
            {
                return Err(Error::invalid("resume master factor shape/dtype mismatch"));
            }
            let factors = Factors {
                inputs,
                outputs,
                rank: c.rank,
                alpha: c.alpha,
                a: floats(a)?,
                b: floats(b)?,
            };
            factors.validate()?;
            layers.insert(key, factors);
        }
    }
    Ok(Adapter { layers })
}

fn last_backward(
    models: &Models,
    stream: &mut Stream,
    x: &Tensor,
    embedding: &Tensor,
    grad: &Tensor,
) -> Result<Tensor> {
    let ops = &models.ops;
    let w = models.transformer.get("last.linear.weight")?;
    let wt = ops::transpose(ops, stream, &w.tensor(w.shape[0], w.shape[1])?)?;
    let g = ops::matmul(ops, stream, grad, &wt, 1.0)?;
    let table =
        models.transformer.get("last.modulation.lin")?.tensor(2, 6144)?.view(1, 6144, 0)?;
    let factor = ops.binary(stream, embedding, &table, Binary::Add)?;
    let factor = ops::one_plus(ops, stream, &factor)?;
    let g = ops.binary(stream, &g, &factor, Binary::Mul)?;
    let scales = models.transformer.get("last.norm.scale")?.f32_values(stream)?;
    ops::norm_backward(ops, stream, x, &g, scales, 1e-5)
}

/// Sample the shifted logit-normal flow time; one means noise and zero means clean.
pub fn sample_sigma(image_tokens: usize, rng: &mut ChaCha8Rng) -> f32 {
    let z: f64 = StandardNormal.sample(rng);
    let u = 1.0 / (1.0 + (-z).exp());
    let shift = crate::pipeline::schedule::dynamic_mu(image_tokens).exp();
    (shift * u / (1.0 + (shift - 1.0) * u)) as f32
}

/// Draw posterior samples, normalize each latent channel, and pack 2x2 patches.
pub fn sample_latent(
    moments: &[f32],
    h: usize,
    w: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f32>> {
    if h == 0
        || w == 0
        || !h.is_multiple_of(2)
        || !w.is_multiple_of(2)
        || moments.len() != h * w * 32
        || moments.iter().any(|v| !v.is_finite())
    {
        return Err(Error::invalid("invalid VAE posterior moments"));
    }
    let mut packed = vec![0.0; h * w * 16];
    for y in 0..h {
        for x in 0..w {
            for c in 0..16 {
                let mean = moments[(y * w + x) * 32 + c];
                let logvar = moments[(y * w + x) * 32 + 16 + c].clamp(-30.0, 20.0);
                let noise: f32 = StandardNormal.sample(rng);
                let value = to_f32(from_f32(mean + (0.5 * logvar).exp() * noise));
                let center = to_f32(from_f32(LATENT_MEAN[c]));
                let inv = to_f32(from_f32(1.0 / to_f32(from_f32(LATENT_STDDEV[c]))));
                let norm = to_f32(from_f32(to_f32(from_f32(value - center)) * inv));
                let at = ((y / 2) * (w / 2) + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2;
                packed[at] = norm;
            }
        }
    }
    Ok(packed)
}

fn flow_sample(clean: &[f32], sigma: f32, rng: &mut ChaCha8Rng) -> (Vec<u16>, Vec<f32>) {
    let mut noisy = Vec::with_capacity(clean.len());
    let mut target = Vec::with_capacity(clean.len());
    for &value in clean {
        let noise: f32 = StandardNormal.sample(rng);
        let noise = to_f32(from_f32(noise));
        noisy.push(from_f32((1.0 - sigma) * value + sigma * noise));
        target.push(noise - value);
    }
    (noisy, target)
}

#[cfg(test)]
fn loss_gradient(prediction: &[u16], target: &[f32]) -> Result<(f64, Vec<u16>)> {
    if prediction.is_empty() || prediction.len() != target.len() {
        return Err(Error::invalid("loss dimensions"));
    }
    let mut loss = 0.0f64;
    let mut gradient = Vec::with_capacity(prediction.len());
    for (&p, &t) in prediction.iter().zip(target) {
        let delta = to_f32(p) - t;
        let squared = delta * delta;
        let grad = from_f32((2.0 / prediction.len() as f32) * delta);
        if !squared.is_finite() || !to_f32(grad).is_finite() {
            return Err(Error::invalid("nonfinite flow loss; update cancelled"));
        }
        loss += f64::from(squared);
        gradient.push(grad);
    }
    Ok((loss / prediction.len() as f64, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rng_resume_reproduces_times_noise_and_epoch_orders() {
        let mut a = ChaCha8Rng::seed_from_u64(37);
        for _ in 0..17 {
            sample_sigma(4096, &mut a);
        }
        let pos = a.get_word_pos();
        let mut b = ChaCha8Rng::seed_from_u64(37);
        b.set_word_pos(pos);
        for _ in 0..32 {
            assert_eq!(sample_sigma(4096, &mut a), sample_sigma(4096, &mut b));
        }
        assert_eq!(order(14, 37, 3), order(14, 37, 3));
        assert_ne!(order(14, 37, 3), order(14, 37, 4));
        let mut sorted = order(14, 37, 3);
        sorted.sort();
        assert_eq!(sorted, (0..14).collect::<Vec<_>>());
    }
    #[test]
    fn flow_endpoints_loss_and_nonfinite_checks() {
        let clean = vec![1.0, -2.0, 0.5, 4.0];
        let mut a = ChaCha8Rng::seed_from_u64(9);
        let (x, t) = flow_sample(&clean, 0.0, &mut a);
        assert_eq!(x.iter().copied().map(to_f32).collect::<Vec<_>>(), clean);
        let (loss, grad) = loss_gradient(&[from_f32(2.0), from_f32(4.0)], &[1.0, 2.0]).unwrap();
        assert_eq!(loss, 2.5);
        assert_eq!(grad, [from_f32(1.0), from_f32(2.0)]);
        assert!(loss_gradient(&[from_f32(f32::INFINITY)], &[0.0]).is_err());
        assert!(loss_gradient(&[from_f32(1e30)], &[0.0]).is_err());
        assert_eq!(t.len(), clean.len());
    }
}
