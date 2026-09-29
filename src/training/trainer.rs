//! Checkpointed main-block training, deterministic sampling and resumable state.
use super::{
    PreparedDataset, TrainConfig,
    auxiliary::Tape,
    model::Transformer,
    ops::{self, FloatTensor},
    optimizer, prepare,
};
use crate::checkpoint::Checkpoint;
use crate::lora::{Adapter, Factors, SavedTensor, Targets, floats, io, save_tensors};
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
    /// Saved block boundaries or complete activation tapes.
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
        // Auxiliary and main-block loaders partition this file; they do not
        // retain two copies of every weight. F32 tensors additionally retain
        // their BF16 execution copy, and BF16 norms gain a lazy F32 scale copy.
        let mut frozen: usize = 28 * 6 * 6144 * 2 + 2 * 4096;
        for name in file.names() {
            let t = file.get(name)?;
            frozen = frozen
                .checked_add(resident_weight_bytes(
                    t.dtype,
                    t.bytes.len(),
                    name.ends_with(".scale"),
                )?)
                .ok_or_else(|| Error::invalid("checkpoint size overflow"))?;
        }
        // Upload staging is bounded by HRX, not the size of the file. Host
        // conversion/checkpoint buffers are covered by the separate RAM reserve.
        let parameters: usize =
            c.targets.layers().iter().map(|(_, o, i)| (o + i) * c.rank).sum();
        let adapters = parameters * 18;
        let mut tokens = 0;
        let mut text_tokens = 0;
        for sample in &data.samples {
            let f = Checkpoint::open(&prepare::cache_path(c, sample, "text"))?;
            let text = f.get(prepare::text_spec(c).0)?.shape[0]
                / if c.targets == Targets::All { 12 } else { 1 };
            text_tokens = text_tokens.max(text);
            tokens = tokens.max(sample.width / 16 * (sample.height / 16) + text);
        }
        let mut activations = if c.gradient_checkpointing {
            29 * tokens * 6144 * 2
        } else {
            // Every BlockTape field, including FP32 attention output and LSE.
            // The boundary vector aliases these tensors, allocating no storage.
            // Adjacent blocks' shared input/output is still counted twice here.
            28 * tokens * (137728 * 2 + 6144 * 4 + 48 * 4)
        };
        if c.targets == Targets::All {
            // Four extra main-block tensors only when modulation is trainable.
            if !c.gradient_checkpointing {
                activations += 28 * tokens * 4 * 6144 * 2;
            }
            // All text-fusion forward nodes, gradients and the first/last tapes.
            // Layerwise towers have twelve rows per conditioning token.
            activations += text_tokens * 12 * 2560 * 2 * 96 + tokens * 6144 * 2 * 16;
        }
        // Temporary activations/gradients and the small final projection's
        // transpose are covered by the shape-dependent workspace below. The
        // old 2 GiB allowance for dense weight transposes is no longer needed:
        // dX consumes row-major weights directly. Bound cached + in-flight HRX
        // staging (2 * 64 MiB), plus 128 MiB for dispatch/code/graph storage.
        let runtime_overhead = 256usize << 20;
        let scratch = tokens * (6144 * 24 + 16384 * 10) * 2
            + runtime_overhead
            + (c.scratch_pool_mib << 20);
        let total = frozen + adapters + activations + scratch;
        Ok(Self { frozen, adapters, activations, scratch, total })
    }
}

fn resident_weight_bytes(
    dtype: crate::checkpoint::DType,
    bytes: usize,
    norm: bool,
) -> Result<usize> {
    use crate::checkpoint::DType;
    let aligned = |size: usize, alignment: usize| {
        size.checked_add(alignment - 1).map(|n| n / alignment * alignment)
    };
    let extra = match dtype {
        DType::F32 => bytes / 2,
        DType::BF16 if norm => {
            bytes.checked_mul(2).ok_or_else(|| Error::invalid("norm size overflow"))?
        }
        DType::BF16 => 0,
        _ => {
            return Err(Error::invalid(
                "training memory estimate requires dense BF16/F32 weights",
            ));
        }
    };
    aligned(bytes, 256)
        .and_then(|base| aligned(extra, 4096).and_then(|extra| base.checked_add(extra)))
        .ok_or_else(|| Error::invalid("resident weight size overflow"))
}

/// One resident training run, with a single stream and update-boundary checkpoints.
pub struct Trainer {
    config: TrainConfig,
    data: PreparedDataset,
    models: Models,
    model: Transformer,
    optimizer: optimizer::PreparedOptimizer,
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
        super::memory::during_run()?;
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
        super::memory::before_load(memory.total)?;
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
            None => Adapter::initialize_targets(
                config.rank,
                config.alpha,
                config.seed,
                config.targets,
            )?,
        };
        let mut models =
            Models::load_parts(&mut stream, &config.model, None, None, None, None)?;
        models.ops = crate::ops::Ops::new(std::sync::Arc::new(hrx::BufferPool::with_limit(
            config.scratch_pool_mib << 20,
        )));
        let model = Transformer::load(&models.ops, &mut stream, &config.model, Some(&adapter))?;
        super::memory::during_run()?;
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
        let parameters: Vec<_> = model.adapters.values().flat_map(|p| [&p.a, &p.b]).collect();
        let optimizer = optimizer::PreparedOptimizer::new(&stream, &parameters)?;
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ 0x1234_5678_abcd_ef01);
        rng.set_word_pos(state.rng_word.parse().map_err(io)?);
        let order = order(data.samples.len(), config.seed, state.epoch);
        Ok(Self {
            config,
            data,
            models,
            model,
            optimizer,
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
        super::memory::during_run()?;
        ops::clear_host_timings();
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
        let optimizer_started = Instant::now();
        let norm = self.optimizer.update(&mut self.stream, &self.config, next)?;
        self.stream.synchronize()?;
        if crate::kernels::native_profile() {
            eprintln!(
                "training stage optimizer: {:.3} ms",
                optimizer_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.state.step = next;
        ops::report_host_timings();
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
        let text = textfile.get(prepare::text_spec(&self.config).0)?;
        let textbits: Vec<u16> =
            text.bytes.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect();
        let text =
            Tensor::from_slice(ops.pool(), stream, &textbits, text.shape[0], text.shape[1])?;
        let full = self.config.targets == Targets::All;
        let mut auxiliary = Tape::new(&self.models, &self.model, 1.0);
        let (text, embedding, modvec, image, roots) = if full {
            let text_id = auxiliary.text(stream, &text)?;
            let (embedding_id, mod_id) = auxiliary.time(stream, sigma)?;
            let image_id = auxiliary.image(stream, &noisy)?;
            (
                auxiliary.value(text_id).clone(),
                auxiliary.value(embedding_id).clone(),
                auxiliary.value(mod_id).clone(),
                auxiliary.value(image_id).clone(),
                Some((text_id, embedding_id, mod_id, image_id)),
            )
        } else {
            let (embedding, modvec) = self.models.time_continuous(stream, sigma)?;
            (text, embedding, modvec, self.models.image_in(stream, &noisy)?, None)
        };
        let mod_grad = if full { Some(FloatTensor::zero(stream, 6, 6144)?) } else { None };
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
        let mut tapes = Vec::with_capacity(28);
        let profile = crate::kernels::native_profile();
        let forward_started = Instant::now();
        for block in 0..28 {
            super::memory::during_run()?;
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
            boundaries.push(tape.output.clone());
            if !self.config.gradient_checkpointing {
                tapes.push(Some(tape));
            }
            // Limit pending buffer ownership to one block. Dropping a host tensor
            // does not release storage still referenced by queued dispatches.
            stream.synchronize()?;
        }
        if profile {
            eprintln!(
                "training stage forward: {:.3} ms",
                forward_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let last = boundaries.last().expect("output").view(image.rows(), 6144, text.size())?;
        let mut final_tape = Tape::new(&self.models, &self.model, 1.0);
        let final_input = final_tape.input(&last);
        let final_embedding = final_tape.input(&embedding);
        let final_output = if full {
            Some(final_tape.last(stream, final_input, final_embedding)?)
        } else {
            None
        };
        let prediction = match final_output {
            Some(id) => final_tape.value(id).clone(),
            None => self.models.last(stream, &last, &embedding)?,
        };
        let target = FloatTensor::from_slice(stream, image.rows(), 64, &target)?;
        let (loss, gradient) = ops::flow_loss(ops, stream, &prediction, &target)?;
        let (image_grad, embedding_grad) = if let Some(id) = final_output {
            let mut grads = final_tape.backward(stream, &[(id, gradient)])?;
            (
                grads[final_input]
                    .take()
                    .ok_or_else(|| Error::internal("missing final input gradient"))?,
                grads[final_embedding].take(),
            )
        } else {
            (last_backward(&self.models, stream, &last, &embedding, &gradient)?, None)
        };
        drop(final_tape);
        let mut grad = ops.tensor(stream, tokens, 6144)?;
        grad.zero(stream)?;
        stream.copy(
            grad.binding()?.slice(text.size() * 2, image.size() * 2)?,
            image_grad.binding()?,
        )?;
        let mut recompute_ms = 0.0;
        let mut backward_ms = 0.0;
        for block in (0..28).rev() {
            super::memory::during_run()?;
            boundaries.pop();
            let input = boundaries.last().expect("saved block input");
            let m = mods.view(6, 6144, block * 6 * 6144)?;
            let started = Instant::now();
            let tape = if self.config.gradient_checkpointing {
                self.model.block(ops, stream, block, input, &m, &cos, &sin, 1.0)?
            } else {
                tapes[block].take().expect("saved block tape")
            };
            if profile {
                stream.synchronize()?;
                recompute_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
            let started = Instant::now();
            grad = self.model.backward_with_modulation(
                ops,
                stream,
                block,
                tape,
                &m,
                &cos,
                &sin,
                &grad,
                mod_grad.as_ref(),
            )?;
            stream.synchronize()?;
            if profile {
                backward_ms += started.elapsed().as_secs_f64() * 1000.0;
            }
        }
        if profile {
            eprintln!(
                "training stage recompute: {recompute_ms:.3} ms; backward: {backward_ms:.3} ms"
            );
        }
        if let Some((text_id, embedding_id, mod_id, image_id)) = roots {
            let mod_gradient =
                ops::cast(ops, stream, mod_grad.as_ref().expect("full target modulation"))?
                    .view(1, 6 * 6144, 0)?;
            auxiliary.backward(
                stream,
                &[
                    (text_id, grad.view(text.rows(), 6144, 0)?),
                    (image_id, grad.view(image.rows(), 6144, text.size())?),
                    (
                        embedding_id,
                        embedding_grad
                            .ok_or_else(|| Error::internal("missing timestep gradient"))?,
                    ),
                    (mod_id, mod_gradient),
                ],
            )?;
            stream.synchronize()?;
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
                (
                    "targets".into(),
                    match self.config.targets {
                        Targets::All => "all",
                        Targets::MainBlocks => "main_blocks",
                    }
                    .into(),
                ),
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
    for (key, outputs, inputs) in c.targets.layers() {
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
    fn resident_memory_counts_execution_copies_without_duplicating_bf16_weights() {
        use crate::checkpoint::DType;
        assert_eq!(resident_weight_bytes(DType::BF16, 256 << 20, false).unwrap(), 256 << 20);
        assert_eq!(resident_weight_bytes(DType::F32, 256 << 20, false).unwrap(), 384 << 20);
        assert_eq!(resident_weight_bytes(DType::BF16, 256, true).unwrap(), 256 + 4096);
        assert_eq!(resident_weight_bytes(DType::F32, 48, false).unwrap(), 256 + 4096);
        assert!(resident_weight_bytes(DType::I8, 256, false).is_err());
        assert!(resident_weight_bytes(DType::BF16, usize::MAX, true).is_err());
        assert!(resident_weight_bytes(DType::BF16, usize::MAX, false).is_err());
    }
    #[test]
    fn full_target_resume_restores_every_master_and_rejects_missing_auxiliary_factors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("optimizer.safetensors");
        let c = TrainConfig {
            targets: Targets::All,
            rank: 1,
            alpha: 1.0,
            ..TrainConfig::default()
        };
        let mut tensors = BTreeMap::new();
        for (name, outputs, inputs) in c.targets.layers() {
            for (part, rows, cols, value) in
                [("a", 1, inputs, 0.125f32), ("b", outputs, 1, 0.25f32)]
            {
                tensors.insert(
                    format!("{name}.{part}.master"),
                    SavedTensor {
                        dtype: "F32",
                        shape: vec![rows, cols],
                        bytes: value.to_le_bytes().repeat(rows * cols),
                    },
                );
            }
        }
        save_tensors(&path, tensors.clone(), BTreeMap::new()).unwrap();
        let restored = read_master_adapter(&path, &c).unwrap();
        assert_eq!(restored.layers.len(), 264);
        for factors in restored.layers.values() {
            assert!(factors.a.iter().all(|v| *v == 0.125));
            assert!(factors.b.iter().all(|v| *v == 0.25));
        }
        tensors.remove("txtfusion.projector.b.master");
        save_tensors(&path, tensors, BTreeMap::new()).unwrap();
        assert!(read_master_adapter(&path, &c).is_err());
    }
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
