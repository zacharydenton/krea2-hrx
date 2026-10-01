//! Authoritative parameter buffers shared by forward execution and the optimizer.
use super::{
    numerics,
    spec::{self, ParameterSpec},
};
use crate::training::{
    TrainConfig,
    ops::{self as train, FloatTensor},
};
use crate::{
    Error, Result,
    checkpoint::{Checkpoint, DType},
    kernels::Scalars,
    models::Weights,
    numerics::{from_f32_carrying, to_f32},
    ops::{Ops, Tensor, Weight, config},
};
use hrx::{Buffer, Stream};
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct Parameter {
    pub spec: ParameterSpec,
    pub value: Arc<Buffer>,
    pub master: Option<Arc<Buffer>>,
    pub grad: Arc<Buffer>,
    pub first: Arc<Buffer>,
    pub second: Arc<Buffer>,
    pub scales: Option<Arc<Buffer>>,
    pub key: [u32; 2],
}

/// Complete model state. All forward views alias these authoritative buffers.
pub struct Parameters {
    pub(crate) values: BTreeMap<String, Parameter>,
    clock: Buffer,
    maps: Buffer,
    controls: Buffer,
    partials: FloatTensor,
    // One stream-ordered dW accumulator, reused by every large projection.
    gradient_scratch: Arc<Buffer>,
}

fn allocate(stream: &Stream, bytes: usize) -> Result<Arc<Buffer>> {
    let b = Arc::new(stream.allocate(bytes)?);
    stream.fill(b.binding(), 0)?;
    Ok(b)
}

impl Parameters {
    pub(crate) fn load(
        stream: &mut Stream,
        file: &Checkpoint,
        seed: u64,
    ) -> Result<(Self, Weights, Weights)> {
        let schema = spec::validate(file)?;
        Self::load_schema(stream, file, seed, schema)
    }

    fn load_schema(
        stream: &mut Stream,
        file: &Checkpoint,
        seed: u64,
        schema: BTreeMap<String, ParameterSpec>,
    ) -> Result<(Self, Weights, Weights)> {
        let mut values = BTreeMap::new();
        let mut main = BTreeMap::new();
        let mut auxiliary = BTreeMap::new();
        for (name, spec) in schema {
            let t = file.get(&name)?;
            let n = spec.count;
            let value = allocate(stream, n * 2)?;
            let master = if spec.small { Some(allocate(stream, n * 4)?) } else { None };
            // Keep conversion bounded even for the 864 MiB FP32 modulation projection.
            let chunk = 8 * 1024 * 1024;
            let width = if t.dtype == DType::F32 { 4 } else { 2 };
            for (part, bytes) in t.bytes.chunks(chunk).enumerate() {
                let start = part * chunk / width;
                let floats: Vec<f32> = if width == 4 {
                    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
                } else {
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| to_f32(u16::from_le_bytes(*b)))
                        .collect()
                };
                let bits: Vec<_> = floats.iter().copied().map(from_f32_carrying).collect();
                if floats.iter().any(|v| !v.is_finite())
                    || bits.iter().any(|&b| !to_f32(b).is_finite())
                {
                    return Err(Error::invalid(format!(
                        "nonfinite full-training weight {name}"
                    )));
                }
                stream.upload(
                    value.try_slice(start * 2, bits.len() * 2)?,
                    bytemuck::cast_slice(&bits),
                )?;
                if let Some(m) = &master {
                    stream.upload(
                        m.try_slice(start * 4, floats.len() * 4)?,
                        bytemuck::cast_slice(&floats),
                    )?;
                }
            }
            let weight = Weight::new(
                &value,
                0,
                spec.shape.clone(),
                n,
                master.as_ref().map(|b| (Arc::clone(b), 0)),
            );
            if name.starts_with("blocks.") && !name.ends_with(".mod.lin") {
                main.insert(name.clone(), weight);
            } else {
                auxiliary.insert(name.clone(), weight);
            }
            let hash = blake3::hash(format!("full-parameter-v1:{seed}:{name}").as_bytes());
            let key = [
                u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap()),
                u32::from_le_bytes(hash.as_bytes()[4..8].try_into().unwrap()),
            ];
            let p = Parameter {
                grad: allocate(stream, n * if spec.small { 4 } else { 2 })?,
                first: allocate(stream, n * if spec.small { 4 } else { 1 })?,
                second: allocate(stream, n * if spec.small { 4 } else { 1 })?,
                scales: if spec.small {
                    None
                } else {
                    Some(allocate(stream, n.div_ceil(numerics::BLOCK) * 8)?)
                },
                spec,
                value,
                master,
                key,
            };
            values.insert(name, p);
            stream.synchronize()?;
        }
        let maps = stream.allocate(512 * 4)?;
        let codes: Vec<f32> =
            numerics::codebook(true).into_iter().chain(numerics::codebook(false)).collect();
        stream.upload(maps.binding(), bytemuck::cast_slice(&codes))?;
        let partials = FloatTensor::zero(
            stream,
            1,
            values.values().map(|p| p.spec.count.div_ceil(1024)).sum(),
        )?;
        let gradient_scratch = allocate(
            stream,
            values
                .values()
                .filter(|p| !p.spec.small)
                .map(|p| p.spec.count * 4)
                .max()
                .unwrap_or(4),
        )?;
        Ok((
            Self {
                values,
                clock: stream.allocate(16)?,
                maps,
                gradient_scratch,
                controls: stream.allocate(32)?,
                partials,
            },
            Weights::from_values(main),
            Weights::from_values(auxiliary),
        ))
    }

    pub(crate) fn set_clock(
        &self,
        stream: &mut Stream,
        step: usize,
        microbatch: usize,
        update: bool,
    ) -> Result<()> {
        let microbatch = u32::try_from(microbatch)
            .map_err(|_| Error::invalid("full-training accumulation too large"))?;
        let clock = [
            step as u32,
            (step as u64 >> 32) as u32,
            microbatch,
            if update { 0x57454947 } else { 0x47524144 },
        ];
        stream.upload(self.clock.binding(), bytemuck::cast_slice(&clock))?;
        Ok(())
    }

    pub(crate) fn gradient_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<FloatTensor> {
        let p = self.get(name)?;
        if !p.spec.small || rows.checked_mul(cols) != Some(p.spec.count) {
            return Err(Error::invalid("expected an FP32 small-parameter gradient"));
        }
        FloatTensor::shared(&p.grad, rows, cols)
    }

    pub(crate) fn add_gradient_slice(
        &self,
        ops: &Ops,
        stream: &Stream,
        name: &str,
        offset: usize,
        grad: &Tensor,
    ) -> Result<()> {
        let p = self.get(name)?;
        if !p.spec.small || offset.checked_add(grad.cols()).is_none_or(|n| n > p.spec.count) {
            return Err(Error::invalid("parameter gradient slice dimensions"));
        }
        // SAFETY: the row reduction writes only the selected contiguous FP32 parameter slice.
        unsafe {
            ops.launch_1d(
                stream,
                "train_sum_rows",
                config(&[("rows", grad.rows()), ("cols", grad.cols()), ("size", grad.size())]),
                &Scalars::new().index(grad.cols()),
                &[grad.binding()?, p.grad.try_slice(offset * 4, grad.cols() * 4)?],
                grad.cols(),
            )
        }
    }

    fn get(&self, name: &str) -> Result<&Parameter> {
        self.values
            .get(name)
            .ok_or_else(|| Error::invalid(format!("unknown trainable parameter {name}")))
    }

    pub(crate) fn norm_gradient(
        &self,
        ops: &Ops,
        stream: &Stream,
        name: &str,
        x: &Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        let p = self.get(name)?;
        let cols = p.spec.count;
        if x.size() != grad.size() || !x.size().is_multiple_of(cols) {
            return Err(Error::invalid("normalization parameter gradient dimensions"));
        }
        let rows = x.size() / cols;
        let inverse = FloatTensor::scratch(ops, stream, rows, 1)?;
        let configs = config(&[("rows", rows), ("cols", cols), ("size", x.size())]);
        // SAFETY: one subgroup computes each row's RMS inverse; the second kernel owns each scale gradient.
        unsafe {
            ops.launch(
                stream,
                "train_full_norm_inv",
                configs.clone(),
                &Scalars::new().index(rows),
                &[x.binding()?, inverse.binding()],
                rows,
                1,
                32,
            )?;
            ops.launch_1d(
                stream,
                "train_full_norm_scale",
                configs,
                &Scalars::new().index(cols),
                &[x.binding()?, grad.binding()?, inverse.binding(), p.grad.binding()],
                cols,
            )?;
        }
        Ok(())
    }

    pub(crate) fn linear_gradient(
        &self,
        ops: &Ops,
        stream: &Stream,
        name: &str,
        x: &Tensor,
        dy: &Tensor,
    ) -> Result<()> {
        let p = self.get(&format!("{name}.weight"))?;
        let (o, i) = (dy.cols(), x.cols());
        if p.spec.shape != [o, i] {
            return Err(Error::invalid("full linear gradient dimensions"));
        }
        let gradient = if p.spec.small {
            FloatTensor::shared(&p.grad, o, i)?
        } else {
            let fp = FloatTensor::shared(&self.gradient_scratch, o, i)?;
            // SAFETY: matching BF16 gradient and FP32 scratch have exactly o*i elements.
            unsafe {
                ops.launch_1d(
                    stream,
                    "train_full_upcast",
                    config(&[]),
                    &Scalars::new().index(p.spec.count),
                    &[p.grad.binding(), fp.binding()],
                    p.spec.count,
                )?;
            }
            fp
        };
        train::matmul_tn_accumulate(ops, stream, dy, x, &gradient, 1.0)?;
        if !p.spec.small {
            // SAFETY: one stochastic store per gradient element, with a stable Philox clock/key.
            unsafe {
                ops.launch_1d(
                    stream,
                    "train_full_store",
                    config(&[]),
                    &Scalars::new()
                        .index(p.spec.count)
                        .index(p.key[0] as usize)
                        .index(p.key[1] as usize),
                    &[gradient.binding(), p.grad.binding(), self.clock.binding()],
                    p.spec.count,
                )?;
            }
        }
        if let Some(bias) = self.values.get(&format!("{name}.bias")) {
            let dst = FloatTensor::shared(&bias.grad, 1, bias.spec.count)?;
            train::sum_rows_accumulate(ops, stream, dy, &dst, 0)?;
        }
        Ok(())
    }

    pub(crate) fn update(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        c: &TrainConfig,
        step: usize,
    ) -> Result<f64> {
        let mut offset = 0;
        for p in self.values.values() {
            let parts = p.spec.count.div_ceil(1024);
            let name = if p.spec.small { "train_grad_norm" } else { "train_full_grad_norm" };
            // SAFETY: each parameter writes a disjoint, bounded slice of norm partials.
            unsafe {
                ops.launch(
                    stream,
                    name,
                    config(&[("parts", parts)]),
                    &Scalars::new().index(p.spec.count),
                    &[p.grad.binding(), self.partials.binding().slice(offset * 4, parts * 4)?],
                    parts,
                    1,
                    32,
                )?;
            }
            offset += parts;
        }
        let partials = self.partials.download(stream)?;
        let (norm, controls) = crate::training::optimizer::update_controls(&partials, c, step)?;
        stream.upload(self.controls.binding(), bytemuck::cast_slice(&controls))?;
        self.set_clock(stream, step, 0, true)?;
        for p in self.values.values() {
            if let Some(master) = &p.master {
                // SAFETY: this small parameter owns matching FP32 master/gradient/moments and a BF16 execution copy.
                unsafe {
                    ops.launch_1d(
                        stream,
                        "train_adamw_graph",
                        config(&[]),
                        &Scalars::new().index(p.spec.count),
                        &[
                            self.controls.binding(),
                            master.binding(),
                            p.grad.binding(),
                            p.first.binding(),
                            p.second.binding(),
                            p.value.binding(),
                        ],
                        p.spec.count,
                    )?;
                }
            } else {
                let parts = p.spec.count.div_ceil(numerics::BLOCK);
                // SAFETY: each subgroup owns one 256-element block and its two scales; codebooks and clock are read-only.
                unsafe {
                    ops.launch(
                        stream,
                        "train_full_adamw",
                        config(&[("parts", parts)]),
                        &Scalars::new()
                            .index(p.spec.count)
                            .index(p.key[0] as usize)
                            .index(p.key[1] as usize),
                        &[
                            self.controls.binding(),
                            p.value.binding(),
                            p.grad.binding(),
                            p.first.binding(),
                            p.second.binding(),
                            p.scales.as_ref().expect("quantized scales").binding(),
                            self.maps.binding(),
                            self.clock.binding(),
                        ],
                        parts,
                        1,
                        32,
                    )?;
                }
            }
        }
        Ok(norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{SavedTensor, save_tensors};
    fn fixture(path: &std::path::Path) -> BTreeMap<String, ParameterSpec> {
        let schema: BTreeMap<String, ParameterSpec> = [
            ("dense.weight", vec![65, 67], false),
            ("dense.bias", vec![65], true),
            ("norm.scale", vec![128], true),
            ("mod.lin", vec![12], true),
        ]
        .into_iter()
        .map(|(name, shape, small)| {
            (name.into(), ParameterSpec { count: shape.iter().product(), shape, small })
        })
        .collect();
        let tensors = schema
            .iter()
            .map(|(name, p)| {
                (
                    name.clone(),
                    SavedTensor {
                        dtype: if p.small { "F32" } else { "BF16" },
                        shape: p.shape.clone(),
                        bytes: if p.small {
                            0.0012345f32.to_le_bytes().repeat(p.count)
                        } else {
                            crate::numerics::from_f32(0.01).to_le_bytes().repeat(p.count)
                        },
                    },
                )
            })
            .collect();
        save_tensors(path, tensors, BTreeMap::new()).unwrap();
        schema
    }
    fn snapshot(p: &Parameters, s: &mut Stream) -> BTreeMap<String, Vec<u8>> {
        let mut result = BTreeMap::new();
        for (name, p) in &p.values {
            for (kind, b) in [
                ("value", Some(&p.value)),
                ("master", p.master.as_ref()),
                ("grad", Some(&p.grad)),
                ("first", Some(&p.first)),
                ("second", Some(&p.second)),
                ("scales", p.scales.as_ref()),
            ] {
                if let Some(b) = b {
                    let mut bytes = vec![0; b.binding().len()];
                    s.read_blocking(b.binding(), &mut bytes).unwrap();
                    result.insert(format!("{name}.{kind}"), bytes);
                }
            }
        }
        result
    }
    #[test]
    #[ignore = "requires gfx1151"]
    fn full_parameter_accumulation_resume_and_nonfinite_rejection() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let schema = fixture(&base);
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(hrx::BufferPool::new());
        let (p, _, _) = Parameters::load_schema(
            &mut stream,
            &Checkpoint::open(&base).unwrap(),
            37,
            schema.clone(),
        )
        .unwrap();
        let x: Vec<_> = (0..3 * 67)
            .map(|i| crate::numerics::from_f32(((i % 11) as f32 - 5.0) * 0.03))
            .collect();
        let dy: Vec<_> = (0..3 * 65)
            .map(|i| crate::numerics::from_f32(((i % 7) as f32 - 3.0) * 0.01))
            .collect();
        let x = Tensor::from_slice(ops.pool(), &mut stream, &x, 3, 67).unwrap();
        let dy = Tensor::from_slice(ops.pool(), &mut stream, &dy, 3, 65).unwrap();
        let c = TrainConfig { accumulation: 2, learning_rate: 0.001, ..TrainConfig::default() };
        let step = |p: &Parameters, s: &mut Stream, step| {
            for micro in 0..2 {
                p.set_clock(s, step, micro, false).unwrap();
                p.linear_gradient(&ops, s, "dense", &x, &dy).unwrap();
            }
            p.update(&ops, s, &c, step).unwrap();
            s.synchronize().unwrap();
        };
        step(&p, &mut stream, 1);
        let saved = dir.path().join("saved");
        std::fs::create_dir(&saved).unwrap();
        let digest = p.save(&mut stream, &saved).unwrap();
        let before_export = snapshot(&p, &mut stream);
        let model_only = dir.path().join("model-only");
        std::fs::create_dir(&model_only).unwrap();
        assert_eq!(p.save_model(&mut stream, &model_only).unwrap(), digest);
        assert!(!model_only.join("optimizer.safetensors").exists());
        assert_eq!(snapshot(&p, &mut stream), before_export);
        let (resumed, _, _) = Parameters::load_schema(
            &mut stream,
            &Checkpoint::open(&saved.join("model.safetensors")).unwrap(),
            37,
            schema,
        )
        .unwrap();
        resumed.restore(&mut stream, &saved.join("optimizer.safetensors")).unwrap();
        assert_eq!(snapshot(&p, &mut stream), snapshot(&resumed, &mut stream));
        step(&p, &mut stream, 2);
        step(&resumed, &mut stream, 2);
        assert_eq!(snapshot(&p, &mut stream), snapshot(&resumed, &mut stream));
        let before = snapshot(&p, &mut stream);
        stream
            .upload(
                p.values["dense.bias"].grad.try_slice(0, 4).unwrap(),
                &f32::NAN.to_le_bytes(),
            )
            .unwrap();
        assert!(p.update(&ops, &mut stream, &c, 3).is_err());
        let after = snapshot(&p, &mut stream);
        for (name, bytes) in &before {
            if !name.ends_with(".grad") {
                assert_eq!(bytes, &after[name], "partial update: {name}");
            }
        }
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::{
        models::Models,
        numerics::{from_f32, to_f32},
        training::{auxiliary::Tape, model::Transformer},
    };
    #[test]
    #[ignore = "requires gfx1151 and KREA2_RAW_CHECKPOINT; loads one block and conditioning towers"]
    fn complete_parameter_families_receive_gradients_and_refresh_forward_views() {
        crate::training::memory::before_load(16usize << 30).unwrap();
        let path = std::env::var_os("KREA2_RAW_CHECKPOINT").expect("set KREA2_RAW_CHECKPOINT");
        let file = Checkpoint::open(std::path::Path::new(&path)).unwrap();
        let mut schema = spec::validate(&file).unwrap();
        schema.retain(|name, _| {
            !name.starts_with("blocks.")
                || name.starts_with("blocks.0.")
                || name.ends_with(".mod.lin")
        });
        let mut stream = Stream::open().unwrap();
        let (parameters, weights, auxiliary) =
            Parameters::load_schema(&mut stream, &file, 37, schema).unwrap();
        let ops = Ops::new(std::sync::Arc::new(hrx::BufferPool::with_limit(1 << 30)));
        let models = Models::trainable(&stream, ops, auxiliary).unwrap();
        let model = Transformer::full(weights, parameters);
        let full = model.full.as_ref().unwrap();
        let upload = |s: &mut Stream, rows, cols, seed| {
            let bits: Vec<_> = (0..rows * cols)
                .map(|i| from_f32(((i * 17 + seed) % 71) as f32 / 71.0 - 0.5))
                .collect();
            Tensor::from_slice(models.ops.pool(), s, &bits, rows, cols).unwrap()
        };
        let taps = upload(&mut stream, 36, 2560, 7);
        let latent = upload(&mut stream, 5, 64, 13);
        let hidden = upload(&mut stream, 5, 6144, 19);
        full.set_clock(&mut stream, 1, 0, false).unwrap();
        let mut tape = Tape::new(&models, &model, 1.0);
        let text = tape.text(&mut stream, &taps).unwrap();
        let (embedding, modulation) = tape.time(&mut stream, 0.501).unwrap();
        let image = tape.image(&stream, &latent).unwrap();
        let input = tape.input(&hidden);
        let last = tape.last(&mut stream, input, embedding).unwrap();
        let before = tape.value(last).download(&mut stream).unwrap();
        let mut seeds = Vec::new();
        for id in [text, modulation, image, last] {
            seeds.push((
                id,
                upload(&mut stream, tape.value(id).rows(), tape.value(id).cols(), id + 3),
            ));
        }
        tape.backward(&mut stream, &seeds).unwrap();
        let mods = models.modulation(&stream, tape.value(modulation)).unwrap();
        let mods =
            train::cast_view(&models.ops, &stream, mods.binding(), 28 * 6, 6144).unwrap();
        let m = mods.view(6, 6144, 0).unwrap();
        let cos = FloatTensor::from_slice(&mut stream, 5, 128, &[1.0; 5 * 128]).unwrap();
        let sin = FloatTensor::zero(&stream, 5, 128).unwrap();
        let block =
            model.block(&models.ops, &mut stream, 0, &hidden, &m, &cos, &sin, 1.0).unwrap();
        let grad = upload(&mut stream, 5, 6144, 29);
        model.backward(&models.ops, &mut stream, 0, block, &m, &cos, &sin, &grad).unwrap();
        stream.synchronize().unwrap();
        for (name, p) in &full.values {
            if name.starts_with("blocks.") && !name.starts_with("blocks.0.") {
                continue;
            }
            let mut nonzero = false;
            for offset in (0..p.spec.count).step_by(1 << 20) {
                let n = (p.spec.count - offset).min(1 << 20);
                let width = if p.spec.small { 4 } else { 2 };
                let mut bytes = vec![0; n * width];
                stream
                    .read_blocking(
                        p.grad.try_slice(offset * width, n * width).unwrap(),
                        &mut bytes,
                    )
                    .unwrap();
                let values: Vec<f32> = if p.spec.small {
                    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
                } else {
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| to_f32(u16::from_le_bytes(*b)))
                        .collect()
                };
                assert!(values.iter().all(|v| v.is_finite()), "{name} nonfinite");
                nonzero |= values.iter().any(|v| *v != 0.0);
            }
            assert!(nonzero, "{name} disconnected");
        }
        drop(tape);
        let c = TrainConfig { learning_rate: 1e-3, ..TrainConfig::default() };
        full.update(&models.ops, &mut stream, &c, 1).unwrap();
        models.tables(&stream).unwrap();
        let mut after = Tape::new(&models, &model, 1.0);
        let (embedding, _) = after.time(&mut stream, 0.501).unwrap();
        let input = after.input(&hidden);
        let last = after.last(&mut stream, input, embedding).unwrap();
        assert_ne!(before, after.value(last).download(&mut stream).unwrap());
        // The gathered table must equal the freshly updated execution copy with zero conditioning.
        let zero = models.ops.tensor(&stream, 1, 36864).unwrap();
        zero.zero(&mut stream).unwrap();
        let packed = models.modulation(&stream, &zero).unwrap();
        let mut table = vec![0u16; 36864];
        stream
            .read_blocking(
                full.values["blocks.0.mod.lin"].value.binding(),
                bytemuck::cast_slice_mut(&mut table),
            )
            .unwrap();
        let mut actual = vec![0f32; 36864];
        stream
            .read_blocking(
                packed.binding().slice(0, 36864 * 4).unwrap(),
                bytemuck::cast_slice_mut(&mut actual),
            )
            .unwrap();
        assert_eq!(actual, table.into_iter().map(to_f32).collect::<Vec<_>>());
        let norm = &full.values["last.norm.scale"];
        let mut master = vec![0u8; norm.spec.count * 4];
        let mut view = master.clone();
        stream.read_blocking(norm.master.as_ref().unwrap().binding(), &mut master).unwrap();
        let scale =
            models.transformer.get("last.norm.scale").unwrap().f32_values(&mut stream).unwrap();
        stream.read_blocking(scale, &mut view).unwrap();
        assert_eq!(master, view);
    }
}
