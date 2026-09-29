//! Adapter-enabled W8A8 projections with a separate original-basis LoRA branch.
use crate::checkpoint::{Checkpoint, DType};
use crate::kernels::{Scalars, shape};
use crate::ops::{Ops, Tensor, config};
use crate::{Error, Result};
use hrx::{Buffer, Stream};

pub(crate) struct Quantized {
    weight: Buffer,
    scales: Buffer,
    inputs: usize,
    outputs: usize,
    pitch: usize,
}

impl Quantized {
    pub fn load(
        stream: &mut Stream,
        file: &Checkpoint,
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Self> {
        let tensor = file.get(&format!("{name}.weight"))?;
        let scale = file.get(&format!("{name}.weight_scale"))?;
        if tensor.dtype != DType::I8
            || tensor.shape != [outputs, inputs]
            || scale.dtype != DType::F32
            || scale.bytes.len() != outputs * 4
        {
            return Err(Error::invalid(format!("invalid ConvRot projection {name}")));
        }
        let pitch = shape::gemm_pitch(inputs);
        let weight = stream.allocate(outputs * pitch)?;
        let scales = stream.allocate(outputs * 4)?;
        stream.upload(scales.binding(), scale.bytes)?;
        let rows_per_chunk = (16usize << 20) / pitch;
        for first in (0..outputs).step_by(rows_per_chunk) {
            let count = (outputs - first).min(rows_per_chunk);
            let mut bytes = vec![0u8; count * pitch];
            for row in 0..count {
                bytes[row * pitch..row * pitch + inputs].copy_from_slice(
                    &tensor.bytes[(first + row) * inputs..(first + row + 1) * inputs],
                );
            }
            stream.upload(weight.try_slice(first * pitch, bytes.len())?, &bytes)?;
        }
        Ok(Self { weight, scales, inputs, outputs, pitch })
    }

    pub fn forward(&self, ops: &Ops, stream: &Stream, x: &Tensor) -> Result<Tensor> {
        if x.cols() != self.inputs {
            return Err(Error::invalid("ConvRot projection dimensions"));
        }
        let transport = ops.pool().acquire(stream, x.size() * 2)?;
        let quantized = ops.pool().acquire(stream, x.rows() * self.pitch)?;
        let scales = ops.pool().acquire(stream, x.rows() * 4)?;
        let out_f16 = ops.pool().acquire(stream, x.rows() * self.outputs * 2)?;
        let out = ops.tensor(stream, x.rows(), self.outputs)?;
        // SAFETY: every binding matches the configured rows, pitches and output columns.
        // Conversion preserves the original-basis x used independently by the adapter.
        unsafe {
            ops.launch_1d(
                stream,
                "lora_transport",
                config(&[("reverse", 1)]),
                &Scalars::new().index(x.size()),
                &[x.binding()?, transport.binding()],
                x.size(),
            )?;
            ops.launch(
                stream,
                "prepare_plain_i8",
                config(&[("width", self.inputs), ("out_stride", self.pitch)]),
                &Scalars::new().index(x.rows()),
                &[transport.binding(), quantized.binding(), scales.binding()],
                x.rows(),
                1,
                256,
            )?;
            ops.launch(
                stream,
                "gemm_i8_256",
                config(&[
                    ("k_size", self.inputs),
                    ("k_stride", self.pitch),
                    ("n_size", self.outputs),
                    ("m_group", shape::GEMM_M_GROUP),
                ]),
                &Scalars::new().index(x.rows()),
                &[
                    quantized.binding(),
                    self.weight.binding(),
                    self.scales.binding(),
                    scales.binding(),
                    out_f16.binding(),
                ],
                self.outputs / 128,
                shape::gemm_grid_rows(x.rows()),
                256,
            )?;
            ops.launch_1d(
                stream,
                "lora_transport",
                config(&[("reverse", 2)]),
                &Scalars::new().index(out.size()),
                &[out_f16.binding(), out.binding()?],
                out.size(),
            )?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{Factors, SavedTensor, save_tensors};
    use crate::numerics::{from_f32, to_f32};
    use crate::training::model::Projection;
    use half::f16;
    use std::collections::BTreeMap;

    #[test]
    #[ignore = "requires a gfx1151 GPU; no model weights needed"]
    fn convrot_adapter_path_matches_integer_and_original_basis_cpu_algebra() {
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(hrx::BufferPool::new());
        let directory = tempfile::tempdir().unwrap();
        let h4 = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
        for inputs in [6144usize, 16384] {
            let (tokens, outputs) = (3, 128);
            let weights: Vec<i8> =
                (0..outputs * inputs).map(|i| ((i * 37 % 31) as i32 - 15) as i8).collect();
            let scales: Vec<f32> = (0..outputs).map(|i| (i % 7 + 1) as f32 / 256.0).collect();
            let path = directory.path().join(format!("{inputs}.safetensors"));
            save_tensors(
                &path,
                [
                    (
                        "projection.weight".into(),
                        SavedTensor {
                            dtype: "I8",
                            shape: vec![outputs, inputs],
                            bytes: weights.iter().map(|v| *v as u8).collect(),
                        },
                    ),
                    (
                        "projection.weight_scale".into(),
                        SavedTensor {
                            dtype: "F32",
                            shape: vec![outputs],
                            bytes: scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        },
                    ),
                ]
                .into(),
                BTreeMap::new(),
            )
            .unwrap();
            let file = Checkpoint::open(&path).unwrap();
            let quantized =
                Quantized::load(&mut stream, &file, "projection", outputs, inputs).unwrap();
            let x: Vec<u16> = (0..tokens * inputs)
                .map(|i| {
                    from_f32(if i < inputs {
                        0.0
                    } else {
                        ((i * 17 % 97) as f32 - 48.0) / 128.0
                    })
                })
                .collect();
            let input =
                Tensor::from_slice(ops.pool(), &mut stream, &x, tokens, inputs).unwrap();
            let base = quantized.forward(&ops, &stream, &input).unwrap();
            let factors = Factors {
                inputs,
                outputs,
                rank: 2,
                alpha: 1.0,
                a: vec![0.03125; 2 * inputs],
                b: vec![0.125; 2 * outputs],
            };
            let adapter = Projection::new(&ops, &mut stream, &factors).unwrap();
            let output = adapter.forward(&ops, &stream, &input, &base, 0.75).unwrap();
            let actual = output.download(&mut stream).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for row in x.chunks_exact(inputs) {
                let low = to_f32(from_f32(
                    row.iter().map(|x| f64::from(to_f32(*x)) / 32.0).sum::<f64>() as f32,
                ));
                let delta = to_f32(from_f32(low * 0.25));
                let mut rotated: Vec<f16> =
                    row.iter().map(|v| f16::from_f32(to_f32(*v))).collect();
                for step in [1, 4, 16, 64] {
                    for block in (0..inputs).step_by(4 * step) {
                        for col in 0..step {
                            let x: Vec<_> = (0..4)
                                .map(|j| rotated[block + col + j * step].to_f64())
                                .collect();
                            for i in 0..4 {
                                let sum: f64 = (0..4).map(|j| h4[i][j] * x[j]).sum();
                                rotated[block + col + i * step] = f16::from_f64(sum * 0.25);
                            }
                        }
                    }
                }
                let max = rotated.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                let scale = (max * 16.0).max(1e-30) / 127.0;
                let codes: Vec<i32> = rotated
                    .iter()
                    .map(|v| {
                        (v.to_f32() * (16.0 / scale)).round_ties_even().clamp(-127.0, 127.0)
                            as i32
                    })
                    .collect();
                for (col, weight) in weights.chunks_exact(inputs).enumerate() {
                    let dot: i32 =
                        codes.iter().zip(weight).map(|(a, b)| *a * i32::from(*b)).sum();
                    let base = to_f32(from_f32((dot as f32 * scales[col]) * scale));
                    expected.push(from_f32(base + 0.375 * delta));
                }
            }
            assert_eq!(actual, expected, "ConvRot plus original-basis adapter, width {inputs}");
        }
    }
}
