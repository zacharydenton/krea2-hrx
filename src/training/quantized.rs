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
