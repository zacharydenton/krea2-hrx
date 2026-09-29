//! Deterministic global gradient clipping and GPU-resident FP32 AdamW.
use super::{
    TrainConfig,
    model::{Parameter, Transformer},
    ops::FloatTensor,
};
use crate::kernels::Scalars;
use crate::ops::{Ops, config};
use crate::{Error, Result};
use hrx::Stream;

/// Update all adapter parameters after validating the complete gradient norm.
/// Returns the averaged, unclipped norm; `step` starts at one.
pub fn update(
    ops: &Ops,
    stream: &mut Stream,
    model: &Transformer,
    config: &TrainConfig,
    step: usize,
) -> Result<f64> {
    if step == 0 {
        return Err(Error::invalid("AdamW step starts at one"));
    }
    let parameters: Vec<&Parameter> =
        model.adapters.values().flat_map(|p| [&p.a, &p.b]).collect();
    let mut partials = Vec::with_capacity(parameters.len());
    for p in &parameters {
        let parts = p.grad.size().div_ceil(1024);
        let out = FloatTensor::zero(stream, 1, parts)?;
        // SAFETY: each wave reduces at most 1024 input floats and writes one partial.
        unsafe {
            ops.launch(
                stream,
                "train_grad_norm",
                crate::ops::config(&[("parts", parts)]),
                &Scalars::new().index(p.grad.size()),
                &[p.grad.binding(), out.binding()],
                parts,
                1,
                32,
            )?;
        }
        partials.push(out);
    }
    let mut sum = 0.0f64;
    for partial in partials {
        for value in partial.download(stream)? {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(
                    "nonfinite adapter gradient; optimizer update cancelled",
                ));
            }
            sum += f64::from(value);
        }
    }
    let norm = sum.sqrt() / config.accumulation as f64;
    let clip = if norm > f64::from(config.max_grad_norm) {
        f64::from(config.max_grad_norm) / norm
    } else {
        1.0
    };
    let grad_scale = (clip / config.accumulation as f64) as f32;
    let bias1 = (1.0 - f64::from(config.beta1).powf(step as f64)) as f32;
    let bias2 = (1.0 - f64::from(config.beta2).powf(step as f64)) as f32;
    for p in parameters {
        adamw(ops, stream, p, config, grad_scale, bias1, bias2)?;
    }
    Ok(norm)
}

/// One parameter update; callers must validate all gradients before dispatching.
pub fn adamw(
    ops: &Ops,
    stream: &Stream,
    p: &Parameter,
    c: &TrainConfig,
    scale: f32,
    bias1: f32,
    bias2: f32,
) -> Result<()> {
    if [(&p.grad, &p.master), (&p.first, &p.master), (&p.second, &p.master)]
        .iter()
        .any(|(a, b)| a.rows != b.rows || a.cols != b.cols)
        || p.value.rows() != p.master.rows
        || p.value.cols() != p.master.cols
        || !scale.is_finite()
        || scale < 0.0
        || !bias1.is_finite()
        || bias1 <= 0.0
        || !bias2.is_finite()
        || bias2 <= 0.0
    {
        return Err(Error::invalid("AdamW state dimensions or update scale"));
    }
    let scalars = Scalars::new()
        .index(p.master.size())
        .float(c.learning_rate)
        .float(c.weight_decay)
        .float(c.beta1)
        .float(c.beta2)
        .float(c.epsilon)
        .float(scale)
        .float(bias1)
        .float(bias2);
    // SAFETY: all FP32 states and the BF16 execution copy have the same shape.
    unsafe {
        ops.launch_1d(
            stream,
            "train_adamw",
            config(&[]),
            &scalars,
            &[
                p.master.binding(),
                p.grad.binding(),
                p.first.binding(),
                p.second.binding(),
                p.value.binding()?,
            ],
            p.master.size(),
        )
    }
}
