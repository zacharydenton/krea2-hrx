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

/// Reusable norm and AdamW graphs bound to the parameter storage at construction.
/// The graphs retain their buffers; replacing a parameter does not rebind them.
pub struct PreparedOptimizer {
    norm: hrx::GraphExec,
    adam: hrx::GraphExec,
    partials: FloatTensor,
    controls: FloatTensor,
    // Native graphs retain allocations, but a pooled lease must also remain alive
    // so its storage cannot be handed to another tensor while replay still uses it.
    _states: Vec<FloatTensor>,
    _values: Vec<crate::ops::Tensor>,
}

impl PreparedOptimizer {
    /// Prepare independent dispatches for distinct parameter/state allocations.
    /// No parameter or moment is modified during preparation.
    pub fn new(stream: &Stream, parameters: &[&Parameter]) -> Result<Self> {
        if parameters.is_empty() {
            return Err(Error::invalid("optimizer requires parameters"));
        }
        let mut allocations = std::collections::HashSet::new();
        for p in parameters {
            validate_parameter(p)?;
            for view in [
                p.master.binding(),
                p.grad.binding(),
                p.first.binding(),
                p.second.binding(),
                p.value.binding()?,
            ] {
                // Independent nodes must never write overlapping state, including
                // aliases within one parameter. Reject even disjoint shared allocations.
                if !allocations.insert(std::ptr::from_ref(view.owner())) {
                    return Err(Error::invalid(
                        "optimizer graph requires distinct state allocations",
                    ));
                }
            }
        }
        let parts_total = parameters.iter().map(|p| p.grad.size().div_ceil(1024)).sum();
        let partials = FloatTensor::zero(stream, 1, parts_total)?;
        let controls = FloatTensor::zero(stream, 1, 8)?;
        let kernels = crate::kernels::cache::PreparedKernels::default();
        let prepared = parameters
            .iter()
            .map(|p| {
                let parts = p.grad.size().div_ceil(1024);
                let groups = p.master.size().div_ceil(256);
                let norm = kernels.get(
                    stream,
                    "train_grad_norm",
                    config(&[("parts", parts)]),
                    (parts as u32, 1),
                )?;
                let adam = kernels.get(
                    stream,
                    "train_adamw_graph",
                    config(&[]),
                    (groups as u32, 1),
                )?;
                Ok((norm, adam, parts, groups))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut norm_graph = stream.graph()?;
        let mut adam_graph = stream.graph()?;
        let mut offset = 0;
        for (p, (norm, adam, parts, groups)) in parameters.iter().zip(&prepared) {
            let norm_constants =
                Scalars::new().index(p.grad.size()).pack("train_grad_norm", norm)?;
            let adam_constants =
                Scalars::new().index(p.master.size()).pack("train_adamw_graph", adam)?;
            // SAFETY: every norm node reads one gradient and writes its own partial
            // slice. Adam nodes have distinct validated state and read shared controls.
            // Separate graph submissions plus blocking norm readback order the phases.
            unsafe {
                norm_graph.dispatch(
                    &[],
                    norm,
                    [*parts as u32, 1, 1],
                    [32, 1, 1],
                    &norm_constants,
                    &[p.grad.binding(), partials.binding().slice(offset * 4, parts * 4)?],
                )?;
                adam_graph.dispatch(
                    &[],
                    adam,
                    [*groups as u32, 1, 1],
                    [256, 1, 1],
                    &adam_constants,
                    &[
                        controls.binding(),
                        p.master.binding(),
                        p.grad.binding(),
                        p.first.binding(),
                        p.second.binding(),
                        p.value.binding()?,
                    ],
                )?;
            }
            offset += parts;
        }
        Ok(Self {
            norm: norm_graph.finish()?,
            adam: adam_graph.finish()?,
            partials,
            controls,
            _states: parameters
                .iter()
                .flat_map(|p| {
                    [p.master.clone(), p.grad.clone(), p.first.clone(), p.second.clone()]
                })
                .collect(),
            _values: parameters.iter().map(|p| p.value.clone()).collect(),
        })
    }

    /// Validate the complete averaged norm, then replay the parameter updates.
    /// A nonfinite gradient leaves all parameters, moments and gradients unchanged.
    pub fn update(
        &mut self,
        stream: &mut Stream,
        config: &TrainConfig,
        step: usize,
    ) -> Result<f64> {
        validate_update(config, step)?;
        super::ops::profile("grad_norm_graph", || {
            stream.launch(&mut self.norm).map_err(Error::from)
        })?;
        let values =
            super::ops::profile("grad_norm_readback", || self.partials.download(stream))?;
        let (norm, controls) = update_controls(&values, config, step)?;
        stream.upload(self.controls.binding(), bytemuck::cast_slice(&controls))?;
        super::ops::profile("adamw_graph", || {
            stream.launch(&mut self.adam).map_err(Error::from)
        })?;
        Ok(norm)
    }
}

/// Update all adapter parameters after validating the complete gradient norm.
/// Returns the averaged, unclipped norm; `step` starts at one.
pub fn update(
    ops: &Ops,
    stream: &mut Stream,
    model: &Transformer,
    config: &TrainConfig,
    step: usize,
) -> Result<f64> {
    let parameters: Vec<&Parameter> =
        model.adapters.values().flat_map(|p| [&p.a, &p.b]).collect();
    update_parameters(ops, stream, &parameters, config, step)
}

/// Unprepared update for standalone callers and the runnable graph baseline.
pub fn update_parameters(
    ops: &Ops,
    stream: &mut Stream,
    parameters: &[&Parameter],
    config: &TrainConfig,
    step: usize,
) -> Result<f64> {
    validate_update(config, step)?;
    if parameters.is_empty() {
        return Err(Error::invalid("optimizer requires parameters"));
    }
    for p in parameters {
        validate_parameter(p)?;
    }
    let parts_total = parameters.iter().map(|p| p.grad.size().div_ceil(1024)).sum();
    let partials = FloatTensor::scratch(ops, stream, 1, parts_total)?;
    let mut offset = 0;
    for p in parameters {
        let parts = p.grad.size().div_ceil(1024);
        super::ops::profile("grad_norm_submit", || {
            // SAFETY: each wave reduces at most 1024 input floats and writes one partial.
            unsafe {
                ops.launch(
                    stream,
                    "train_grad_norm",
                    crate::ops::config(&[("parts", parts)]),
                    &Scalars::new().index(p.grad.size()),
                    &[p.grad.binding(), partials.binding().slice(offset * 4, parts * 4)?],
                    parts,
                    1,
                    32,
                )
            }
        })?;
        offset += parts;
    }
    let values = super::ops::profile("grad_norm_readback", || partials.download(stream))?;
    let (norm, controls) = update_controls(&values, config, step)?;
    for p in parameters {
        adamw(ops, stream, p, config, controls[5], controls[6], controls[7])?;
    }
    Ok(norm)
}

fn validate_update(c: &TrainConfig, step: usize) -> Result<()> {
    if step == 0 || c.accumulation == 0 {
        return Err(Error::invalid("AdamW step and accumulation start at one"));
    }
    if [c.learning_rate, c.epsilon, c.max_grad_norm].iter().any(|v| !v.is_finite() || *v <= 0.0)
        || !c.weight_decay.is_finite()
        || c.weight_decay < 0.0
        || !(0.0..1.0).contains(&c.beta1)
        || !(0.0..1.0).contains(&c.beta2)
    {
        return Err(Error::invalid("invalid AdamW coefficients"));
    }
    Ok(())
}

fn update_controls(
    values: &[f32],
    config: &TrainConfig,
    step: usize,
) -> Result<(f64, [f32; 8])> {
    let mut sum = 0.0f64;
    for &value in values {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid(
                "nonfinite adapter gradient; optimizer update cancelled",
            ));
        }
        sum += f64::from(value);
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
    Ok((
        norm,
        [
            config.learning_rate,
            config.weight_decay,
            config.beta1,
            config.beta2,
            config.epsilon,
            grad_scale,
            bias1,
            bias2,
        ],
    ))
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
    super::ops::profile("adamw", || adamw_inner(ops, stream, p, c, scale, bias1, bias2))
}

fn adamw_inner(
    ops: &Ops,
    stream: &Stream,
    p: &Parameter,
    c: &TrainConfig,
    scale: f32,
    bias1: f32,
    bias2: f32,
) -> Result<()> {
    validate_parameter(p)?;
    if !scale.is_finite()
        || scale < 0.0
        || !bias1.is_finite()
        || bias1 <= 0.0
        || !bias2.is_finite()
        || bias2 <= 0.0
    {
        return Err(Error::invalid("AdamW update scale"));
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

fn validate_parameter(p: &Parameter) -> Result<()> {
    if [(&p.grad, &p.master), (&p.first, &p.master), (&p.second, &p.master)]
        .iter()
        .any(|(a, b)| a.rows() != b.rows() || a.cols() != b.cols())
        || p.value.rows() != p.master.rows()
        || p.value.cols() != p.master.cols()
    {
        return Err(Error::invalid("AdamW state dimensions or update scale"));
    }
    Ok(())
}
