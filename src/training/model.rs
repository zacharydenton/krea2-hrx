//! BF16 main blocks, LoRA projections, and their explicit reverse pass.
use std::collections::BTreeMap;
use std::path::Path;

use hrx::Stream;

use super::ops::{self as train, FloatTensor};
use crate::checkpoint::{Checkpoint, DType};
use crate::lora::{Adapter, Factors, PROJECTIONS};
use crate::models::Weights;
use crate::ops::{Binary, Ops, Tensor, Weight};
use crate::{Error, Result};

/// FP32 master parameter, gradient, Adam moments, and a BF16 execution copy.
pub struct Parameter {
    /// Full-precision value updated by AdamW.
    pub master: FloatTensor,
    /// Accumulated gradient, cleared after an update.
    pub grad: FloatTensor,
    /// First Adam moment.
    pub first: FloatTensor,
    /// Second Adam moment.
    pub second: FloatTensor,
    /// BF16 value consumed by projection kernels.
    pub value: Tensor,
}

impl Parameter {
    fn new(
        ops: &Ops,
        stream: &mut Stream,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> Result<Self> {
        let master = FloatTensor::from_slice(stream, rows, cols, values)?;
        Ok(Self {
            value: train::cast(ops, stream, &master)?,
            master,
            grad: FloatTensor::zero(stream, rows, cols)?,
            first: FloatTensor::zero(stream, rows, cols)?,
            second: FloatTensor::zero(stream, rows, cols)?,
        })
    }
}

/// A trainable low-rank branch, independent of the frozen base projection.
pub struct Projection {
    /// Down projection.
    pub a: Parameter,
    /// Up projection.
    pub b: Parameter,
    /// Saved alpha, before division by rank.
    pub alpha: f32,
}

impl Projection {
    fn scale(&self) -> f32 {
        self.alpha / self.a.master.rows() as f32
    }
    /// Upload validated factors and initialize zero optimizer state.
    pub fn new(ops: &Ops, stream: &mut Stream, f: &Factors) -> Result<Self> {
        f.validate()?;
        Ok(Self {
            a: Parameter::new(ops, stream, f.rank, f.inputs, &f.a)?,
            b: Parameter::new(ops, stream, f.outputs, f.rank, &f.b)?,
            alpha: f.alpha,
        })
    }
    /// Add this low-rank branch to a frozen base projection.
    pub fn forward(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        base: &Tensor,
        strength: f32,
    ) -> Result<Tensor> {
        Ok(self.forward_cached(ops, stream, x, base, strength)?.0)
    }
    /// Forward with the rank-sized activation retained for `backward_cached`.
    pub fn forward_cached(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        base: &Tensor,
        strength: f32,
    ) -> Result<(Tensor, Tensor)> {
        let low = train::matmul(ops, stream, x, &self.a.value, 1.0)?;
        let output = train::matmul_add(
            ops,
            stream,
            &low,
            &self.b.value,
            base,
            self.scale() * strength,
            true,
        )?;
        Ok((output, low))
    }
    /// Accumulate parameter gradients and add the branch's input gradient.
    pub fn backward(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        grad: &Tensor,
        base_grad: &Tensor,
    ) -> Result<Tensor> {
        let low = train::matmul(ops, stream, x, &self.a.value, 1.0)?;
        self.backward_cached(ops, stream, x, grad, base_grad, &low)
    }
    /// Backward using the activation from the matching forward, before any weight update.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_cached(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        grad: &Tensor,
        base_grad: &Tensor,
        low: &Tensor,
    ) -> Result<Tensor> {
        self.backward_cached_with_residual(ops, stream, x, grad, base_grad, low, None)
    }

    /// Accumulate parameter gradients and add an optional existing input gradient.
    /// The branch is rounded to BF16 before the residual addition; inputs stay immutable.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_cached_with_residual(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        grad: &Tensor,
        base_grad: &Tensor,
        low: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        if (low.rows(), low.cols()) != (x.rows(), self.a.value.rows())
            || x.cols() != self.a.value.cols()
            || (grad.rows(), grad.cols()) != (x.rows(), self.b.value.rows())
            || (base_grad.rows(), base_grad.cols()) != (x.rows(), x.cols())
            || residual.is_some_and(|r| (r.rows(), r.cols()) != (x.rows(), x.cols()))
        {
            return Err(Error::invalid("cached LoRA backward dimensions"));
        }
        train::matmul_tn_accumulate(ops, stream, grad, low, &self.b.grad, self.scale())?;
        let dl = train::matmul_nn(ops, stream, grad, &self.b.value, self.scale())?;
        train::matmul_tn_accumulate(ops, stream, &dl, x, &self.a.grad, 1.0)?;
        train::matmul_add_residual(
            ops,
            stream,
            &dl,
            &self.a.value,
            base_grad,
            1.0,
            false,
            residual,
        )
    }
}

/// Frozen dense main-block weights plus original-basis adapter parameters.
pub struct Transformer {
    pub(crate) full: Option<super::full::parameters::Parameters>,
    weights: Weights,
    quantized: BTreeMap<String, super::quantized::Quantized>,
    /// Adapter projections, in stable checkpoint order.
    pub adapters: BTreeMap<String, Projection>,
    /// Number of transformer blocks.
    pub layers: usize,
}

/// Activations for one recomputed block. Dropped immediately after its backward pass.
pub struct BlockTape {
    // Rank-sized forward activations in PROJECTIONS order, before any optimizer update.
    adapter_lows: [Option<Tensor>; 8],
    input: Tensor,
    pre: Tensor,
    q0: Tensor,
    k0: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate0: Tensor,
    gate: Tensor,
    attention: train::Attention,
    attended: Tensor,
    residual: Tensor,
    post: Tensor,
    mlp_gate0: Tensor,
    mlp_gate: Tensor,
    up: Tensor,
    mixed: Tensor,
    modulation: Option<[Tensor; 4]>,
    /// Resulting block output.
    pub output: Tensor,
}

impl Transformer {
    pub(crate) fn full(
        weights: Weights,
        parameters: super::full::parameters::Parameters,
    ) -> Self {
        Self {
            full: Some(parameters),
            weights,
            quantized: BTreeMap::new(),
            adapters: BTreeMap::new(),
            layers: 28,
        }
    }
    #[cfg(test)]
    pub(crate) fn auxiliary_test_model(
        ops: &Ops,
        stream: &mut Stream,
        adapter: &Adapter,
    ) -> Result<Self> {
        let adapters = adapter
            .layers
            .iter()
            .map(|(name, f)| Ok((name.clone(), Projection::new(ops, stream, f)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            full: None,
            weights: Weights::empty(),
            quantized: BTreeMap::new(),
            adapters,
            layers: 28,
        })
    }

    /// Validate all RAW projection shapes and storage on CPU, without opening a device.
    pub fn validate_checkpoint(path: &Path) -> Result<()> {
        let file = Checkpoint::open(path)?;
        Self::validate_weights(&file, false)?;
        for (name, outputs, inputs) in crate::lora::Targets::All.layers().into_iter().skip(224)
        {
            let t = file.get(&format!("{name}.weight"))?;
            if !matches!(t.dtype, DType::BF16 | DType::F32) || t.shape != [outputs, inputs] {
                return Err(Error::invalid(format!(
                    "{name}: expected dense [{outputs}, {inputs}]"
                )));
            }
        }
        Ok(())
    }

    fn validate_weights(file: &Checkpoint, allow_quantized: bool) -> Result<DType> {
        if file.block_count() != 28 {
            return Err(Error::invalid("training requires the 28-block Krea 2 transformer"));
        }
        let dtype = file.get("blocks.0.attn.wq.weight")?.dtype;
        if dtype != DType::BF16 && !(allow_quantized && dtype == DType::I8) {
            return Err(Error::invalid(
                "training requires original-basis BF16 weights; inference also accepts INT8 ConvRot",
            ));
        }
        for block in 0..28 {
            for (name, outputs, inputs) in PROJECTIONS {
                let key = format!("blocks.{block}.{name}.weight");
                let tensor = file.get(&key)?;
                if tensor.dtype != dtype || tensor.shape != [outputs, inputs] {
                    return Err(Error::invalid(format!(
                        "{key}: expected {dtype:?} [{outputs}, {inputs}]"
                    )));
                }
            }
        }
        Ok(dtype)
    }

    /// Load the 28 dense blocks. Quantized weights are rejected before allocation.
    pub fn load(
        ops: &Ops,
        stream: &mut Stream,
        path: &Path,
        adapter: Option<&Adapter>,
    ) -> Result<Self> {
        Self::load_inner(ops, stream, path, adapter, false)
    }

    /// Load inference blocks, allowing the existing INT8 ConvRot base projections.
    pub fn load_inference(
        ops: &Ops,
        stream: &mut Stream,
        path: &Path,
        adapter: Option<&Adapter>,
    ) -> Result<Self> {
        Self::load_inner(ops, stream, path, adapter, true)
    }

    fn load_inner(
        ops: &Ops,
        stream: &mut Stream,
        path: &Path,
        adapter: Option<&Adapter>,
        allow_quantized: bool,
    ) -> Result<Self> {
        if let Some(adapter) = adapter {
            adapter.validate()?;
        }
        let file = Checkpoint::open(path)?;
        let dtype = Self::validate_weights(&file, allow_quantized)?;
        let weights = Weights::load(stream, &file, |key| {
            if key.starts_with("blocks.")
                && !key.contains(".mod.")
                && !file.get(key).is_ok_and(|t| t.dtype == DType::I8)
            {
                key.into()
            } else {
                String::new()
            }
        })?;
        let mut quantized = BTreeMap::new();
        if dtype == DType::I8 {
            for block in 0..28 {
                for (name, outputs, inputs) in PROJECTIONS {
                    let name = format!("blocks.{block}.{name}");
                    quantized.insert(
                        name.clone(),
                        super::quantized::Quantized::load(
                            stream, &file, &name, outputs, inputs,
                        )?,
                    );
                }
            }
        }
        let adapters = adapter
            .map(|adapter| {
                adapter
                    .layers
                    .iter()
                    .map(|(name, factors)| {
                        Ok((name.clone(), Projection::new(ops, stream, factors)?))
                    })
                    .collect::<Result<_>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self { full: None, weights, quantized, adapters, layers: 28 })
    }

    /// Whether any auxiliary DiT projection has an adapter.
    pub(crate) fn has_auxiliary(&self) -> bool {
        self.full.is_some() || self.adapters.keys().any(|name| !name.starts_with("blocks."))
    }

    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights.get(&format!("{name}.weight"))
    }

    fn linear(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        name: &str,
        strength: f32,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let base = match self.quantized.get(name) {
            Some(weight) => weight.forward(ops, stream, x)?,
            None => {
                let weight = self.weight(name)?;
                train::matmul(
                    ops,
                    stream,
                    x,
                    &weight.tensor(weight.shape[0], weight.shape[1])?,
                    1.0,
                )?
            }
        };
        match self.adapters.get(name) {
            Some(adapter) if strength != 0.0 => {
                let (output, low) = adapter.forward_cached(ops, stream, x, &base, strength)?;
                Ok((output, Some(low)))
            }
            _ => Ok((base, None)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        grad: &Tensor,
        name: &str,
        low: Option<&Tensor>,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let w = self.weight(name)?;
        let base_grad =
            train::matmul_nn(ops, stream, grad, &w.tensor(w.shape[0], w.shape[1])?, 1.0)?;
        if let Some(full) = &self.full {
            full.linear_gradient(ops, stream, name, x, grad)?;
        }
        if let Some(adapter) = self.adapters.get(name) {
            let recomputed;
            let low = match low {
                Some(low) => low,
                None => {
                    recomputed = train::matmul(ops, stream, x, &adapter.a.value, 1.0)?;
                    &recomputed
                }
            };
            adapter
                .backward_cached_with_residual(ops, stream, x, grad, &base_grad, low, residual)
        } else {
            match residual {
                Some(r) => train::add_scaled(ops, stream, r, &base_grad, 1.0),
                None => Ok(base_grad),
            }
        }
    }

    /// Forward one block, retaining the activations needed for its reverse pass.
    #[allow(clippy::too_many_arguments)]
    pub fn block(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        index: usize,
        x: &Tensor,
        mods: &Tensor,
        cos: &FloatTensor,
        sin: &FloatTensor,
        strength: f32,
    ) -> Result<BlockTape> {
        if index >= self.layers || x.cols() != 6144 || mods.size() != 6 * 6144 {
            return Err(Error::invalid("dense block dimensions"));
        }
        let p = format!("blocks.{index}");
        let row = |i: usize| mods.view(1, 6144, i * 6144);
        let (norm1, pre) = train::norm_modulated(
            ops,
            stream,
            x,
            self.weights.get(&format!("{p}.prenorm.scale"))?,
            &row(0)?,
            &row(1)?,
            1e-5,
        )?;
        let (q0, low0) = self.linear(ops, stream, &pre, &format!("{p}.attn.wq"), strength)?;
        let (k0, low1) = self.linear(ops, stream, &pre, &format!("{p}.attn.wk"), strength)?;
        let (v, low2) = self.linear(ops, stream, &pre, &format!("{p}.attn.wv"), strength)?;
        let (gate0, low3) =
            self.linear(ops, stream, &pre, &format!("{p}.attn.gate"), strength)?;
        let q = train::norm_rope(
            ops,
            stream,
            &q0,
            self.weights.get(&format!("{p}.attn.qknorm.qnorm.scale"))?,
            cos,
            sin,
            1e-5,
        )?;
        let k = train::norm_rope(
            ops,
            stream,
            &k0,
            self.weights.get(&format!("{p}.attn.qknorm.knorm.scale"))?,
            cos,
            sin,
            1e-5,
        )?;
        let attention = train::attention(ops, stream, &q, &k, &v)?;
        let (gate, attended) =
            train::gated_forward(ops, stream, &gate0, &attention.output, true)?;
        let (projected, low4) =
            self.linear(ops, stream, &attended, &format!("{p}.attn.wo"), strength)?;
        let residual = train::residual_gate(ops, stream, x, &projected, &row(2)?)?;
        let (norm2, post) = train::norm_modulated(
            ops,
            stream,
            &residual,
            self.weights.get(&format!("{p}.postnorm.scale"))?,
            &row(3)?,
            &row(4)?,
            1e-5,
        )?;
        let (mlp_gate0, low5) =
            self.linear(ops, stream, &post, &format!("{p}.mlp.gate"), strength)?;
        let (up, low6) = self.linear(ops, stream, &post, &format!("{p}.mlp.up"), strength)?;
        let (mlp_gate, mixed) = train::gated_forward(ops, stream, &mlp_gate0, &up, false)?;
        let (down, low7) =
            self.linear(ops, stream, &mixed, &format!("{p}.mlp.down"), strength)?;
        let output = train::residual_gate(ops, stream, &residual, &down, &row(5)?)?;
        Ok(BlockTape {
            adapter_lows: [low0, low1, low2, low3, low4, low5, low6, low7],
            input: x.clone(),
            pre,
            q0,
            k0,
            q,
            k,
            v,
            gate0,
            gate,
            attention,
            attended,
            residual,
            post,
            mlp_gate0,
            mlp_gate,
            up,
            mixed,
            modulation: (self.full.is_some() || self.adapters.contains_key("tproj.1"))
                .then_some([norm1, projected, norm2, down]),
            output,
        })
    }

    /// Backpropagate one recomputed block, accumulating only adapter parameter gradients.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        index: usize,
        t: BlockTape,
        mods: &Tensor,
        cos: &FloatTensor,
        sin: &FloatTensor,
        grad: &Tensor,
    ) -> Result<Tensor> {
        self.backward_with_modulation(ops, stream, index, t, mods, cos, sin, grad, None)
    }

    /// Also accumulate all six broadcast modulation gradients in FP32.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_with_modulation(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        index: usize,
        t: BlockTape,
        mods: &Tensor,
        cos: &FloatTensor,
        sin: &FloatTensor,
        grad: &Tensor,
        modulation_grad: Option<&FloatTensor>,
    ) -> Result<Tensor> {
        if !self.quantized.is_empty() {
            return Err(Error::invalid("quantized training is not supported"));
        }
        let p = format!("blocks.{index}");
        let row = |i: usize| mods.view(1, 6144, i * 6144);
        let gd = ops.binary(stream, grad, &row(5)?, Binary::Mul)?;
        let gm = self.linear_backward(
            ops,
            stream,
            &t.mixed,
            &gd,
            &format!("{p}.mlp.down"),
            t.adapter_lows[7].as_ref(),
            None,
        )?;
        let (gg, gu) =
            train::gated_backward(ops, stream, &t.mlp_gate0, &t.mlp_gate, &t.up, &gm, false)?;
        let gp1 = self.linear_backward(
            ops,
            stream,
            &t.post,
            &gg,
            &format!("{p}.mlp.gate"),
            t.adapter_lows[5].as_ref(),
            None,
        )?;
        let gp = self.linear_backward(
            ops,
            stream,
            &t.post,
            &gu,
            &format!("{p}.mlp.up"),
            t.adapter_lows[6].as_ref(),
            Some(&gp1),
        )?;
        if let Some(dst) = modulation_grad {
            let [_, _, norm2, down] = t
                .modulation
                .as_ref()
                .ok_or_else(|| Error::invalid("missing modulation tape"))?;
            train::modulation_backward(ops, stream, grad, down, &gp, norm2, dst, 3)?;
        }
        let post_scale =
            self.weights.get(&format!("{p}.postnorm.scale"))?.f32_values(stream)?;
        if let Some(full) = &self.full {
            let factor = train::one_plus(ops, stream, &row(3)?)?;
            let dy = ops.binary(stream, &gp, &factor, Binary::Mul)?;
            full.norm_gradient(ops, stream, &format!("{p}.postnorm.scale"), &t.residual, &dy)?;
            let dst = full.gradient_matrix(&format!("{p}.mod.lin"), 6, 6144)?;
            let [_, _, norm2, down] = t.modulation.as_ref().expect("full modulation tape");
            train::modulation_backward(ops, stream, grad, down, &gp, norm2, &dst, 3)?;
        }
        let gr = train::norm_modulated_backward(
            ops,
            stream,
            &t.residual,
            &gp,
            post_scale,
            &row(3)?,
            grad,
            1e-5,
        )?;
        let go = ops.binary(stream, &gr, &row(2)?, Binary::Mul)?;
        let ga = self.linear_backward(
            ops,
            stream,
            &t.attended,
            &go,
            &format!("{p}.attn.wo"),
            t.adapter_lows[4].as_ref(),
            None,
        )?;
        let (gg, gat) = train::gated_backward(
            ops,
            stream,
            &t.gate0,
            &t.gate,
            &t.attention.output,
            &ga,
            true,
        )?;
        let (gq, gk, gv) =
            train::attention_backward(ops, stream, &t.q, &t.k, &t.v, &t.attention, &gat)?;
        let qscale =
            self.weights.get(&format!("{p}.attn.qknorm.qnorm.scale"))?.f32_values(stream)?;
        if let Some(full) = &self.full {
            let dy = train::rope(ops, stream, &gq, cos, sin, true)?;
            full.norm_gradient(
                ops,
                stream,
                &format!("{p}.attn.qknorm.qnorm.scale"),
                &t.q0,
                &dy,
            )?;
            let dy = train::rope(ops, stream, &gk, cos, sin, true)?;
            full.norm_gradient(
                ops,
                stream,
                &format!("{p}.attn.qknorm.knorm.scale"),
                &t.k0,
                &dy,
            )?;
        }
        let gq = train::rope_norm_backward(ops, stream, &t.q0, &gq, qscale, cos, sin, 1e-5)?;
        let kscale =
            self.weights.get(&format!("{p}.attn.qknorm.knorm.scale"))?.f32_values(stream)?;
        let gk = train::rope_norm_backward(ops, stream, &t.k0, &gk, kscale, cos, sin, 1e-5)?;
        let mut pre_grad = self.linear_backward(
            ops,
            stream,
            &t.pre,
            &gq,
            &format!("{p}.attn.wq"),
            t.adapter_lows[0].as_ref(),
            None,
        )?;
        for (i, (name, g)) in [("wk", &gk), ("wv", &gv), ("gate", &gg)].into_iter().enumerate()
        {
            pre_grad = self.linear_backward(
                ops,
                stream,
                &t.pre,
                g,
                &format!("{p}.attn.{name}"),
                t.adapter_lows[i + 1].as_ref(),
                Some(&pre_grad),
            )?;
        }
        if let Some(dst) = modulation_grad {
            let [norm1, projected, _, _] = t
                .modulation
                .as_ref()
                .ok_or_else(|| Error::invalid("missing modulation tape"))?;
            train::modulation_backward(ops, stream, &gr, projected, &pre_grad, norm1, dst, 0)?;
        }
        let pre_scale = self.weights.get(&format!("{p}.prenorm.scale"))?.f32_values(stream)?;
        if let Some(full) = &self.full {
            let factor = train::one_plus(ops, stream, &row(0)?)?;
            let dy = ops.binary(stream, &pre_grad, &factor, Binary::Mul)?;
            full.norm_gradient(ops, stream, &format!("{p}.prenorm.scale"), &t.input, &dy)?;
            let dst = full.gradient_matrix(&format!("{p}.mod.lin"), 6, 6144)?;
            let [norm1, projected, _, _] = t.modulation.as_ref().expect("full modulation tape");
            train::modulation_backward(ops, stream, &gr, projected, &pre_grad, norm1, &dst, 0)?;
        }
        train::norm_modulated_backward(
            ops,
            stream,
            &t.input,
            &pre_grad,
            pre_scale,
            &row(0)?,
            &gr,
            1e-5,
        )
    }

    /// Download current FP32 adapter masters for export or checkpointing.
    pub fn adapter(&self, stream: &mut Stream) -> Result<Adapter> {
        let layers = self
            .adapters
            .iter()
            .map(|(name, p)| {
                Ok((
                    name.clone(),
                    Factors {
                        inputs: p.a.master.cols(),
                        outputs: p.b.master.rows(),
                        rank: p.a.master.rows(),
                        alpha: p.alpha,
                        a: p.a.master.download(stream)?,
                        b: p.b.master.download(stream)?,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Adapter { layers })
    }
}

#[cfg(test)]
mod full_target_tests {
    use super::*;
    use crate::numerics::{from_f32, to_f32};

    #[test]
    #[ignore = "requires GPU and KREA2_RAW_CHECKPOINT; loads one main block only"]
    fn full_targets_main_modulation_gradients_match_cpu_chain_rule() {
        super::super::memory::before_load(2usize << 30).unwrap();
        let path = std::env::var_os("KREA2_RAW_CHECKPOINT").expect("set KREA2_RAW_CHECKPOINT");
        let file = Checkpoint::open(std::path::Path::new(&path)).unwrap();
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(hrx::BufferPool::new());
        let weights = Weights::load(&mut stream, &file, |name| {
            if name.starts_with("blocks.0.") { name.into() } else { String::new() }
        })
        .unwrap();
        let factors = Factors {
            inputs: 6144,
            outputs: 36864,
            rank: 1,
            alpha: 1.0,
            a: vec![0.0; 6144],
            b: vec![0.0; 36864],
        };
        let mut adapters: BTreeMap<_, _> =
            [("tproj.1".into(), Projection::new(&ops, &mut stream, &factors).unwrap())].into();
        for (name, outputs, inputs) in PROJECTIONS {
            let factors = Factors {
                inputs,
                outputs,
                rank: 32,
                alpha: 32.0,
                a: (0..inputs * 32).map(|i| ((i % 23) as f32 - 11.0) * 0.001).collect(),
                b: (0..outputs * 32).map(|i| ((i % 19) as f32 - 9.0) * 0.001).collect(),
            };
            adapters.insert(
                format!("blocks.0.{name}"),
                Projection::new(&ops, &mut stream, &factors).unwrap(),
            );
        }
        let model = Transformer {
            full: None,
            weights,
            quantized: BTreeMap::new(),
            adapters,
            layers: 28,
        };
        let upload = |s: &mut Stream, rows, offset, gain| {
            let bits: Vec<_> = (0..rows * 6144)
                .map(|i| from_f32((((i * 17 + offset) % 71) as f32 / 71.0 - 0.5) * gain))
                .collect();
            Tensor::from_slice(ops.pool(), s, &bits, rows, 6144).unwrap()
        };
        let x = upload(&mut stream, 1, 3, 1.0);
        let mods = upload(&mut stream, 6, 7, 0.25);
        let g = upload(&mut stream, 1, 11, 0.125);
        let cos = FloatTensor::from_slice(&mut stream, 1, 128, &[1.0; 128]).unwrap();
        let sin = FloatTensor::zero(&stream, 1, 128).unwrap();
        let tape = model.block(&ops, &mut stream, 0, &x, &mods, &cos, &sin, 1.0).unwrap();
        let read = |t: &Tensor, s: &mut Stream| {
            t.download(s).unwrap().into_iter().map(to_f32).collect::<Vec<_>>()
        };
        let [n1, projected, n2, down] = tape.modulation.as_ref().unwrap();
        let (n1, projected, n2, down) = (
            read(n1, &mut stream),
            read(projected, &mut stream),
            read(n2, &mut stream),
            read(down, &mut stream),
        );
        let residual = read(&tape.residual, &mut stream);
        let incoming = read(&g, &mut stream);
        let modulation = read(&mods, &mut stream);
        let dst = FloatTensor::zero(&stream, 6, 6144).unwrap();
        let input_grad = model
            .backward_with_modulation(
                &ops,
                &mut stream,
                0,
                tape,
                &mods,
                &cos,
                &sin,
                &g,
                Some(&dst),
            )
            .unwrap();
        let dm = dst.download(&mut stream).unwrap();
        let bf = |v: f32| to_f32(from_f32(v));
        let scales = crate::lora::floats(file.get("blocks.0.postnorm.scale").unwrap()).unwrap();
        // A single token exposes each broadcast reduction directly. Independently
        // differentiate the residual RMSNorm to verify the attention gate branch.
        let inv = 1.0
            / ((residual.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 6144.0 + 1e-5)
                .sqrt());
        let weighted: Vec<_> = (0..6144)
            .map(|c| {
                f64::from(bf(dm[4 * 6144 + c] * bf(1.0 + modulation[3 * 6144 + c])))
                    * (1.0 + f64::from(scales[c]))
            })
            .collect();
        let dot = weighted.iter().zip(&residual).map(|(g, x)| g * f64::from(*x)).sum::<f64>()
            / 6144.0;
        let mut actual_gate = Vec::new();
        let mut expected_gate = Vec::new();
        for c in 0..6144 {
            assert_eq!(dm[c], bf(dm[6144 + c] * n1[c]), "pre scale {c}");
            assert_eq!(dm[3 * 6144 + c], bf(dm[4 * 6144 + c] * n2[c]), "post scale {c}");
            assert_eq!(dm[5 * 6144 + c], bf(incoming[c] * down[c]), "MLP gate {c}");
            let norm_grad =
                bf((inv * (weighted[c] - f64::from(residual[c]) * inv * inv * dot)) as f32);
            expected_gate.push(bf(bf(incoming[c] + norm_grad) * projected[c]));
            actual_gate.push(dm[2 * 6144 + c]);
        }
        let err: f64 =
            actual_gate.iter().zip(&expected_gate).map(|(a, b)| f64::from(a - b).powi(2)).sum();
        let norm: f64 = expected_gate.iter().map(|v| f64::from(*v).powi(2)).sum();
        assert!(
            err <= norm * 0.000025 + 1e-12,
            "attention modulation relative L2 {}",
            (err / norm).sqrt()
        );
        assert!(dm.iter().all(|v| v.is_finite()));
        for row in dm.as_chunks::<6144>().0 {
            assert!(row.iter().any(|v| *v != 0.0));
        }
        let parameters: Vec<_> = model.adapters.values().flat_map(|p| [&p.a, &p.b]).collect();
        let gradients: Vec<_> =
            parameters.iter().map(|p| p.grad.download(&mut stream).unwrap()).collect();
        for p in &parameters {
            p.grad.clear(&stream).unwrap();
        }
        dst.clear(&stream).unwrap();
        let mut uncached =
            model.block(&ops, &mut stream, 0, &x, &mods, &cos, &sin, 1.0).unwrap();
        assert!(uncached.adapter_lows.iter().all(Option::is_some));
        uncached.adapter_lows = std::array::from_fn(|_| None);
        let reference = model
            .backward_with_modulation(
                &ops,
                &mut stream,
                0,
                uncached,
                &mods,
                &cos,
                &sin,
                &g,
                Some(&dst),
            )
            .unwrap();
        assert_eq!(
            input_grad.download(&mut stream).unwrap(),
            reference.download(&mut stream).unwrap()
        );
        assert_eq!(dst.download(&mut stream).unwrap(), dm);
        for (p, expected) in parameters.iter().zip(&gradients) {
            assert_eq!(&p.grad.download(&mut stream).unwrap(), expected);
        }
    }
}
