//! BF16 main blocks, LoRA projections, and their explicit reverse pass.
use std::collections::BTreeMap;
use std::path::Path;

use hrx::Stream;

use super::ops::{self as train, FloatTensor};
use crate::checkpoint::{Checkpoint, DType};
use crate::lora::{Adapter, Factors, PROJECTIONS};
use crate::models::Weights;
use crate::ops::{Binary, Norm, Ops, Tensor, Unary, Weight};
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
        let low = train::matmul(ops, stream, x, &self.a.value, 1.0)?;
        let delta = train::matmul(ops, stream, &low, &self.b.value, 1.0)?;
        train::add_scaled(ops, stream, base, &delta, self.scale() * strength)
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
        let gt = train::transpose(ops, stream, grad)?;
        let lt = train::transpose(ops, stream, &low)?;
        let db = train::matmul_float(ops, stream, &gt, &lt, self.scale())?;
        train::accumulate(ops, stream, &self.b.grad, &db)?;
        let bt = train::transpose(ops, stream, &self.b.value)?;
        let dl = train::matmul(ops, stream, grad, &bt, self.scale())?;
        let dlt = train::transpose(ops, stream, &dl)?;
        let xt = train::transpose(ops, stream, x)?;
        let da = train::matmul_float(ops, stream, &dlt, &xt, 1.0)?;
        train::accumulate(ops, stream, &self.a.grad, &da)?;
        let at = train::transpose(ops, stream, &self.a.value)?;
        let dx = train::matmul(ops, stream, &dl, &at, 1.0)?;
        train::add_scaled(ops, stream, base_grad, &dx, 1.0)
    }
}

/// Frozen dense main-block weights plus original-basis adapter parameters.
pub struct Transformer {
    weights: Weights,
    quantized: BTreeMap<String, super::quantized::Quantized>,
    /// Adapter projections, in stable checkpoint order.
    pub adapters: BTreeMap<String, Projection>,
    /// Number of transformer blocks.
    pub layers: usize,
}

/// Activations for one recomputed block. Dropped immediately after its backward pass.
pub struct BlockTape {
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
    /// Resulting block output.
    pub output: Tensor,
}

impl Transformer {
    /// Validate all RAW projection shapes and storage on CPU, without opening a device.
    pub fn validate_checkpoint(path: &Path) -> Result<()> {
        let file = Checkpoint::open(path)?;
        Self::validate_weights(&file, false).map(|_| ())
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
        Ok(Self { weights, quantized, adapters, layers: 28 })
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
    ) -> Result<Tensor> {
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
                adapter.forward(ops, stream, x, &base, strength)
            }
            _ => Ok(base),
        }
    }

    fn linear_backward(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        grad: &Tensor,
        name: &str,
    ) -> Result<Tensor> {
        let w = self.weight(name)?;
        let base_grad =
            train::matmul_nn(ops, stream, grad, &w.tensor(w.shape[0], w.shape[1])?, 1.0)?;
        match self.adapters.get(name) {
            Some(adapter) => adapter.backward(ops, stream, x, grad, &base_grad),
            None => Ok(base_grad),
        }
    }

    fn norm(&self, ops: &Ops, stream: &mut Stream, x: &Tensor, name: &str) -> Result<Tensor> {
        ops.norm(stream, x, self.weights.get(name)?, Norm::OnePlusScale, 1e-5)
    }

    fn norm_backward(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        x: &Tensor,
        grad: &Tensor,
        name: &str,
    ) -> Result<Tensor> {
        let scale = self.weights.get(name)?.f32_values(stream)?;
        train::norm_backward(ops, stream, x, grad, scale, 1e-5)
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
        let scale1 = train::one_plus(ops, stream, &row(0)?)?;
        let norm1 = self.norm(ops, stream, x, &format!("{p}.prenorm.scale"))?;
        let pre = ops.binary(
            stream,
            &ops.binary(stream, &norm1, &scale1, Binary::Mul)?,
            &row(1)?,
            Binary::Add,
        )?;
        let q0 = self.linear(ops, stream, &pre, &format!("{p}.attn.wq"), strength)?;
        let k0 = self.linear(ops, stream, &pre, &format!("{p}.attn.wk"), strength)?;
        let v = self.linear(ops, stream, &pre, &format!("{p}.attn.wv"), strength)?;
        let gate0 = self.linear(ops, stream, &pre, &format!("{p}.attn.gate"), strength)?;
        let gate = ops.unary(stream, &gate0, Unary::Sigmoid)?;
        let qn = self
            .norm(
                ops,
                stream,
                &q0.view(x.rows() * 48, 128, 0)?,
                &format!("{p}.attn.qknorm.qnorm.scale"),
            )?
            .view(x.rows(), 6144, 0)?;
        let kn = self
            .norm(
                ops,
                stream,
                &k0.view(x.rows() * 12, 128, 0)?,
                &format!("{p}.attn.qknorm.knorm.scale"),
            )?
            .view(x.rows(), 1536, 0)?;
        let q = train::rope(ops, stream, &qn, cos, sin, false)?;
        let k = train::rope(ops, stream, &kn, cos, sin, false)?;
        let attention = train::attention(ops, stream, &q, &k, &v)?;
        let attended = ops.binary(stream, &attention.output, &gate, Binary::Mul)?;
        let projected =
            self.linear(ops, stream, &attended, &format!("{p}.attn.wo"), strength)?;
        let residual = ops.binary(
            stream,
            x,
            &ops.binary(stream, &projected, &row(2)?, Binary::Mul)?,
            Binary::Add,
        )?;
        let scale2 = train::one_plus(ops, stream, &row(3)?)?;
        let norm2 = self.norm(ops, stream, &residual, &format!("{p}.postnorm.scale"))?;
        let post = ops.binary(
            stream,
            &ops.binary(stream, &norm2, &scale2, Binary::Mul)?,
            &row(4)?,
            Binary::Add,
        )?;
        let mlp_gate0 = self.linear(ops, stream, &post, &format!("{p}.mlp.gate"), strength)?;
        let mlp_gate = ops.unary(stream, &mlp_gate0, Unary::Silu)?;
        let up = self.linear(ops, stream, &post, &format!("{p}.mlp.up"), strength)?;
        let mixed = ops.binary(stream, &mlp_gate, &up, Binary::Mul)?;
        let down = self.linear(ops, stream, &mixed, &format!("{p}.mlp.down"), strength)?;
        let output = ops.binary(
            stream,
            &residual,
            &ops.binary(stream, &down, &row(5)?, Binary::Mul)?,
            Binary::Add,
        )?;
        Ok(BlockTape {
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
        if !self.quantized.is_empty() {
            return Err(Error::invalid("quantized training is not supported"));
        }
        let p = format!("blocks.{index}");
        let row = |i: usize| mods.view(1, 6144, i * 6144);
        let gd = ops.binary(stream, grad, &row(5)?, Binary::Mul)?;
        let gm = self.linear_backward(ops, stream, &t.mixed, &gd, &format!("{p}.mlp.down"))?;
        let gu = ops.binary(stream, &gm, &t.mlp_gate, Binary::Mul)?;
        let gg = ops.binary(stream, &gm, &t.up, Binary::Mul)?;
        let gg = train::activation_backward(ops, stream, &t.mlp_gate0, &gg, false)?;
        let gp1 = self.linear_backward(ops, stream, &t.post, &gg, &format!("{p}.mlp.gate"))?;
        let gp2 = self.linear_backward(ops, stream, &t.post, &gu, &format!("{p}.mlp.up"))?;
        let gp = train::add_scaled(ops, stream, &gp1, &gp2, 1.0)?;
        let gn2 =
            ops.binary(stream, &gp, &train::one_plus(ops, stream, &row(3)?)?, Binary::Mul)?;
        let gr =
            self.norm_backward(ops, stream, &t.residual, &gn2, &format!("{p}.postnorm.scale"))?;
        let gr = train::add_scaled(ops, stream, grad, &gr, 1.0)?;
        let go = ops.binary(stream, &gr, &row(2)?, Binary::Mul)?;
        let ga =
            self.linear_backward(ops, stream, &t.attended, &go, &format!("{p}.attn.wo"))?;
        let gat = ops.binary(stream, &ga, &t.gate, Binary::Mul)?;
        let gg = ops.binary(stream, &ga, &t.attention.output, Binary::Mul)?;
        let gg = train::activation_backward(ops, stream, &t.gate0, &gg, true)?;
        let (gq, gk, gv) =
            train::attention_backward(ops, stream, &t.q, &t.k, &t.v, &t.attention, &gat)?;
        let gq = train::rope(ops, stream, &gq, cos, sin, true)?;
        let gk = train::rope(ops, stream, &gk, cos, sin, true)?;
        let tokens = grad.rows();
        let gq = self
            .norm_backward(
                ops,
                stream,
                &t.q0.view(tokens * 48, 128, 0)?,
                &gq.view(tokens * 48, 128, 0)?,
                &format!("{p}.attn.qknorm.qnorm.scale"),
            )?
            .view(tokens, 6144, 0)?;
        let gk = self
            .norm_backward(
                ops,
                stream,
                &t.k0.view(tokens * 12, 128, 0)?,
                &gk.view(tokens * 12, 128, 0)?,
                &format!("{p}.attn.qknorm.knorm.scale"),
            )?
            .view(tokens, 1536, 0)?;
        let mut pre_grad =
            self.linear_backward(ops, stream, &t.pre, &gq, &format!("{p}.attn.wq"))?;
        for (name, g) in [("wk", &gk), ("wv", &gv), ("gate", &gg)] {
            let part =
                self.linear_backward(ops, stream, &t.pre, g, &format!("{p}.attn.{name}"))?;
            pre_grad = train::add_scaled(ops, stream, &pre_grad, &part, 1.0)?;
        }
        let gn1 = ops.binary(
            stream,
            &pre_grad,
            &train::one_plus(ops, stream, &row(0)?)?,
            Binary::Mul,
        )?;
        let gx =
            self.norm_backward(ops, stream, &t.input, &gn1, &format!("{p}.prenorm.scale"))?;
        train::add_scaled(ops, stream, &gr, &gx, 1.0)
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
