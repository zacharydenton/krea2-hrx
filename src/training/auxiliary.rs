//! Small explicit reverse graph for the trainable DiT conditioning towers.
//! Qwen and the VAE remain outside this graph. Main blocks use their specialized tape.
use super::{model::Transformer, ops as train};
use crate::{
    Error, Result,
    models::Models,
    ops::{Binary, Norm, Tensor, Unary},
};
use hrx::Stream;

type Id = usize;
enum Operation {
    Input,
    Linear(Id, String, Option<Tensor>),
    Norm(Id, String),
    Unary(Id, Unary),
    OnePlus(Id),
    Binary(Id, Id, Binary),
    Reshape(Id),
    Permute(Id),
    Attention(Id, Id, Id, train::Attention),
}
struct Node {
    value: Tensor,
    op: Operation,
}

pub(crate) struct Tape<'a> {
    models: &'a Models,
    model: &'a Transformer,
    strength: f32,
    nodes: Vec<Node>,
}
impl<'a> Tape<'a> {
    pub(crate) fn new(models: &'a Models, model: &'a Transformer, strength: f32) -> Self {
        Self { models, model, strength, nodes: Vec::new() }
    }
    pub(crate) fn value(&self, id: Id) -> &Tensor {
        &self.nodes[id].value
    }
    fn push(&mut self, value: Tensor, op: Operation) -> Id {
        let id = self.nodes.len();
        self.nodes.push(Node { value, op });
        id
    }
    pub(crate) fn input(&mut self, x: &Tensor) -> Id {
        self.push(x.clone(), Operation::Input)
    }
    fn reshape(&mut self, x: Id, rows: usize, cols: usize) -> Result<Id> {
        if rows.checked_mul(cols) != Some(self.value(x).size()) {
            return Err(Error::invalid("auxiliary reshape"));
        }
        Ok(self.push(self.value(x).view(rows, cols, 0)?, Operation::Reshape(x)))
    }
    fn linear(&mut self, s: &Stream, x: Id, name: &str) -> Result<Id> {
        let w = &self.models.transformer;
        let weight = w.get(&format!("{name}.weight"))?;
        let bias =
            w.find(&format!("{name}.bias")).map(crate::ops::Weight::values).transpose()?;
        let base = self.models.ops.linear(s, self.value(x), weight, bias)?;
        let (y, low) = match self.model.adapters.get(name) {
            Some(p) if self.strength != 0.0 => {
                let (y, low) =
                    p.forward_cached(&self.models.ops, s, self.value(x), &base, self.strength)?;
                (y, Some(low))
            }
            _ => (base, None),
        };
        Ok(self.push(y, Operation::Linear(x, name.into(), low)))
    }
    fn norm(&mut self, s: &mut Stream, x: Id, name: &str) -> Result<Id> {
        let y = self.models.ops.norm(
            s,
            self.value(x),
            self.models.transformer.get(name)?,
            Norm::OnePlusScale,
            1e-5,
        )?;
        Ok(self.push(y, Operation::Norm(x, name.into())))
    }
    fn unary(&mut self, s: &Stream, x: Id, kind: Unary) -> Result<Id> {
        let y = self.models.ops.unary(s, self.value(x), kind)?;
        Ok(self.push(y, Operation::Unary(x, kind)))
    }
    fn binary(&mut self, s: &Stream, a: Id, b: Id, kind: Binary) -> Result<Id> {
        let y = self.models.ops.binary(s, self.value(a), self.value(b), kind)?;
        Ok(self.push(y, Operation::Binary(a, b, kind)))
    }
    fn one_plus(&mut self, s: &Stream, x: Id) -> Result<Id> {
        let y = train::one_plus(&self.models.ops, s, self.value(x))?;
        Ok(self.push(y, Operation::OnePlus(x)))
    }
    fn fusion_block(
        &mut self,
        s: &mut Stream,
        x: Id,
        prefix: &str,
        sequence: usize,
    ) -> Result<Id> {
        let rows = self.value(x).rows();
        let pre = self.norm(s, x, &format!("{prefix}.prenorm.scale"))?;
        let q = self.linear(s, pre, &format!("{prefix}.attn.wq"))?;
        let k = self.linear(s, pre, &format!("{prefix}.attn.wk"))?;
        let v = self.linear(s, pre, &format!("{prefix}.attn.wv"))?;
        let q = self.reshape(q, rows * 20, 128)?;
        let k = self.reshape(k, rows * 20, 128)?;
        let q = self.norm(s, q, &format!("{prefix}.attn.qknorm.qnorm.scale"))?;
        let k = self.norm(s, k, &format!("{prefix}.attn.qknorm.knorm.scale"))?;
        let q = self.reshape(q, rows, 2560)?;
        let k = self.reshape(k, rows, 2560)?;
        let attention = train::attention_batched(
            &self.models.ops,
            s,
            self.value(q),
            self.value(k),
            self.value(v),
            sequence,
        )?;
        let attended =
            self.push(attention.output.clone(), Operation::Attention(q, k, v, attention));
        let gate = self.linear(s, pre, &format!("{prefix}.attn.gate"))?;
        let gate = self.unary(s, gate, Unary::Sigmoid)?;
        let attended = self.binary(s, attended, gate, Binary::Mul)?;
        let projected = self.linear(s, attended, &format!("{prefix}.attn.wo"))?;
        let residual = self.binary(s, x, projected, Binary::Add)?;
        let post = self.norm(s, residual, &format!("{prefix}.postnorm.scale"))?;
        let gate = self.linear(s, post, &format!("{prefix}.mlp.gate"))?;
        let gate = self.unary(s, gate, Unary::Silu)?;
        let up = self.linear(s, post, &format!("{prefix}.mlp.up"))?;
        let mixed = self.binary(s, gate, up, Binary::Mul)?;
        let down = self.linear(s, mixed, &format!("{prefix}.mlp.down"))?;
        let output = self.binary(s, residual, down, Binary::Add)?;
        s.synchronize()?;
        Ok(output)
    }
    pub(crate) fn text(&mut self, s: &mut Stream, taps: &Tensor) -> Result<Id> {
        if taps.cols() != 2560 || !taps.rows().is_multiple_of(12) {
            return Err(Error::invalid("text fusion dimensions"));
        }
        let tokens = taps.rows() / 12;
        let mut x = self.input(taps);
        for i in 0..2 {
            x = self.fusion_block(s, x, &format!("txtfusion.layerwise_blocks.{i}"), 12)?;
        }
        let permuted = train::permute_taps(&self.models.ops, s, self.value(x), false)?;
        x = self.push(permuted, Operation::Permute(x));
        x = self.linear(s, x, "txtfusion.projector")?;
        x = self.reshape(x, tokens, 2560)?;
        for i in 0..2 {
            x = self.fusion_block(s, x, &format!("txtfusion.refiner_blocks.{i}"), tokens)?;
        }
        x = self.norm(s, x, "txtmlp.0.scale")?;
        x = self.linear(s, x, "txtmlp.1")?;
        x = self.unary(s, x, Unary::Gelu)?;
        self.linear(s, x, "txtmlp.3")
    }
    pub(crate) fn time(&mut self, s: &mut Stream, sigma: f32) -> Result<(Id, Id)> {
        let features = crate::models::graph::timestep_features(sigma);
        let t = Tensor::from_slice(self.models.ops.pool(), s, &features, 1, 256)?;
        let t = self.input(&t);
        let t = self.linear(s, t, "tmlp.0")?;
        let t = self.unary(s, t, Unary::Gelu)?;
        let embedding = self.linear(s, t, "tmlp.2")?;
        let t = self.unary(s, embedding, Unary::Gelu)?;
        let modulation = self.linear(s, t, "tproj.1")?;
        Ok((embedding, modulation))
    }
    pub(crate) fn image(&mut self, s: &Stream, latents: &Tensor) -> Result<Id> {
        let x = self.input(latents);
        self.linear(s, x, "first")
    }
    pub(crate) fn last(&mut self, s: &mut Stream, x: Id, embedding: Id) -> Result<Id> {
        let table = self.models.transformer.get("last.modulation.lin")?.tensor(2, 6144)?;
        let scale = self.input(&table.view(1, 6144, 0)?);
        let shift = self.input(&table.view(1, 6144, 6144)?);
        let scale = self.binary(s, embedding, scale, Binary::Add)?;
        let shift = self.binary(s, embedding, shift, Binary::Add)?;
        let factor = self.one_plus(s, scale)?;
        let norm = self.norm(s, x, "last.norm.scale")?;
        let scaled = self.binary(s, norm, factor, Binary::Mul)?;
        let shifted = self.binary(s, scaled, shift, Binary::Add)?;
        self.linear(s, shifted, "last.linear")
    }
    pub(crate) fn backward(
        &self,
        s: &mut Stream,
        seeds: &[(Id, Tensor)],
    ) -> Result<Vec<Option<Tensor>>> {
        if self.strength != 1.0 {
            return Err(Error::invalid("training adapter strength must be one"));
        }
        let ops = &self.models.ops;
        let mut grads: Vec<Option<Tensor>> = vec![None; self.nodes.len()];
        let add = |grads: &mut Vec<Option<Tensor>>,
                   s: &Stream,
                   id: Id,
                   g: Tensor|
         -> Result<()> {
            let x = self.value(id);
            let g =
                if x.rows() == 1 && g.rows() != 1 { train::sum_rows(ops, s, &g)? } else { g };
            if (x.rows(), x.cols()) != (g.rows(), g.cols()) {
                return Err(Error::invalid("auxiliary gradient shape"));
            }
            grads[id] = Some(match grads[id].take() {
                Some(old) => train::add_scaled(ops, s, &old, &g, 1.0)?,
                None => g,
            });
            Ok(())
        };
        for (id, g) in seeds {
            add(&mut grads, s, *id, g.clone())?;
        }
        for id in (0..self.nodes.len()).rev() {
            let Some(g) = grads[id].take() else {
                continue;
            };
            match &self.nodes[id].op {
                Operation::Input => {
                    grads[id] = Some(g);
                }
                Operation::Linear(x, name, low) => {
                    let w = self.models.transformer.get(&format!("{name}.weight"))?;
                    let dx =
                        train::matmul_nn(ops, s, &g, &w.tensor(w.shape[0], w.shape[1])?, 1.0)?;
                    let dx = match (self.model.adapters.get(name), low) {
                        (Some(p), Some(low)) => {
                            p.backward_cached(ops, s, self.value(*x), &g, &dx, low)?
                        }
                        (Some(p), None) => p.backward(ops, s, self.value(*x), &g, &dx)?,
                        (None, _) => dx,
                    };
                    add(&mut grads, s, *x, dx)?;
                }
                Operation::Norm(x, name) => {
                    let scales = self.models.transformer.get(name)?.f32_values(s)?;
                    let dx = train::norm_backward(ops, s, self.value(*x), &g, scales, 1e-5)?;
                    add(&mut grads, s, *x, dx)?;
                }
                Operation::Unary(x, kind) => {
                    let dx = match kind {
                        Unary::Gelu => train::gelu_backward(ops, s, self.value(*x), &g)?,
                        _ => train::activation_backward(
                            ops,
                            s,
                            self.value(*x),
                            &g,
                            *kind == Unary::Sigmoid,
                        )?,
                    };
                    add(&mut grads, s, *x, dx)?;
                }
                Operation::OnePlus(x) => add(&mut grads, s, *x, g)?,
                Operation::Binary(a, b, kind) => {
                    let (ga, gb) = if *kind == Binary::Add {
                        (g.clone(), g)
                    } else {
                        (
                            ops.binary(s, &g, self.value(*b), Binary::Mul)?,
                            ops.binary(s, &g, self.value(*a), Binary::Mul)?,
                        )
                    };
                    add(&mut grads, s, *a, ga)?;
                    add(&mut grads, s, *b, gb)?;
                }
                Operation::Reshape(x) => add(
                    &mut grads,
                    s,
                    *x,
                    g.view(self.value(*x).rows(), self.value(*x).cols(), 0)?,
                )?,
                Operation::Permute(x) => {
                    add(&mut grads, s, *x, train::permute_taps(ops, s, &g, true)?)?
                }
                Operation::Attention(q, k, v, forward) => {
                    let (dq, dk, dv) = train::attention_backward(
                        ops,
                        s,
                        self.value(*q),
                        self.value(*k),
                        self.value(*v),
                        forward,
                        &g,
                    )?;
                    add(&mut grads, s, *q, dq)?;
                    add(&mut grads, s, *k, dk)?;
                    add(&mut grads, s, *v, dv)?;
                }
            }
            // Keep queued scratch ownership bounded while all forward nodes remain live.
            if id.is_multiple_of(24) {
                s.synchronize()?;
            }
        }
        Ok(grads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lora::{Adapter, Targets},
        numerics::{from_f32, to_f32},
        training::{TrainConfig, optimizer},
    };

    #[test]
    #[ignore = "requires GPU and KREA2_RAW_CHECKPOINT; loads auxiliary towers only"]
    fn full_targets_real_auxiliary_forward_and_all_forty_gradients() {
        // Keep the desktop reserve even though this test never loads the main blocks.
        super::super::memory::before_load(5usize << 30).unwrap();
        let path = std::env::var_os("KREA2_RAW_CHECKPOINT").expect("set KREA2_RAW_CHECKPOINT");
        let mut stream = Stream::open().unwrap();
        let models = Models::load_parts(
            &mut stream,
            std::path::Path::new(&path),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let mut adapter = Adapter::initialize_targets(32, 32.0, 37, Targets::All).unwrap();
        adapter.layers.retain(|name, _| !name.starts_with("blocks."));
        assert_eq!(adapter.layers.len(), 40);
        let model =
            Transformer::auxiliary_test_model(&models.ops, &mut stream, &adapter).unwrap();
        let upload = |s: &mut Stream, rows, cols, seed| {
            let bits: Vec<_> = (0..rows * cols)
                .map(|i| from_f32(((i * 17 + seed) % 71) as f32 / 71.0 - 0.5))
                .collect();
            Tensor::from_slice(models.ops.pool(), s, &bits, rows, cols).unwrap()
        };
        let taps = upload(&mut stream, 3 * 12, 2560, 7);
        let latent = upload(&mut stream, 5, 64, 13);
        let hidden = upload(&mut stream, 5, 6144, 19);
        let reference_text = models.text_fusion(&mut stream, &taps).unwrap();
        let (reference_time, reference_mod) =
            models.time_continuous(&mut stream, 0.501).unwrap();
        let reference_image = models.image_in(&stream, &latent).unwrap();
        let reference_last = models.last(&mut stream, &hidden, &reference_time).unwrap();
        let parameters: Vec<_> = model.adapters.values().flat_map(|p| [&p.a, &p.b]).collect();
        let config = TrainConfig::default();
        for step in 1..=2 {
            let mut tape = Tape::new(&models, &model, 1.0);
            let text = tape.text(&mut stream, &taps).unwrap();
            let (embedding, modulation) = tape.time(&mut stream, 0.501).unwrap();
            let image = tape.image(&stream, &latent).unwrap();
            let input = tape.input(&hidden);
            let last = tape.last(&mut stream, input, embedding).unwrap();
            if step == 1 {
                for (name, id, expected) in [
                    ("text", text, &reference_text),
                    ("time", embedding, &reference_time),
                    ("modulation", modulation, &reference_mod),
                    ("image", image, &reference_image),
                    ("last", last, &reference_last),
                ] {
                    let actual = tape.value(id).download(&mut stream).unwrap();
                    let expected = expected.download(&mut stream).unwrap();
                    let error: f64 = actual
                        .iter()
                        .zip(&expected)
                        .map(|(&a, &b)| f64::from(to_f32(a) - to_f32(b)).powi(2))
                        .sum();
                    let norm: f64 =
                        expected.iter().map(|&b| f64::from(to_f32(b)).powi(2)).sum();
                    eprintln!("{name} zero-adapter relative L2 {}", (error / norm).sqrt());
                    assert!(
                        error <= norm * 0.0004 + 1e-10,
                        "{name} differs from frozen reference"
                    );
                }
            }
            let mut seeds = Vec::new();
            for id in [text, modulation, image, last] {
                seeds.push((
                    id,
                    upload(&mut stream, tape.value(id).rows(), tape.value(id).cols(), id + 3),
                ));
            }
            let grads = tape.backward(&mut stream, &seeds).unwrap();
            assert!(grads[input].is_some());
            for (name, p) in &model.adapters {
                for (part, param) in [("A", &p.a), ("B", &p.b)] {
                    let grad = param.grad.download(&mut stream).unwrap();
                    assert!(grad.iter().all(|v| v.is_finite()), "{name} {part} nonfinite");
                    if part == "B" || step == 2 {
                        assert!(grad.iter().any(|v| *v != 0.0), "{name} {part} disconnected");
                    } else {
                        assert!(grad.iter().all(|v| *v == 0.0), "zero B should give zero dA");
                    }
                }
            }
            let cached: Vec<_> =
                parameters.iter().map(|p| p.grad.download(&mut stream).unwrap()).collect();
            for p in &parameters {
                p.grad.clear(&stream).unwrap();
            }
            for node in &mut tape.nodes {
                if let Operation::Linear(_, _, low) = &mut node.op {
                    *low = None;
                }
            }
            let reference = tape.backward(&mut stream, &seeds).unwrap();
            for (actual, expected) in grads.iter().zip(&reference) {
                match (actual, expected) {
                    (Some(a), Some(b)) => assert_eq!(
                        a.download(&mut stream).unwrap(),
                        b.download(&mut stream).unwrap()
                    ),
                    (None, None) => {}
                    _ => panic!("cached input gradient connectivity"),
                }
            }
            for (p, expected) in parameters.iter().zip(&cached) {
                assert_eq!(&p.grad.download(&mut stream).unwrap(), expected);
            }
            optimizer::update_parameters(&models.ops, &mut stream, &parameters, &config, step)
                .unwrap();
            stream.synchronize().unwrap();
        }
    }
}
