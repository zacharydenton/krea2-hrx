//! Independent numerical oracles for training kernels. GPU execution is opt-in.
use hrx::{BufferPool, Stream};
use krea2::lora::Factors;
use krea2::numerics::{from_f32, to_f32};
use krea2::ops::{Ops, Tensor};
use krea2::training::{
    TrainConfig,
    model::Projection,
    ops::{self, FloatTensor},
    optimizer,
};

fn values(n: usize, offset: usize) -> Vec<f32> {
    (0..n).map(|i| to_f32(from_f32(((i * 17 + offset) % 71) as f32 / 71.0 - 0.5))).collect()
}
fn upload(ops: &Ops, stream: &mut Stream, rows: usize, cols: usize, values: &[f32]) -> Tensor {
    Tensor::from_slice(
        ops.pool(),
        stream,
        &values.iter().copied().map(from_f32).collect::<Vec<_>>(),
        rows,
        cols,
    )
    .unwrap()
}
fn read(x: &Tensor, stream: &mut Stream) -> Vec<f32> {
    x.download(stream).unwrap().into_iter().map(to_f32).collect()
}
fn close(got: &[f32], expected: &[f32], relative: f64, absolute: f64) {
    assert_eq!(got.len(), expected.len());
    let mut error = 0.0;
    let mut norm = 0.0;
    for (&a, &b) in got.iter().zip(expected) {
        assert!(a.is_finite());
        error += (f64::from(a) - f64::from(b)).powi(2);
        norm += f64::from(b).powi(2);
    }
    let rms = (error / got.len() as f64).sqrt();
    assert!(
        rms <= absolute + relative * (norm / got.len() as f64).sqrt(),
        "RMS error {rms}, reference norm {}",
        (norm / got.len() as f64).sqrt()
    );
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn flow_loss_matches_cpu_mse_and_rejects_nonfinite_inputs() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for count in [1, 1023, 1024, 1031] {
        let prediction = values(count, 9);
        let target: Vec<f32> = values(count, 7).into_iter().map(|v| v + 0.001).collect();
        let p = upload(&ops, &mut stream, 1, count, &prediction);
        let t = FloatTensor::from_slice(&mut stream, 1, count, &target).unwrap();
        let (loss, gradient) = ops::flow_loss(&ops, &mut stream, &p, &t).unwrap();
        let squared: f64 =
            prediction.iter().zip(&target).map(|(p, t)| f64::from(p - t).powi(2)).sum();
        assert!((loss - squared / count as f64).abs() < 1e-6);
        let expected: Vec<f32> =
            prediction.iter().zip(&target).map(|(p, t)| 2.0 * (p - t) / count as f32).collect();
        close(&read(&gradient, &mut stream), &expected, 0.004, 1e-7);
    }
    let p = upload(&ops, &mut stream, 1, 1, &[f32::INFINITY]);
    let t = FloatTensor::zero(&stream, 1, 1).unwrap();
    assert!(ops::flow_loss(&ops, &mut stream, &p, &t).is_err());
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn zero_b_preserves_base_output_and_accumulates_only_b_gradients() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let (m, k, n, rank) = (7, 19, 11, 3);
    let p = Projection::new(
        &ops,
        &mut stream,
        &Factors {
            inputs: k,
            outputs: n,
            rank,
            alpha: 1.5,
            a: values(rank * k, 5),
            b: vec![0.0; n * rank],
        },
    )
    .unwrap();
    let x = upload(&ops, &mut stream, m, k, &values(m * k, 2));
    let base_values = values(m * n, 11);
    let base = upload(&ops, &mut stream, m, n, &base_values);
    for strength in [-0.5, 0.0, 0.9, 1.1] {
        let y = p.forward(&ops, &stream, &x, &base, strength).unwrap();
        assert_eq!(read(&y, &mut stream), base_values);
    }
    let grad = upload(&ops, &mut stream, m, n, &values(m * n, 7));
    let base_gradient = values(m * k, 13);
    let dx = upload(&ops, &mut stream, m, k, &base_gradient);
    let first = p.backward(&ops, &stream, &x, &grad, &dx).unwrap();
    assert_eq!(read(&first, &mut stream), base_gradient);
    assert!(p.a.grad.download(&mut stream).unwrap().iter().all(|v| *v == 0.0));
    let db = p.b.grad.download(&mut stream).unwrap();
    assert!(db.iter().any(|v| v.abs() > 1e-3));
    p.backward(&ops, &stream, &x, &grad, &dx).unwrap();
    close(
        &p.b.grad.download(&mut stream).unwrap(),
        &db.iter().map(|v| v * 2.0).collect::<Vec<_>>(),
        1e-6,
        1e-7,
    );
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn lora_backward_and_adam_match_independent_dense_algebra() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let (m, k, n, r) = (7, 19, 11, 3);
    let x = values(m * k, 2);
    let g = values(m * n, 7);
    let a = values(r * k, 5);
    let b = values(n * r, 11);
    let factors =
        Factors { inputs: k, outputs: n, rank: r, alpha: 1.5, a: a.clone(), b: b.clone() };
    let p = Projection::new(&ops, &mut stream, &factors).unwrap();
    let xt = upload(&ops, &mut stream, m, k, &x);
    let gt = upload(&ops, &mut stream, m, n, &g);
    let zero = upload(&ops, &mut stream, m, k, &vec![0.0; m * k]);
    let dx = p.backward(&ops, &stream, &xt, &gt, &zero).unwrap();
    let mut low = vec![0.0; m * r];
    let mut dl = vec![0.0; m * r];
    for i in 0..m {
        for j in 0..r {
            low[i * r + j] = (0..k).map(|c| x[i * k + c] * a[j * k + c]).sum::<f32>();
            dl[i * r + j] = 0.5 * (0..n).map(|c| g[i * n + c] * b[c * r + j]).sum::<f32>();
        }
    }
    let base_values = values(m * n, 13);
    let base = upload(&ops, &mut stream, m, n, &base_values);
    for strength in [-0.5, 0.0, 1.0, 1.1] {
        let y = p.forward(&ops, &stream, &xt, &base, strength).unwrap();
        let expected: Vec<f32> = (0..m * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                base_values[index]
                    + 0.5
                        * strength
                        * (0..r).map(|j| low[row * r + j] * b[col * r + j]).sum::<f32>()
            })
            .collect();
        close(&read(&y, &mut stream), &expected, 0.02, 2e-4);
    }
    let mut da = vec![0.0; r * k];
    let mut db = vec![0.0; n * r];
    let mut dx_ref = vec![0.0; m * k];
    for j in 0..r {
        for c in 0..k {
            da[j * k + c] = (0..m).map(|i| dl[i * r + j] * x[i * k + c]).sum();
        }
    }
    for o in 0..n {
        for j in 0..r {
            db[o * r + j] = 0.5 * (0..m).map(|i| g[i * n + o] * low[i * r + j]).sum::<f32>();
        }
    }
    for i in 0..m {
        for c in 0..k {
            dx_ref[i * k + c] = (0..r).map(|j| dl[i * r + j] * a[j * k + c]).sum();
        }
    }
    close(&read(&dx, &mut stream), &dx_ref, 0.02, 2e-4);
    close(&p.a.grad.download(&mut stream).unwrap(), &da, 0.02, 2e-4);
    close(&p.b.grad.download(&mut stream).unwrap(), &db, 0.02, 2e-4);
    let gradient = p.a.grad.download(&mut stream).unwrap();
    let c = TrainConfig::default();
    optimizer::adamw(&ops, &stream, &p.a, &c, 1.0, 1.0 - c.beta1, 1.0 - c.beta2).unwrap();
    let expected: Vec<f32> = a
        .iter()
        .zip(&gradient)
        .map(|(&w, &g)| {
            w * (1.0 - c.learning_rate * c.weight_decay)
                - c.learning_rate * g / (g.abs() + c.epsilon)
        })
        .collect();
    close(&p.a.master.download(&mut stream).unwrap(), &expected, 1e-5, 1e-7);
    assert!(p.a.grad.download(&mut stream).unwrap().iter().all(|v| *v == 0.0));
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn rmsnorm_and_rotary_backward_match_cpu_derivatives() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for cols in [128, 6144] {
        let rows = 3;
        let x = values(rows * cols, 4);
        let g = values(rows * cols, 9);
        let w = values(cols, 3);
        let xt = upload(&ops, &mut stream, rows, cols, &x);
        let gt = upload(&ops, &mut stream, rows, cols, &g);
        let wt = FloatTensor::from_slice(&mut stream, 1, cols, &w).unwrap();
        let result = ops::norm_backward(&ops, &stream, &xt, &gt, wt.binding(), 1e-5).unwrap();
        let mut expected = vec![0.0; rows * cols];
        for row in 0..rows {
            let xx = &x[row * cols..(row + 1) * cols];
            let gg = &g[row * cols..(row + 1) * cols];
            let var =
                xx.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / cols as f64 + 1e-5;
            let dot = (0..cols)
                .map(|j| f64::from(xx[j]) * f64::from(gg[j]) * (1.0 + f64::from(w[j])))
                .sum::<f64>()
                / cols as f64;
            for j in 0..cols {
                expected[row * cols + j] = ((f64::from(gg[j]) * (1.0 + f64::from(w[j]))
                    - f64::from(xx[j]) * dot / var)
                    / var.sqrt()) as f32;
            }
        }
        close(&read(&result, &mut stream), &expected, 0.004, 1e-5);
    }
    let x = values(3 * 256, 11);
    let xt = upload(&ops, &mut stream, 3, 256, &x);
    let cos = FloatTensor::from_slice(&mut stream, 3, 128, &vec![0.6; 384]).unwrap();
    let sin = FloatTensor::from_slice(&mut stream, 3, 128, &vec![0.8; 384]).unwrap();
    let y = ops::rope(&ops, &stream, &xt, &cos, &sin, false).unwrap();
    let inverse = ops::rope(&ops, &stream, &y, &cos, &sin, true).unwrap();
    close(&read(&inverse, &mut stream), &x, 0.005, 1e-4);
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn streaming_gqa_forward_and_backward_match_materialized_f64_attention() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let (t, heads, kv, d) = (9, 8, 2, 128);
    let q = values(t * heads * d, 3);
    let k = values(t * kv * d, 5);
    let v = values(t * kv * d, 7);
    let g = values(q.len(), 13);
    let qt = upload(&ops, &mut stream, t, heads * d, &q);
    let kt = upload(&ops, &mut stream, t, kv * d, &k);
    let vt = upload(&ops, &mut stream, t, kv * d, &v);
    let gt = upload(&ops, &mut stream, t, heads * d, &g);
    let mut f = ops::attention(&ops, &stream, &qt, &kt, &vt).unwrap();
    let (dq, dk, dv) = ops::attention_backward(&ops, &stream, &qt, &kt, &vt, &f, &gt).unwrap();
    let mut output = vec![0.0; q.len()];
    let mut eq = vec![0.0f64; q.len()];
    let mut ek = vec![0.0f64; k.len()];
    let mut ev = vec![0.0f64; v.len()];
    let scale = 1.0 / (d as f64).sqrt();
    for i in 0..t {
        for h in 0..heads {
            let hi = h / (heads / kv);
            let qi = (i * heads + h) * d;
            let mut scores = (0..t)
                .map(|j| {
                    let kj = (j * kv + hi) * d;
                    (0..d).map(|c| f64::from(q[qi + c]) * f64::from(k[kj + c])).sum::<f64>()
                        * scale
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let sum = scores.iter().sum::<f64>();
            for s in &mut scores {
                *s /= sum;
            }
            let mut o = vec![0.0; d];
            for c in 0..d {
                o[c] = (0..t).map(|j| scores[j] * f64::from(v[(j * kv + hi) * d + c])).sum();
                output[qi + c] = o[c] as f32;
            }
            let delta = (0..d).map(|c| o[c] * f64::from(g[qi + c])).sum::<f64>();
            for (j, &probability) in scores.iter().enumerate() {
                let kj = (j * kv + hi) * d;
                let dp =
                    (0..d).map(|c| f64::from(g[qi + c]) * f64::from(v[kj + c])).sum::<f64>();
                let ds = probability * (dp - delta) * scale;
                for c in 0..d {
                    eq[qi + c] += ds * f64::from(k[kj + c]);
                    ek[kj + c] += ds * f64::from(q[qi + c]);
                    ev[kj + c] += probability * f64::from(g[qi + c]);
                }
            }
        }
    }
    close(&read(&f.output, &mut stream), &output, 0.005, 1e-5);
    for (got, reference) in [(dq, eq), (dk, ek), (dv, ev)] {
        close(
            &read(&got, &mut stream),
            &reference.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
            0.005,
            1e-5,
        );
    }
    // Public output replacement cannot disguise smaller private forward statistics.
    let larger_q = upload(&ops, &mut stream, t + 1, heads * d, &values((t + 1) * heads * d, 3));
    let larger_kv = upload(&ops, &mut stream, t + 1, kv * d, &values((t + 1) * kv * d, 5));
    f.output = larger_q.clone();
    assert!(
        ops::attention_backward(
            &ops, &stream, &larger_q, &larger_kv, &larger_kv, &f, &larger_q
        )
        .is_err()
    );
}
