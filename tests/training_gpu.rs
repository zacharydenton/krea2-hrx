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
#[ignore = "requires a gfx1151 GPU"]
fn modulation_backward_matches_separate_reductions_and_preserves_other_rows() {
    use krea2::ops::Binary;
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in [(1, 1), (7, 19), (31, 255), (33, 257), (1043, 64), (4115, 65)] {
        let inputs = [3, 7, 11, 19].map(|offset| values(rows * cols, offset));
        let [r, b, a, n] = std::array::from_fn::<_, 4, _>(|i| {
            upload(&ops, &mut stream, rows, cols, &inputs[i])
        });
        let mut expected = values(7 * cols, 23);
        let dst = FloatTensor::from_slice(&mut stream, 7, cols, &expected).unwrap();
        let reference = FloatTensor::from_slice(&mut stream, 7, cols, &expected).unwrap();
        for row in [3, 0, 3] {
            ops::modulation_backward(&ops, &stream, &r, &b, &a, &n, &dst, row).unwrap();
            let gate = ops.binary(&stream, &r, &b, Binary::Mul).unwrap();
            let scale = ops.binary(&stream, &a, &n, Binary::Mul).unwrap();
            ops::sum_rows_accumulate(&ops, &stream, &scale, &reference, row).unwrap();
            ops::sum_rows_accumulate(&ops, &stream, &a, &reference, row + 1).unwrap();
            ops::sum_rows_accumulate(&ops, &stream, &gate, &reference, row + 2).unwrap();
            for c in 0..cols {
                let mut sum = [0.0f32; 3];
                for r in 0..rows {
                    let i = r * cols + c;
                    sum[0] += to_f32(from_f32(inputs[2][i] * inputs[3][i]));
                    sum[1] += inputs[2][i];
                    sum[2] += to_f32(from_f32(inputs[0][i] * inputs[1][i]));
                }
                for j in 0..3 {
                    expected[(row + j) * cols + c] += sum[j];
                }
            }
            let actual = dst.download(&mut stream).unwrap();
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(actual.clone()), bits(reference.download(&mut stream).unwrap()));
            assert_eq!(bits(actual), bits(expected.clone()));
        }
        for row in [5, usize::MAX] {
            assert!(
                ops::modulation_backward(&ops, &stream, &r, &b, &a, &n, &dst, row).is_err()
            );
        }
        let short = FloatTensor::zero(&stream, 2, cols).unwrap();
        assert!(ops::modulation_backward(&ops, &stream, &r, &b, &a, &n, &short, 0).is_err());
        let mismatch = upload(&ops, &mut stream, 1, cols + 1, &vec![0.0; cols + 1]);
        for inputs in [(&mismatch, &b, &n), (&r, &mismatch, &n), (&r, &b, &mismatch)] {
            assert!(
                ops::modulation_backward(
                    &ops, &stream, inputs.0, inputs.1, &a, inputs.2, &dst, 0
                )
                .is_err()
            );
        }
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn gated_backward_preserves_product_rounding_and_matches_chain_rule() {
    use krea2::ops::Binary;
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in [(1, 1), (1, 31), (1, 255), (1, 256), (1, 257), (3, 6144)] {
        let size = rows * cols;
        let x: Vec<_> = values(size, 3).into_iter().map(|v| v * 32.0).collect();
        // Supply an independent cached activation so recomputing it would fail.
        let activation = values(size, 7);
        let value = values(size, 11);
        let grad = values(size, 19);
        let xt = upload(&ops, &mut stream, rows, cols, &x);
        let at = upload(&ops, &mut stream, rows, cols, &activation);
        let vt = upload(&ops, &mut stream, rows, cols, &value);
        let gt = upload(&ops, &mut stream, rows, cols, &grad);
        for sigmoid in [false, true] {
            let (dx, dv) =
                ops::gated_backward(&ops, &stream, &xt, &at, &vt, &gt, sigmoid).unwrap();
            let baseline_v = ops.binary(&stream, &gt, &at, Binary::Mul).unwrap();
            let product = ops.binary(&stream, &gt, &vt, Binary::Mul).unwrap();
            let baseline_x =
                ops::activation_backward(&ops, &stream, &xt, &product, sigmoid).unwrap();
            assert_eq!(
                dx.download(&mut stream).unwrap(),
                baseline_x.download(&mut stream).unwrap()
            );
            assert_eq!(
                dv.download(&mut stream).unwrap(),
                baseline_v.download(&mut stream).unwrap()
            );
            let expected: Vec<_> = (0..size)
                .map(|i| {
                    let sig = 1.0 / (1.0 + (-f64::from(x[i])).exp());
                    let derivative = if sigmoid {
                        sig * (1.0 - sig)
                    } else {
                        sig + f64::from(x[i]) * sig * (1.0 - sig)
                    };
                    (f64::from(to_f32(from_f32(grad[i] * value[i]))) * derivative) as f32
                })
                .collect();
            close(&read(&dx, &mut stream), &expected, 0.004, 1e-7);
        }
        let mismatch = upload(&ops, &mut stream, 1, size + 1, &vec![0.0; size + 1]);
        for inputs in [(&mismatch, &vt, &gt), (&at, &mismatch, &gt), (&at, &vt, &mismatch)] {
            assert!(
                ops::gated_backward(&ops, &stream, &xt, inputs.0, inputs.1, inputs.2, false)
                    .is_err()
            );
        }
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn dense_training_gemm_matches_cpu_across_dispatch_and_tile_boundaries() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (m, n, k) in [
        (511, 64, 64),
        (512, 64, 64),
        (513, 128, 192),
        (513, 64, 64),
        (513, 192, 64),
        (513, 64, 6144),
        (641, 192, 128),
        (769, 192, 64),
        (1043, 64, 128),
        (1070, 64, 192),
        (4115, 64, 64),
        (4115, 128, 192),
        (513, 65, 63),
    ] {
        let a = values(m * k, 3);
        let b = values(n * k, 11);
        let x = upload(&ops, &mut stream, m, k, &a);
        let w = upload(&ops, &mut stream, n, k, &b);
        let out = ops::matmul(&ops, &stream, &x, &w, -0.75).unwrap();
        let mut expected = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                expected[row * n + col] =
                    (0..k).map(|j| a[row * k + j] * b[col * k + j]).sum::<f32>() * -0.75;
            }
        }
        close(&read(&out, &mut stream), &expected, 0.004, 1e-6);
        let wt = ops::transpose(&ops, &stream, &w).unwrap();
        let nn = ops::matmul_nn(&ops, &stream, &x, &wt, -0.75).unwrap();
        close(&read(&nn, &mut stream), &expected, 0.004, 1e-6);
        if m >= 512 && n % 64 == 0 && k % 64 == 0 {
            assert_eq!(out.download(&mut stream).unwrap(), nn.download(&mut stream).unwrap());
        }
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn adapter_gradient_accumulation_matches_cpu_across_ragged_tiles() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (k, m, n) in [
        (1, 1, 1),
        (7, 3, 19),
        (7, 32, 32),
        (31, 64, 32),
        (32, 32, 32),
        (33, 32, 64),
        (31, 33, 65),
        (33, 65, 32),
        (1070, 32, 64),
        (4115, 64, 32),
    ] {
        let a = values(k * m, 5);
        let b = values(k * n, 11);
        let av = upload(&ops, &mut stream, k, m, &a);
        let bv = upload(&ops, &mut stream, k, n, &b);
        let mut expected = values(m * n, 13);
        let dst = FloatTensor::from_slice(&mut stream, m, n, &expected).unwrap();
        for alpha in [-0.75, 0.25, 0.0] {
            ops::matmul_tn_accumulate(&ops, &stream, &av, &bv, &dst, alpha).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let dot = (0..k)
                        .map(|j| f64::from(a[j * m + row]) * f64::from(b[j * n + col]))
                        .sum::<f64>() as f32;
                    expected[row * n + col] += alpha * dot;
                }
            }
            close(&dst.download(&mut stream).unwrap(), &expected, 2e-5, 1e-6);
        }
        assert!(ops::matmul_tn_accumulate(&ops, &stream, &av, &bv, &dst, f32::NAN).is_err());
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn float_cast_uses_logical_dimensions_with_extra_allocation_capacity() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let source = stream.allocate(32).unwrap();
    let values = [0.25f32, -2.5, 1.0078125, 3.125];
    stream.fill(source.binding(), 0xff).unwrap();
    stream.upload(source.try_slice(0, 16).unwrap(), bytemuck::cast_slice(&values)).unwrap();
    let out = ops::cast_view(&ops, &stream, source.binding(), 2, 2).unwrap();
    assert_eq!(out.download(&mut stream).unwrap(), values.map(from_f32));
    assert!(ops::cast_view(&ops, &stream, source.binding(), 3, 3).is_err());
    assert!(ops::cast_view(&ops, &stream, source.binding(), usize::MAX, 2).is_err());
    assert!(ops::cast_view(&ops, &stream, source.binding(), 0, 2).is_err());
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn transpose_preserves_bits_across_tiles_and_ragged_edges() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in [(1, 1), (1, 65), (65, 1), (7, 19), (32, 32), (33, 65), (1043, 128)] {
        // Include signed zero, infinities and NaN payloads: transpose is a copy.
        let bits: Vec<u16> = (0..rows * cols).map(|i| (i * 31337) as u16).collect();
        let input = Tensor::from_slice(ops.pool(), &mut stream, &bits, rows, cols).unwrap();
        let out = ops::transpose(&ops, &stream, &input).unwrap();
        let actual = out.download(&mut stream).unwrap();
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(actual[col * rows + row], bits[row * cols + col]);
            }
        }
    }
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
    for (m, k, n, r) in
        [(7, 19, 11, 3), (65, 128, 64, 32), (513, 64, 128, 64), (257, 12, 1, 32)]
    {
        let x = values(m * k, 2);
        let g = values(m * n, 7);
        let a = values(r * k, 5);
        let b = values(n * r, 11);
        let factors = Factors {
            inputs: k,
            outputs: n,
            rank: r,
            alpha: r as f32 * 0.5,
            a: a.clone(),
            b: b.clone(),
        };
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
                db[o * r + j] =
                    0.5 * (0..m).map(|i| g[i * n + o] * low[i * r + j]).sum::<f32>();
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
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn lora_cached_activation_preserves_outputs_and_accumulated_gradients() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (m, k, n, rank, zero_b) in [
        (7, 19, 11, 3, false),
        (65, 128, 64, 32, false),
        (513, 64, 128, 64, false),
        (7, 19, 11, 3, true),
    ] {
        let p = Projection::new(
            &ops,
            &mut stream,
            &Factors {
                inputs: k,
                outputs: n,
                rank,
                alpha: rank as f32 * 0.5,
                a: values(rank * k, 5),
                b: if zero_b { vec![0.0; n * rank] } else { values(n * rank, 11) },
            },
        )
        .unwrap();
        let x = upload(&ops, &mut stream, m, k, &values(m * k, 2));
        let grad = upload(&ops, &mut stream, m, n, &values(m * n, 7));
        let base = upload(&ops, &mut stream, m, n, &values(m * n, 13));
        let dx = upload(&ops, &mut stream, m, k, &values(m * k, 17));
        let (candidate, low) = p.forward_cached(&ops, &stream, &x, &base, 1.0).unwrap();
        let reference = p.forward(&ops, &stream, &x, &base, 1.0).unwrap();
        assert_eq!(
            candidate.download(&mut stream).unwrap(),
            reference.download(&mut stream).unwrap()
        );
        let expected_low = ops::matmul(&ops, &stream, &x, &p.a.value, 1.0).unwrap();
        assert_eq!(
            low.download(&mut stream).unwrap(),
            expected_low.download(&mut stream).unwrap()
        );
        for _ in 0..2 {
            p.backward(&ops, &stream, &x, &grad, &dx).unwrap();
        }
        let a = p.a.grad.download(&mut stream).unwrap();
        let b = p.b.grad.download(&mut stream).unwrap();
        p.a.grad.clear(&stream).unwrap();
        p.b.grad.clear(&stream).unwrap();
        for _ in 0..2 {
            let candidate = p.backward_cached(&ops, &stream, &x, &grad, &dx, &low).unwrap();
            // Recompute dX without changing the accumulated parameter gradients.
            let dl = ops::matmul_nn(&ops, &stream, &grad, &p.b.value, 0.5).unwrap();
            let branch = ops::matmul_nn(&ops, &stream, &dl, &p.a.value, 1.0).unwrap();
            let reference = ops::add_scaled(&ops, &stream, &dx, &branch, 1.0).unwrap();
            assert_eq!(
                candidate.download(&mut stream).unwrap(),
                reference.download(&mut stream).unwrap()
            );
        }
        assert_eq!(p.a.grad.download(&mut stream).unwrap(), a);
        assert_eq!(p.b.grad.download(&mut stream).unwrap(), b);
        let bad_low = upload(&ops, &mut stream, m, rank + 1, &vec![0.0; m * (rank + 1)]);
        assert!(p.backward_cached(&ops, &stream, &x, &grad, &dx, &bad_low).is_err());
        assert!(p.backward_cached(&ops, &stream, &x, &grad, &bad_low, &low).is_err());
        assert!(p.backward_cached(&ops, &stream, &x, &bad_low, &dx, &low).is_err());
        assert!(p.backward_cached(&ops, &stream, &bad_low, &grad, &dx, &low).is_err());
        // Invalid calls must not partially accumulate either parameter gradient.
        assert_eq!(p.a.grad.download(&mut stream).unwrap(), a);
        assert_eq!(p.b.grad.download(&mut stream).unwrap(), b);
    }
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn prepared_optimizer_matches_updates_and_rejects_nonfinite_gradients_atomically() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let factors = [
        Factors {
            inputs: 19,
            outputs: 11,
            rank: 3,
            alpha: 1.5,
            a: values(57, 2),
            b: values(33, 3),
        },
        Factors {
            inputs: 65,
            outputs: 33,
            rank: 32,
            alpha: 16.0,
            a: values(2080, 5),
            b: values(1056, 7),
        },
    ];
    let direct: Vec<_> =
        factors.iter().map(|f| Projection::new(&ops, &mut stream, f).unwrap()).collect();
    let replay: Vec<_> =
        factors.iter().map(|f| Projection::new(&ops, &mut stream, f).unwrap()).collect();
    let dp: Vec<_> = direct.iter().flat_map(|p| [&p.a, &p.b]).collect();
    let rp: Vec<_> = replay.iter().flat_map(|p| [&p.a, &p.b]).collect();
    let mut prepared = optimizer::PreparedOptimizer::new(&stream, &rp).unwrap();
    assert!(optimizer::PreparedOptimizer::new(&stream, &[]).is_err());
    assert!(optimizer::PreparedOptimizer::new(&stream, &[rp[0], rp[0]]).is_err());
    let mut config = TrainConfig::default();
    for step in 1..=4 {
        config.accumulation = if step % 2 == 0 { 3 } else { 1 };
        config.max_grad_norm = if step % 2 == 0 { 0.05 } else { 100.0 };
        config.learning_rate = 1e-4 * step as f32;
        let mut norm_squared = 0.0f64;
        for (index, (a, b)) in dp.iter().zip(&rp).enumerate() {
            let g = values(a.grad.size(), index + step);
            norm_squared += g.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            for p in [a, b] {
                stream.upload(p.grad.binding(), bytemuck::cast_slice(&g)).unwrap();
            }
        }
        let a = optimizer::update_parameters(&ops, &mut stream, &dp, &config, step).unwrap();
        let b = prepared.update(&mut stream, &config, step).unwrap();
        assert_eq!(a, b);
        assert!((b - norm_squared.sqrt() / config.accumulation as f64).abs() < 1e-5);
        for (a, b) in dp.iter().zip(&rp) {
            for (x, y) in [
                (&a.master, &b.master),
                (&a.first, &b.first),
                (&a.second, &b.second),
                (&a.grad, &b.grad),
            ] {
                assert_eq!(x.download(&mut stream).unwrap(), y.download(&mut stream).unwrap());
            }
            assert_eq!(
                a.value.download(&mut stream).unwrap(),
                b.value.download(&mut stream).unwrap()
            );
            assert!(b.grad.download(&mut stream).unwrap().iter().all(|v| *v == 0.0));
        }
    }
    // A late invalid gradient must not partially update earlier parameters.
    for invalid in [f32::NAN, f32::INFINITY, f32::MAX] {
        for (i, p) in rp.iter().enumerate() {
            let mut g = values(p.grad.size(), i + 1);
            if i == rp.len() - 1 {
                *g.last_mut().unwrap() = invalid;
            }
            stream.upload(p.grad.binding(), bytemuck::cast_slice(&g)).unwrap();
        }
        let snapshot = |stream: &mut Stream| -> Vec<u32> {
            rp.iter()
                .flat_map(|p| {
                    let mut bits = [&p.master, &p.first, &p.second, &p.grad]
                        .into_iter()
                        .flat_map(|t| t.download(stream).unwrap().into_iter().map(f32::to_bits))
                        .collect::<Vec<_>>();
                    bits.extend(p.value.download(stream).unwrap().into_iter().map(u32::from));
                    bits
                })
                .collect()
        };
        let before = snapshot(&mut stream);
        assert!(prepared.update(&mut stream, &config, 5).is_err());
        assert_eq!(before, snapshot(&mut stream));
    }
    // A validation failure leaves the graph reusable after correcting gradients.
    for p in dp.iter().chain(&rp) {
        p.grad.clear(&stream).unwrap();
    }
    assert_eq!(optimizer::update_parameters(&ops, &mut stream, &dp, &config, 5).unwrap(), 0.0);
    assert_eq!(prepared.update(&mut stream, &config, 5).unwrap(), 0.0);
    for (a, b) in dp.iter().zip(&rp) {
        assert_eq!(
            a.master.download(&mut stream).unwrap(),
            b.master.download(&mut stream).unwrap()
        );
    }
    assert!(prepared.update(&mut stream, &config, 0).is_err());
    config.accumulation = 0;
    assert!(prepared.update(&mut stream, &config, 5).is_err());
    // Reject aliases within a parameter, not only duplicate parameter references.
    let mut aliased = Projection::new(&ops, &mut stream, &factors[0]).unwrap();
    aliased.a.first = aliased.a.master.clone();
    assert!(optimizer::PreparedOptimizer::new(&stream, &[&aliased.a]).is_err());

    // Graph ownership must retain pooled leases as well as native allocations.
    // Otherwise dropping the projection lets an unrelated tensor reuse its value.
    let (mut orphan, gradient) = {
        let p = Projection::new(&ops, &mut stream, &factors[0]).unwrap();
        (optimizer::PreparedOptimizer::new(&stream, &[&p.a]).unwrap(), p.a.grad.clone())
    };
    let unrelated = upload(&ops, &mut stream, 3, 19, &vec![7.0; 57]);
    gradient.clear(&stream).unwrap();
    orphan.update(&mut stream, &TrainConfig::default(), 1).unwrap();
    assert_eq!(read(&unrelated, &mut stream), vec![7.0; 57]);
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn modulated_norm_backward_preserves_rounding_and_matches_cpu_chain_rule() {
    use krea2::ops::Binary;
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in [(1, 1), (3, 127), (5, 128), (3, 129), (3, 6144)] {
        let x = values(rows * cols, 4);
        let g = values(rows * cols, 9);
        let residual = values(rows * cols, 13);
        let mut modulation = values(cols, 19);
        for (dst, value) in
            modulation.iter_mut().zip([-1.0, 0.0, 1.0 / 256.0, -1.0 / 256.0, 64.0, -64.0])
        {
            *dst = value;
        }
        let w: Vec<_> = (0..cols).map(|i| ((i * 23 % 37) as f32 - 18.0) * 0.013).collect();
        let xt = upload(&ops, &mut stream, rows, cols, &x);
        let gt = upload(&ops, &mut stream, rows, cols, &g);
        let rt = upload(&ops, &mut stream, rows, cols, &residual);
        let mt = upload(&ops, &mut stream, 1, cols, &modulation);
        let wt = FloatTensor::from_slice(&mut stream, 1, cols, &w).unwrap();
        for eps in [1e-5, 1e-3] {
            let result = ops::norm_modulated_backward(
                &ops,
                &stream,
                &xt,
                &gt,
                wt.binding(),
                &mt,
                &rt,
                eps,
            )
            .unwrap();
            let one_plus = ops::one_plus(&ops, &stream, &mt).unwrap();
            let scaled = ops.binary(&stream, &gt, &one_plus, Binary::Mul).unwrap();
            let norm =
                ops::norm_backward(&ops, &stream, &xt, &scaled, wt.binding(), eps).unwrap();
            let reference = ops::add_scaled(&ops, &stream, &rt, &norm, 1.0).unwrap();
            assert_eq!(
                result.download(&mut stream).unwrap(),
                reference.download(&mut stream).unwrap()
            );
            let mut expected = vec![0.0; rows * cols];
            for row in 0..rows {
                let xx = &x[row * cols..(row + 1) * cols];
                let grad: Vec<_> = (0..cols)
                    .map(|c| {
                        let scaled = to_f32(from_f32(
                            g[row * cols + c] * to_f32(from_f32(1.0 + modulation[c])),
                        ));
                        f64::from(scaled * (1.0 + w[c]))
                    })
                    .collect();
                let variance = xx.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
                    / cols as f64
                    + f64::from(eps);
                let dot = xx.iter().zip(&grad).map(|(x, g)| f64::from(*x) * g).sum::<f64>()
                    / cols as f64;
                for c in 0..cols {
                    let norm = ((grad[c] - f64::from(xx[c]) * dot / variance) / variance.sqrt())
                        as f32;
                    expected[row * cols + c] =
                        to_f32(from_f32(residual[row * cols + c] + to_f32(from_f32(norm))));
                }
            }
            close(&read(&result, &mut stream), &expected, 0.006, 1e-5);
        }
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                ops::norm_modulated_backward(
                    &ops,
                    &stream,
                    &xt,
                    &gt,
                    wt.binding(),
                    &mt,
                    &rt,
                    eps
                )
                .is_err()
            );
        }
        let bad_mod = upload(&ops, &mut stream, 2, cols, &vec![0.0; 2 * cols]);
        assert!(
            ops::norm_modulated_backward(
                &ops,
                &stream,
                &xt,
                &gt,
                wt.binding(),
                &bad_mod,
                &rt,
                1e-5
            )
            .is_err()
        );
        let bad = upload(&ops, &mut stream, 1, cols + 1, &vec![0.0; cols + 1]);
        for (g, r) in [(&bad, &rt), (&gt, &bad)] {
            assert!(
                ops::norm_modulated_backward(&ops, &stream, &xt, g, wt.binding(), &mt, r, 1e-5)
                    .is_err()
            );
        }
    }
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn rmsnorm_and_rotary_backward_match_cpu_derivatives() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in
        [(1, 128), (3, 128), (4, 128), (5, 128), (31, 128), (3, 127), (3, 129), (3, 6144)]
    {
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
#[ignore = "requires a gfx1151 GPU"]
fn fused_rotary_norm_backward_preserves_rounding_and_matches_cpu() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (tokens, heads) in [(1, 1), (3, 2), (5, 12), (3, 48)] {
        let cols = heads * 128;
        let x = values(tokens * cols, 4);
        let grad = values(tokens * cols, 9);
        let weights: Vec<_> = (0..128).map(|i| (i as f32 - 64.0) * 0.0031).collect();
        // Token- and pair-dependent angles catch table indexing across heads and quarters.
        let angles: Vec<_> = (0..tokens * 128).map(|i| (i / 2) as f32 * 0.137).collect();
        let cos: Vec<_> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<_> = angles.iter().map(|a| a.sin()).collect();
        let xt = upload(&ops, &mut stream, tokens, cols, &x);
        let gt = upload(&ops, &mut stream, tokens, cols, &grad);
        let wt = FloatTensor::from_slice(&mut stream, 1, 128, &weights).unwrap();
        let ct = FloatTensor::from_slice(&mut stream, tokens, 128, &cos).unwrap();
        let st = FloatTensor::from_slice(&mut stream, tokens, 128, &sin).unwrap();
        for eps in [1e-5, 1e-3] {
            let result =
                ops::rope_norm_backward(&ops, &stream, &xt, &gt, wt.binding(), &ct, &st, eps)
                    .unwrap();
            let rotary = ops::rope(&ops, &stream, &gt, &ct, &st, true).unwrap();
            let reference = ops::norm_backward(
                &ops,
                &stream,
                &xt.view(tokens * heads, 128, 0).unwrap(),
                &rotary.view(tokens * heads, 128, 0).unwrap(),
                wt.binding(),
                eps,
            )
            .unwrap();
            assert_eq!(
                result.download(&mut stream).unwrap(),
                reference.download(&mut stream).unwrap()
            );
            let mut expected = vec![0.0; tokens * cols];
            for row in 0..tokens * heads {
                let table = row / heads * 128;
                let base = row * 128;
                let xx = &x[base..base + 128];
                let g: Vec<_> = (0..128)
                    .map(|c| {
                        let rotated = grad[base + (c ^ 1)]
                            * sin[table + c]
                            * if c % 2 == 0 { 1.0 } else { -1.0 };
                        let rounded =
                            to_f32(from_f32(grad[base + c] * cos[table + c] + rotated));
                        f64::from(rounded * (1.0 + weights[c]))
                    })
                    .collect();
                let variance = xx.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 128.0
                    + f64::from(eps);
                let dot =
                    xx.iter().zip(&g).map(|(x, g)| f64::from(*x) * g).sum::<f64>() / 128.0;
                for c in 0..128 {
                    expected[base + c] =
                        ((g[c] - f64::from(xx[c]) * dot / variance) / variance.sqrt()) as f32;
                }
            }
            close(&read(&result, &mut stream), &expected, 0.006, 1e-5);
        }
        for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                ops::rope_norm_backward(&ops, &stream, &xt, &gt, wt.binding(), &ct, &st, eps)
                    .is_err()
            );
        }
        let bad = upload(&ops, &mut stream, tokens, cols + 1, &vec![0.0; tokens * (cols + 1)]);
        for (x, g) in [(&xt, &bad), (&bad, &bad)] {
            assert!(
                ops::rope_norm_backward(&ops, &stream, x, g, wt.binding(), &ct, &st, 1e-5)
                    .is_err()
            );
        }
        let bad_table =
            FloatTensor::from_slice(&mut stream, tokens, 64, &vec![0.0; tokens * 64]).unwrap();
        for (c, s) in [(&bad_table, &st), (&ct, &bad_table)] {
            assert!(
                ops::rope_norm_backward(&ops, &stream, &xt, &gt, wt.binding(), c, s, 1e-5)
                    .is_err()
            );
        }
        assert!(
            ops::rope_norm_backward(
                &ops,
                &stream,
                &xt,
                &gt,
                bad_table.binding(),
                &ct,
                &st,
                1e-5
            )
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires an idle gfx1151 GPU"]
fn streaming_gqa_forward_and_backward_match_materialized_f64_attention() {
    for (tokens, heads, kv) in [
        (1, 1, 1),
        (3, 3, 1),
        (9, 8, 2),
        (16, 4, 1),
        (17, 4, 1),
        (31, 4, 1),
        (32, 4, 1),
        (33, 4, 1),
        (128, 4, 1),
        (129, 8, 2),
        (257, 4, 1),
        (17, 48, 12),
        (33, 48, 12),
    ] {
        check_attention(tokens, heads, kv, 1.0);
    }
    for gain in [8.0, 32.0] {
        check_attention(129, 8, 2, gain);
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn tiled_attention_matches_scalar_kernels_at_model_dimensions() {
    use krea2::kernels::{Scalars, config};
    let (t, heads, kv, d) = (1070usize, 48usize, 12usize, 128usize);
    for gain in [1.0f32, 8.0, 32.0] {
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(BufferPool::new());
        let scaled =
            |n, offset| values(n, offset).into_iter().map(|v| v * gain).collect::<Vec<_>>();
        let q = upload(&ops, &mut stream, t, heads * d, &scaled(t * heads * d, 3));
        let k = upload(&ops, &mut stream, t, kv * d, &scaled(t * kv * d, 5));
        let v = upload(&ops, &mut stream, t, kv * d, &values(t * kv * d, 7));
        let grad = upload(&ops, &mut stream, t, heads * d, &values(t * heads * d, 13));
        let f = ops::attention(&ops, &stream, &q, &k, &v).unwrap();
        let (dq, dk, dv) =
            ops::attention_backward(&ops, &stream, &q, &k, &v, &f, &grad).unwrap();
        let out = ops.tensor(&stream, t, heads * d).unwrap();
        let exact =
            FloatTensor::from_slice(&mut stream, t, heads * d, &vec![0.0; t * heads * d])
                .unwrap();
        let lse =
            FloatTensor::from_slice(&mut stream, t, heads, &vec![0.0; t * heads]).unwrap();
        let delta =
            FloatTensor::from_slice(&mut stream, t, heads, &vec![0.0; t * heads]).unwrap();
        let rq = ops.tensor(&stream, t, heads * d).unwrap();
        let rk = ops.tensor(&stream, t, kv * d).unwrap();
        let rv = ops.tensor(&stream, t, kv * d).unwrap();
        let conf = config([
            ("tokens", t as u64),
            ("sequence", t as u64),
            ("heads", heads as u64),
            ("kv", kv as u64),
            ("qsize", (t * heads * d) as u64),
            ("ksize", (t * kv * d) as u64),
            ("stats", (t * heads) as u64),
        ]);
        let scalars = Scalars::new().index(t);
        // SAFETY: the scalar reference kernels receive their contiguous logical
        // shapes; each wave owns one token/head and every output is disjoint.
        unsafe {
            ops.launch(
                &stream,
                "train_attention",
                conf.clone(),
                &scalars,
                &[
                    q.binding().unwrap(),
                    k.binding().unwrap(),
                    v.binding().unwrap(),
                    out.binding().unwrap(),
                    exact.binding(),
                    lse.binding(),
                ],
                t * heads,
                1,
                32,
            )
            .unwrap();
            ops.launch(
                &stream,
                "train_attention_delta",
                conf.clone(),
                &scalars,
                &[grad.binding().unwrap(), exact.binding(), delta.binding()],
                t * heads,
                1,
                32,
            )
            .unwrap();
            for (name, a, b, count) in [
                ("train_attention_dq", &rq, &rq, t * heads),
                ("train_attention_dkv", &rk, &rv, t * kv),
            ] {
                ops.launch(
                    &stream,
                    name,
                    conf.clone(),
                    &scalars,
                    &[
                        q.binding().unwrap(),
                        k.binding().unwrap(),
                        v.binding().unwrap(),
                        grad.binding().unwrap(),
                        lse.binding(),
                        delta.binding(),
                        a.binding().unwrap(),
                        b.binding().unwrap(),
                    ],
                    count,
                    1,
                    32,
                )
                .unwrap();
            }
        }
        for (name, candidate, reference) in
            [("output", &f.output, &out), ("dq", &dq, &rq), ("dk", &dk, &rk), ("dv", &dv, &rv)]
        {
            let candidate = read(candidate, &mut stream);
            let reference = read(reference, &mut stream);
            let error = candidate
                .iter()
                .zip(&reference)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>();
            let norm = reference.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            eprintln!(
                "attention gain {gain} {name}: relative L2 error {}",
                (error / norm).sqrt()
            );
            // Relative-only: saturated attention has tiny but nonzero dQ/dK.
            // An absolute floor hid false gradients from approximate division
            // of saved FP32 output and mismatched delta/dP reductions.
            close(&candidate, &reference, 0.005, 0.0);
        }
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn tiled_attention_real_shapes_match_uniform_attention() {
    // K=0 makes attention uniform. Constant Q and upstream gradients across
    // tokens give a closed-form FP64 oracle, including nonzero dK and dV, at
    // full model dimensions without constructing a quadratic CPU reference.
    let (heads, kv, d) = (48, 12, 128);
    for t in [1024, 1043, 1536, 1537, 4096, 4115] {
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(BufferPool::new());
        let qrow = values(heads * d, 3);
        let grow = values(heads * d, 13);
        let v = values(t * kv * d, 7);
        let qt = upload(&ops, &mut stream, t, heads * d, &qrow.repeat(t));
        let kt = upload(&ops, &mut stream, t, kv * d, &vec![0.0; t * kv * d]);
        let vt = upload(&ops, &mut stream, t, kv * d, &v);
        let gt = upload(&ops, &mut stream, t, heads * d, &grow.repeat(t));
        let f = ops::attention(&ops, &stream, &qt, &kt, &vt).unwrap();
        let (dq, dk, dv) =
            ops::attention_backward(&ops, &stream, &qt, &kt, &vt, &f, &gt).unwrap();
        let mut mean = vec![0.0f64; kv * d];
        for row in v.chunks_exact(kv * d) {
            for (sum, value) in mean.iter_mut().zip(row) {
                *sum += f64::from(*value) / t as f64;
            }
        }
        let mut expected_out = vec![0.0; heads * d];
        let mut expected_v = vec![0.0; kv * d];
        for h in 0..heads {
            for c in 0..d {
                expected_out[h * d + c] = mean[(h / 4) * d + c] as f32;
                expected_v[(h / 4) * d + c] += grow[h * d + c];
            }
        }
        let mut expected_k = vec![0.0f32; t * kv * d];
        for row in 0..t {
            for h in 0..heads {
                let base = (row * kv + h / 4) * d;
                let ds = (0..d)
                    .map(|c| {
                        f64::from(grow[h * d + c])
                            * (f64::from(v[base + c]) - mean[(h / 4) * d + c])
                    })
                    .sum::<f64>()
                    / (d as f64).sqrt();
                for c in 0..d {
                    expected_k[base + c] += (ds * f64::from(qrow[h * d + c])) as f32;
                }
            }
        }
        close(&read(&f.output, &mut stream), &expected_out.repeat(t), 0.005, 1e-5);
        close(&read(&dq, &mut stream), &vec![0.0; t * heads * d], 0.0, 1e-5);
        close(&read(&dk, &mut stream), &expected_k, 0.005, 1e-5);
        close(&read(&dv, &mut stream), &expected_v.repeat(t), 0.005, 1e-5);
    }
}

fn check_attention(t: usize, heads: usize, kv: usize, gain: f32) {
    check_batched_attention(t, heads, kv, gain, t);
}

fn check_batched_attention(t: usize, heads: usize, kv: usize, gain: f32, sequence: usize) {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let d = 128;
    let q: Vec<f32> = values(t * heads * d, 3).into_iter().map(|v| v * gain).collect();
    let k: Vec<f32> = values(t * kv * d, 5).into_iter().map(|v| v * gain).collect();
    let v = values(t * kv * d, 7);
    let g = values(q.len(), 13);
    let qt = upload(&ops, &mut stream, t, heads * d, &q);
    let kt = upload(&ops, &mut stream, t, kv * d, &k);
    let vt = upload(&ops, &mut stream, t, kv * d, &v);
    let gt = upload(&ops, &mut stream, t, heads * d, &g);
    let mut f = ops::attention_batched(&ops, &stream, &qt, &kt, &vt, sequence).unwrap();
    let (dq, dk, dv) = ops::attention_backward(&ops, &stream, &qt, &kt, &vt, &f, &gt).unwrap();
    if sequence == t && t >= 16 && heads == kv * 4 {
        let unrelated_q = upload(&ops, &mut stream, t, heads * d, &q);
        assert!(
            ops::attention_backward(&ops, &stream, &unrelated_q, &kt, &vt, &f, &gt).is_err()
        );
    }
    let mut output = vec![0.0; q.len()];
    let mut eq = vec![0.0f64; q.len()];
    let mut ek = vec![0.0f64; k.len()];
    let mut ev = vec![0.0f64; v.len()];
    let scale = 1.0 / (d as f64).sqrt();
    for i in 0..t {
        for h in 0..heads {
            let hi = h / (heads / kv);
            let qi = (i * heads + h) * d;
            let begin = (i / sequence) * sequence;
            let mut scores = (begin..begin + sequence)
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
                o[c] = (begin..begin + sequence)
                    .map(|j| scores[j - begin] * f64::from(v[(j * kv + hi) * d + c]))
                    .sum();
                output[qi + c] = o[c] as f32;
            }
            let delta = (0..d).map(|c| o[c] * f64::from(g[qi + c])).sum::<f64>();
            for (j, &probability) in scores.iter().enumerate() {
                let kj = ((begin + j) * kv + hi) * d;
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

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn full_targets_batched_fusion_attention_matches_f64() {
    for (batch, tokens, heads, kv) in [(3, 12, 20, 20), (2, 7, 4, 1), (1, 43, 20, 20)] {
        for gain in [1.0, 8.0] {
            check_batched_attention(batch * tokens, heads, kv, gain, tokens);
        }
    }
}

#[test]
#[ignore = "requires a gfx1151 GPU"]
fn full_targets_gelu_reduction_and_tap_layout_match_cpu() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    let (rows, cols) = (43, 257);
    let x: Vec<_> = values(rows * cols, 3).into_iter().map(|v| v * 12.0).collect();
    let g = values(rows * cols, 7);
    let xt = upload(&ops, &mut stream, rows, cols, &x);
    let gt = upload(&ops, &mut stream, rows, cols, &g);
    let dx = ops::gelu_backward(&ops, &stream, &xt, &gt).unwrap();
    // Finite differences of independent f64 GELU, including both saturated tails.
    let gelu = |x: f64| {
        0.5 * x
            * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    };
    let expected: Vec<_> = x
        .iter()
        .zip(&g)
        .map(|(&x, &g)| {
            let x = f64::from(x);
            (f64::from(g) * (gelu(x + 1e-5) - gelu(x - 1e-5)) / 2e-5) as f32
        })
        .collect();
    close(&read(&dx, &mut stream), &expected, 0.003, 1e-6);
    let dst = FloatTensor::zero(&stream, 6, cols).unwrap();
    for _ in 0..28 {
        ops::sum_rows_accumulate(&ops, &stream, &gt, &dst, 3).unwrap();
    }
    let sums: Vec<_> = (0..cols)
        .map(|c| (0..rows).map(|r| f64::from(g[r * cols + c])).sum::<f64>() as f32 * 28.0)
        .collect();
    let result = dst.download(&mut stream).unwrap();
    close(&result[3 * cols..4 * cols], &sums, 1e-6, 1e-6);
    assert!(result[..3 * cols].iter().chain(&result[4 * cols..]).all(|v| *v == 0.0));
    let taps = upload(&ops, &mut stream, 3 * 12, 2560, &values(3 * 12 * 2560, 11));
    let permuted = ops::permute_taps(&ops, &stream, &taps, false).unwrap();
    let original = taps.download(&mut stream).unwrap();
    let perm = permuted.download(&mut stream).unwrap();
    for token in 0..3 {
        for c in 0..2560 {
            for layer in 0..12 {
                assert_eq!(
                    perm[(token * 2560 + c) * 12 + layer],
                    original[(token * 12 + layer) * 2560 + c]
                );
            }
        }
    }
    let restored = ops::permute_taps(&ops, &stream, &permuted, true).unwrap();
    assert_eq!(original, restored.download(&mut stream).unwrap());
}
