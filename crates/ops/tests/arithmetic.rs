//! Device operations checked against CPU arithmetic and alternate layouts.
//! Scheduler rounding is checked exactly; convolution reduction orders use a
//! tolerance. Run explicitly with --ignored on gfx1151.
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::{Binary, Layout, Ops, Pool, Tensor, Unary, Weight};

fn upload(ops: &Ops, values: &[f32], rows: usize, cols: usize) -> Tensor {
    let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
    Tensor::from_slice(ops.pool(), &bits, rows, cols).expect("upload")
}

fn download(tensor: &Tensor) -> Vec<f32> {
    tensor.download().expect("download").into_iter().map(to_f32).collect()
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_euler_step_rounds_where_the_sampler_rounds() {
    let ops = Ops::new(Pool::new());
    let samples: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 97.0).collect();
    let velocity: Vec<f32> = (0..1000).map(|i| (i as f32).sin() * 3.0).collect();
    let delta = -0.1234f32;
    let sample = upload(&ops, &samples, 1, 1000);
    let v = upload(&ops, &velocity, 1, 1000);
    ops.euler_step(&sample, &v, delta).expect("euler");

    // kernels/native/euler.loom: bf16 delta, bf16 product, bf16 sum -- the
    // rounding diffusers' CUDA pipeline performs.
    let want: Vec<f32> = samples
        .iter()
        .zip(&velocity)
        .map(|(&x, &v)| {
            let dt = to_f32(from_f32(delta));
            let product = to_f32(from_f32(dt * to_f32(from_f32(v))));
            to_f32(from_f32(to_f32(from_f32(x)) + product))
        })
        .collect();
    assert_eq!(download(&sample), want);
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn guidance_combines_as_krea_defines_it() {
    let ops = Ops::new(Pool::new());
    let cond: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).cos()).collect();
    let uncond: Vec<f32> = (0..512).map(|i| (i as f32 * 0.023).sin()).collect();
    let scale = 3.5f32;
    let c = upload(&ops, &cond, 1, 512);
    let u = upload(&ops, &uncond, 1, 512);
    ops.guidance(&c, &u, scale).expect("guidance");

    let want: Vec<f32> = cond
        .iter()
        .zip(&uncond)
        .map(|(&c, &u)| {
            let (c, u) = (to_f32(from_f32(c)), to_f32(from_f32(u)));
            let difference = to_f32(from_f32(c - u));
            to_f32(from_f32(c + to_f32(from_f32(scale * difference))))
        })
        .collect();
    assert_eq!(download(&c), want);
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_pointwise_and_broadcast_operations_agree_with_the_host() {
    let ops = Ops::new(Pool::new());
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 16.0).collect();
    let x = upload(&ops, &values, 16, 16);

    let silu = ops.unary(&x, Unary::Silu).expect("silu");
    for (got, &value) in download(&silu).iter().zip(&values) {
        let value = to_f32(from_f32(value));
        let want = value / (1.0 + (-value).exp());
        assert!(
            (got - want).abs() <= 0.01 * want.abs().max(1.0),
            "silu({value}): {got} vs {want}"
        );
    }

    // One row broadcast over sixteen.
    let row: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
    let y = upload(&ops, &row, 1, 16);
    let product = ops.binary(&x, &y, Binary::Mul).expect("mul");
    let want: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| to_f32(from_f32(to_f32(from_f32(v)) * to_f32(from_f32(row[i % 16])))))
        .collect();
    assert_eq!(download(&product), want);
}

/// Convolution dispatch must respect the declared weight layout.
/// The two reduction orders must agree within the numerical tolerance.
#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn a_convolution_reads_its_weight_in_the_order_the_weight_is_in() {
    let ops = Ops::new(Pool::new());
    let (height, width, inputs, outputs) = (8usize, 8usize, 8usize, 16usize);
    let image: Vec<f32> =
        (0..height * width * inputs).map(|i| ((i % 23) as f32 - 11.0) / 16.0).collect();
    let x = upload(&ops, &image, height * width, inputs);

    // [out][in][ky][kx], and the same values as [out][ky][kx][in].
    let row_major: Vec<f32> =
        (0..outputs * inputs * 9).map(|i| ((i % 17) as f32 - 8.0) / 64.0).collect();
    let mut packed = vec![0.0f32; row_major.len()];
    for o in 0..outputs {
        for i in 0..inputs {
            for t in 0..9 {
                packed[(o * 9 + t) * inputs + i] = row_major[(o * inputs + i) * 9 + t];
            }
        }
    }

    let weight = |values: &[f32], layout: Layout| {
        let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
        let buffer = std::sync::Arc::new(
            hrx::device().allocate(bits.len() * 2).expect("a weight allocation"),
        );
        hrx::device().write(buffer.ptr(), &bits).expect("upload");
        Weight::new(&buffer, buffer.ptr(), vec![outputs, inputs, 3, 3], bits.len(), None)
            .in_layout(layout)
    };

    let patches = ops
        .conv(&x, height, width, &weight(&row_major, Layout::RowMajor), None)
        .expect("the patch path");
    let implicit = ops
        .conv(&x, height, width, &weight(&packed, Layout::ChannelsLast), None)
        .expect("the implicit path");

    let (a, b) = (download(&patches), download(&implicit));
    assert_eq!(a.len(), height * width * outputs);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= 0.02 * scale.max(1.0),
        "the two paths disagree by {worst} on values up to {scale}"
    );
    // An explicitly packed 3×3 weight is accepted.
    let flat = weight(&row_major, Layout::ChannelsLast);
    assert!(ops.conv(&x, height, width, &flat, None).is_ok(), "3x3 packed is fine");
}

fn weight(values: &[f32], shape: Vec<usize>) -> Weight {
    let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
    let buffer = std::sync::Arc::new(hrx::device().allocate(bits.len() * 2).unwrap());
    hrx::device().write(buffer.ptr(), &bits).unwrap();
    Weight::new(&buffer, buffer.ptr(), shape, bits.len(), None)
}
fn rounded(v: f32) -> f64 {
    to_f32(from_f32(v)) as f64
}
fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a as f64 - b).abs() <= tolerance * (1. + b.abs()),
            "element {i}: {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn linear_layers_match_scalar_matmul_with_bias_and_ragged_tiles() {
    let ops = Ops::new(Pool::new());
    for (m, k, n) in [(1usize, 3usize, 7usize), (17, 33, 65), (65, 129, 257)] {
        let x: Vec<_> = (0..m * k).map(|i| ((i * 7 % 31) as f32 - 15.) / 32.).collect();
        let w: Vec<_> = (0..n * k).map(|i| ((i * 11 % 37) as f32 - 18.) / 64.).collect();
        let bias: Vec<_> = (0..n).map(|i| ((i % 7) as f32 - 3.) / 32.).collect();
        let xdev = upload(&ops, &x, m, k);
        let wdev = weight(&w, vec![n, k]);
        let bdev = hrx::device().allocate(n * 4).unwrap();
        hrx::device()
            .write(bdev.ptr(), &bias.iter().copied().map(from_f32).collect::<Vec<_>>())
            .unwrap();
        for biased in [false, true] {
            let want: Vec<_> = (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k).map(|j| rounded(x[r * k + j]) * rounded(w[c * k + j])).sum::<f64>()
                        + if biased { bias[c] as f64 } else { 0. }
                })
                .collect();
            let y = ops.linear(&xdev, &wdev, biased.then(|| bdev.ptr())).unwrap();
            close(&download(&y), &want, 0.01);
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn normalization_modes_and_fused_silu_match_cpu() {
    use krea2_ops::Norm;
    let ops = Ops::new(Pool::new());
    for cols in [32usize, 256, 1024, 1152] {
        let rows = 9;
        let x: Vec<_> = (0..rows * cols).map(|i| ((i * 3 % 29) as f32 - 14.) / 8.).collect();
        let w: Vec<_> = (0..cols).map(|i| 1. + (i % 7) as f32 / 16.).collect();
        let xd = upload(&ops, &x, rows, cols);
        let wd = weight(&w, vec![cols]);
        for mode in [Norm::OnePlusScale, Norm::Scale, Norm::Group] {
            let mut want = Vec::new();
            for row in x.chunks_exact(cols) {
                let mean = 0.;
                let variance =
                    row.iter().map(|&x| (rounded(x) - mean).powi(2)).sum::<f64>() / cols as f64;
                for (i, &x) in row.iter().enumerate() {
                    let y = if mode == Norm::Group {
                        let l2 = (variance * cols as f64).sqrt().max(1e-12);
                        rounded(
                            (rounded((rounded(x) / l2) as f32) * (cols as f64).sqrt()) as f32,
                        ) * rounded(w[i])
                    } else {
                        rounded(x) / (variance + 1e-5).sqrt()
                            * (rounded(w[i]) + if mode == Norm::OnePlusScale { 1. } else { 0. })
                    };
                    want.push(y);
                }
            }
            close(&download(&ops.norm(&xd, &wd, mode, 1e-5).unwrap()), &want, 0.015);
            if mode == Norm::Group {
                let want: Vec<_> = want.into_iter().map(|x| x / (1. + (-x).exp())).collect();
                close(&download(&ops.norm_silu(&xd, &wd).unwrap()), &want, 0.02);
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn causal_and_grouped_attention_match_scalar_softmax() {
    let ops = Ops::new(Pool::new());
    for tokens in [3usize, 17, 33] {
        for causal in [false, true] {
            let (heads, kv, dim) = (4usize, 2usize, 16usize);
            let q: Vec<_> =
                (0..tokens * heads * dim).map(|i| ((i * 7 % 31) as f32 - 15.) / 16.).collect();
            let k: Vec<_> =
                (0..tokens * kv * dim).map(|i| ((i * 3 % 23) as f32 - 11.) / 16.).collect();
            let v: Vec<_> = k.iter().rev().copied().collect();
            let mut want = vec![0f64; q.len()];
            for t in 0..tokens {
                for head in 0..heads {
                    let kh = head / (heads / kv);
                    let valid = if causal { t + 1 } else { tokens };
                    let scores: Vec<_> = (0..valid)
                        .map(|j| {
                            (0..dim)
                                .map(|c| {
                                    rounded(q[(t * heads + head) * dim + c])
                                        * rounded(k[(j * kv + kh) * dim + c])
                                })
                                .sum::<f64>()
                                / (dim as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let sum = scores.iter().map(|x| (x - max).exp()).sum::<f64>();
                    for (j, score) in scores.iter().enumerate() {
                        let p = rounded(((score - max).exp() / sum) as f32);
                        for c in 0..dim {
                            want[(t * heads + head) * dim + c] +=
                                p * rounded(v[(j * kv + kh) * dim + c]);
                        }
                    }
                }
            }
            let q = upload(&ops, &q, tokens, heads * dim);
            let k = upload(&ops, &k, tokens, kv * dim);
            let v = upload(&ops, &v, tokens, kv * dim);
            close(
                &download(
                    &ops.attention(&q, &k, &v, 1, tokens, heads, kv, dim, causal).unwrap(),
                ),
                &want,
                0.015,
            );
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn rotary_embedding_and_nearest_upsampling_match_cpu_indices() {
    let ops = Ops::new(Pool::new());
    let (tokens, heads, dim) = (7usize, 3usize, 8usize);
    let x: Vec<_> = (0..tokens * heads * dim).map(|i| ((i % 17) as f32 - 8.) / 8.).collect();
    let mut want = vec![0f64; x.len()];
    for t in 0..tokens {
        for h in 0..heads {
            for c in 0..dim {
                let i = (t * heads + h) * dim + c;
                let half = dim / 2;
                let angle = t as f64 / 10000f64.powf((2 * (c % half)) as f64 / dim as f64);
                want[i] = rounded(x[i]) * angle.cos()
                    + if c < half {
                        -rounded(x[i + half]) * angle.sin()
                    } else {
                        rounded(x[i - half]) * angle.sin()
                    };
            }
        }
    }
    let xd = upload(&ops, &x, tokens, heads * dim);
    close(&download(&ops.rope(&xd, tokens, heads, 10000.).unwrap()), &want, 0.015);
    let (height, width, channels) = (3usize, 5usize, 7usize);
    let image: Vec<_> = (0..height * width * channels).map(|i| i as f32 / 32.).collect();
    let input = upload(&ops, &image, height * width, channels);
    let output = download(&ops.upsample(&input, height, width).unwrap());
    for y in 0..height * 2 {
        for x in 0..width * 2 {
            for c in 0..channels {
                assert_eq!(
                    output[(y * width * 2 + x) * channels + c] as f64,
                    rounded(image[((y / 2) * width + x / 2) * channels + c])
                );
            }
        }
    }
}
