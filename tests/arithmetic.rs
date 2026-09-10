//! Device operations checked against CPU arithmetic and alternate layouts.
//! Scheduler rounding is checked exactly; convolution reduction orders use a
//! tolerance. Run explicitly with --ignored on gfx1151.
use hrx::Stream;
use krea2::numerics::{from_f32, to_f32};
use krea2::ops::{Binary, Layout, Ops, Pool, Tensor, Unary, Weight};

/// A stream and the operations over it. Every test needs both, and the stream
/// has to outlive the tensors allocated on it.
fn opened() -> (Stream, Ops) {
    let stream = Stream::open().expect("a stream");
    let ops = Ops::new(Pool::new());
    (stream, ops)
}

fn upload(stream: &mut Stream, ops: &Ops, values: &[f32], rows: usize, cols: usize) -> Tensor {
    let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
    Tensor::from_slice(ops.pool(), stream, &bits, rows, cols).expect("upload")
}

fn download(stream: &mut Stream, tensor: &Tensor) -> Vec<f32> {
    tensor.download(stream).expect("download").into_iter().map(to_f32).collect()
}

/// A bf16 weight of its own allocation, which the tensor keeps a share of.
fn weight_on(stream: &mut Stream, values: &[f32], shape: Vec<usize>) -> Weight {
    let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
    let buffer = std::sync::Arc::new(stream.allocate(bits.len() * 2).expect("an allocation"));
    stream.upload(buffer.binding(), bytemuck::cast_slice(&bits)).expect("upload");
    Weight::new(&buffer, 0, shape, bits.len(), None)
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_euler_step_rounds_where_the_sampler_rounds() {
    let (mut stream, ops) = opened();
    let samples: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 97.0).collect();
    let velocity: Vec<f32> = (0..1000).map(|i| (i as f32).sin() * 3.0).collect();
    let delta = -0.1234f32;
    let sample = upload(&mut stream, &ops, &samples, 1, 1000);
    let v = upload(&mut stream, &ops, &velocity, 1, 1000);
    ops.euler_step(&stream, &sample, &v, delta).expect("euler");

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
    assert_eq!(download(&mut stream, &sample), want);
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn guidance_combines_as_krea_defines_it() {
    let (mut stream, ops) = opened();
    let cond: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).cos()).collect();
    let uncond: Vec<f32> = (0..512).map(|i| (i as f32 * 0.023).sin()).collect();
    let scale = 3.5f32;
    let c = upload(&mut stream, &ops, &cond, 1, 512);
    let u = upload(&mut stream, &ops, &uncond, 1, 512);
    ops.guidance(&stream, &c, &u, scale).expect("guidance");

    let want: Vec<f32> = cond
        .iter()
        .zip(&uncond)
        .map(|(&c, &u)| {
            let (c, u) = (to_f32(from_f32(c)), to_f32(from_f32(u)));
            let difference = to_f32(from_f32(c - u));
            to_f32(from_f32(c + to_f32(from_f32(scale * difference))))
        })
        .collect();
    assert_eq!(download(&mut stream, &c), want);
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_pointwise_and_broadcast_operations_agree_with_the_host() {
    let (mut stream, ops) = opened();
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 16.0).collect();
    let x = upload(&mut stream, &ops, &values, 16, 16);

    let silu = ops.unary(&stream, &x, Unary::Silu).expect("silu");
    for (got, &value) in download(&mut stream, &silu).iter().zip(&values) {
        let value = to_f32(from_f32(value));
        let want = value / (1.0 + (-value).exp());
        assert!(
            (got - want).abs() <= 0.01 * want.abs().max(1.0),
            "silu({value}): {got} vs {want}"
        );
    }

    // One row broadcast over sixteen.
    let row: Vec<f32> = (0..16).map(|i| 1.0 + i as f32).collect();
    let y = upload(&mut stream, &ops, &row, 1, 16);
    let product = ops.binary(&stream, &x, &y, Binary::Mul).expect("mul");
    let want: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| to_f32(from_f32(to_f32(from_f32(v)) * to_f32(from_f32(row[i % 16])))))
        .collect();
    assert_eq!(download(&mut stream, &product), want);
}

/// Convolution dispatch must respect the declared weight layout.
/// The two reduction orders must agree within the numerical tolerance.
#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn a_convolution_reads_its_weight_in_the_order_the_weight_is_in() {
    let (mut stream, ops) = opened();
    let (height, width, inputs, outputs) = (8usize, 8usize, 8usize, 16usize);
    let image: Vec<f32> =
        (0..height * width * inputs).map(|i| ((i % 23) as f32 - 11.0) / 16.0).collect();
    let x = upload(&mut stream, &ops, &image, height * width, inputs);

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

    let shape = vec![outputs, inputs, 3, 3];
    let row_weight =
        weight_on(&mut stream, &row_major, shape.clone()).in_layout(Layout::RowMajor);
    let packed_weight =
        weight_on(&mut stream, &packed, shape.clone()).in_layout(Layout::ChannelsLast);

    let patches =
        ops.conv(&stream, &x, height, width, &row_weight, None).expect("the patch path");
    let implicit =
        ops.conv(&stream, &x, height, width, &packed_weight, None).expect("the implicit path");

    let (a, b) = (download(&mut stream, &patches), download(&mut stream, &implicit));
    assert_eq!(a.len(), height * width * outputs);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= 0.02 * scale.max(1.0),
        "the two paths disagree by {worst} on values up to {scale}"
    );
    // An explicitly packed 3×3 weight is accepted.
    let flat = weight_on(&mut stream, &row_major, shape).in_layout(Layout::ChannelsLast);
    assert!(ops.conv(&stream, &x, height, width, &flat, None).is_ok(), "3x3 packed is fine");
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
    let (mut stream, ops) = opened();
    for (m, k, n) in [(1usize, 3usize, 7usize), (17, 33, 65), (65, 129, 257)] {
        let x: Vec<_> = (0..m * k).map(|i| ((i * 7 % 31) as f32 - 15.) / 32.).collect();
        let w: Vec<_> = (0..n * k).map(|i| ((i * 11 % 37) as f32 - 18.) / 64.).collect();
        let bias: Vec<_> = (0..n).map(|i| ((i % 7) as f32 - 3.) / 32.).collect();
        let xdev = upload(&mut stream, &ops, &x, m, k);
        let wdev = weight_on(&mut stream, &w, vec![n, k]);
        let bdev = stream.allocate(n * 2).unwrap();
        let bits: Vec<u16> = bias.iter().copied().map(from_f32).collect();
        stream.upload(bdev.binding(), bytemuck::cast_slice(&bits)).unwrap();
        for biased in [false, true] {
            let want: Vec<_> = (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k).map(|j| rounded(x[r * k + j]) * rounded(w[c * k + j])).sum::<f64>()
                        + if biased { bias[c] as f64 } else { 0. }
                })
                .collect();
            let y = ops.linear(&stream, &xdev, &wdev, biased.then(|| bdev.binding())).unwrap();
            close(&download(&mut stream, &y), &want, 0.01);
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn normalization_modes_and_fused_silu_match_cpu() {
    use krea2::ops::Norm;
    let (mut stream, ops) = opened();
    for cols in [32usize, 256, 1024, 1152] {
        let rows = 9;
        let x: Vec<_> = (0..rows * cols).map(|i| ((i * 3 % 29) as f32 - 14.) / 8.).collect();
        let w: Vec<_> = (0..cols).map(|i| 1. + (i % 7) as f32 / 16.).collect();
        let xd = upload(&mut stream, &ops, &x, rows, cols);
        let wd = weight_on(&mut stream, &w, vec![cols]);
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
            let normalized = ops.norm(&mut stream, &xd, &wd, mode, 1e-5).unwrap();
            close(&download(&mut stream, &normalized), &want, 0.015);
            if mode == Norm::Group {
                let want: Vec<_> = want.into_iter().map(|x| x / (1. + (-x).exp())).collect();
                let fused = ops.norm_silu(&mut stream, &xd, &wd).unwrap();
                close(&download(&mut stream, &fused), &want, 0.02);
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn causal_and_grouped_attention_match_scalar_softmax() {
    let (mut stream, ops) = opened();
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
            let q = upload(&mut stream, &ops, &q, tokens, heads * dim);
            let k = upload(&mut stream, &ops, &k, tokens, kv * dim);
            let v = upload(&mut stream, &ops, &v, tokens, kv * dim);
            let attention =
                ops.attention(&stream, &q, &k, &v, 1, tokens, heads, kv, dim, causal).unwrap();
            close(&download(&mut stream, &attention), &want, 0.015);
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn rotary_embedding_and_nearest_upsampling_match_cpu_indices() {
    let (mut stream, ops) = opened();
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
    let xd = upload(&mut stream, &ops, &x, tokens, heads * dim);
    let rotated = ops.rope(&stream, &xd, tokens, heads, 10000.).unwrap();
    close(&download(&mut stream, &rotated), &want, 0.015);
    let (height, width, channels) = (3usize, 5usize, 7usize);
    let image: Vec<_> = (0..height * width * channels).map(|i| i as f32 / 32.).collect();
    let input = upload(&mut stream, &ops, &image, height * width, channels);
    let upsampled = ops.upsample(&stream, &input, height, width).unwrap();
    let output = download(&mut stream, &upsampled);
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
