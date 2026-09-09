//! Quantized transformer GEMMs against independent integer CPU dot products.
//! Run explicitly with --ignored on gfx1151.
use half::{bf16, f16};
use hrx::{Buffer, Constants, Stream};
use hrx_core as hrx;
use std::path::Path;

struct Harness {
    compiler: hrx::loom::Compiler,
    stream: Stream,
}
impl Harness {
    fn new() -> Self {
        Self {
            compiler: hrx::loom::Compiler::resolve(None).expect("provisioned Loom compiler"),
            stream: Stream::open().expect("gfx1151 device"),
        }
    }
    fn run(
        &mut self,
        stem: &str,
        cfg: &[(&str, String)],
        grid: [u32; 3],
        threads: u32,
        scalars: &[u64],
        data: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("kernels").join(format!("{stem}.loom"));
        let path = if path.is_file() {
            path
        } else {
            root.join("../experiments").join(format!("{stem}.loom"))
        };
        let source = std::fs::read_to_string(path).unwrap();
        let symbol = format!("krea2_{stem}");
        let mut request = hrx::loom::Request::new(&source, &symbol);
        request.config = cfg
            .iter()
            .map(|(key, value)| (format!("krea2.{stem}.{key}"), value.clone()))
            .collect();
        let path = self
            .compiler
            .compile(&request, &hrx::bundle::cache_root().unwrap().join("kernels"))
            .unwrap();
        // Safety: trusted checked-in source compiled through HRX. Every test below
        // sizes the bindings from the same dimensions passed as kernel configuration.
        let kernel = unsafe { self.stream.load(&path, &symbol).unwrap() };
        let buffers: Vec<Buffer> =
            data.iter().map(|bytes| self.stream.allocate(bytes.len()).unwrap()).collect();
        for (buffer, bytes) in buffers.iter().zip(data) {
            self.stream.upload_queued(buffer, 0, bytes).unwrap();
        }
        let indices: Vec<_> = scalars.iter().map(|&v| u32::try_from(v).unwrap()).collect();
        let constants = Constants::indices(&kernel, &indices).unwrap();
        let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
        unsafe {
            self.stream
                .dispatch(&kernel, grid, [threads, 1, 1], &constants, &bindings)
                .unwrap();
        }
        let reads: Vec<_> =
            bindings.iter().map(|&v| self.stream.read_queued(v).unwrap()).collect();
        reads.into_iter().map(|r| r.wait(&mut self.stream).unwrap()).collect()
    }
}
fn bytes<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn halves(v: &[u8], bf: bool) -> Vec<f64> {
    v.chunks_exact(2)
        .map(|b| {
            let n = u16::from_le_bytes(b.try_into().unwrap());
            if bf {
                bf16::from_bits(n).to_f64()
            } else {
                f16::from_bits(n).to_f64()
            }
        })
        .collect()
}
fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32 - 50.) * scale / 50.).collect()
}
fn close(got: &[f64], want: &[f64], abs: f64, rel: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= abs + rel * b.abs(),
            "element {i}: {a} vs {b}"
        );
    }
}
fn cfg(v: &[(&'static str, usize)]) -> Vec<(&'static str, String)> {
    v.iter().map(|&(k, v)| (k, v.to_string())).collect()
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn rotary_rounds_normalized_and_rotated_values_to_bf16() {
    let mut h = Harness::new();
    let (tokens, q_heads, kv_heads, dim) = (3, 9, 2, 128);
    let stride = (q_heads + 2 * kv_heads) * dim;
    let input: Vec<_> = values(tokens * stride, 0.7)
        .into_iter()
        .map(|x| f16::from_f32(bf16::from_f32(x).to_f32()))
        .collect();
    let qs = values(dim, 0.3);
    let ks = values(dim, -0.2);
    let cos: Vec<_> = (0..tokens * dim).map(|i| ((i / 2) as f32 * 0.03).cos()).collect();
    let sin: Vec<_> = (0..tokens * dim).map(|i| ((i / 2) as f32 * 0.03).sin()).collect();
    let mut config = cfg(&[
        ("row_stride", stride),
        ("q_heads", q_heads),
        ("kv_heads", kv_heads),
        ("k_offset", q_heads * dim),
    ]);
    config.push(("eps", "0.00001".into()));
    let data = [
        bytes(&input),
        bytes(&qs),
        bytes(&ks),
        bytes(&cos),
        bytes(&sin),
        vec![0; tokens * q_heads * dim * 2],
        vec![0; tokens * kv_heads * dim * 2],
        vec![0; tokens * kv_heads * dim * 2],
    ];
    let out =
        h.run("rope_qknorm_f16", &config, [tokens as u32, 1, 1], 256, &[tokens as u64], &data);
    for (slot, heads, offset, scale) in
        [(5, q_heads, 0, &qs), (6, kv_heads, q_heads * dim, &ks)]
    {
        let actual = halves(&out[slot], false);
        let mut expected = Vec::new();
        for t in 0..tokens {
            for head in 0..heads {
                let row = &input[t * stride + offset + head * dim..][..dim];
                let rms = (row.iter().map(|x| x.to_f64().powi(2)).sum::<f64>() / dim as f64
                    + 1e-5)
                    .sqrt();
                let norm: Vec<_> = row
                    .iter()
                    .enumerate()
                    .map(|(c, x)| {
                        bf16::from_f64(x.to_f64() / rms * (1. + scale[c] as f64)).to_f64()
                    })
                    .collect();
                for c in 0..dim {
                    let partner = if c % 2 == 0 { -norm[c + 1] } else { norm[c - 1] };
                    // The reference RoPE multiplies and adds in float32 before
                    // its BF16 cast; evaluating a midpoint in f64 changes ties.
                    expected.push(
                        bf16::from_f32(
                            norm[c] as f32 * cos[t * dim + c]
                                + partner as f32 * sin[t * dim + c],
                        )
                        .to_f64(),
                    );
                }
            }
        }
        assert!(actual.iter().all(|&v| bf16::from_f64(v).to_f64() == v));
        close(&actual, &expected, 1e-3, 4e-3);
    }
    let v = halves(&out[7], false);
    for t in 0..tokens {
        for c in 0..kv_heads * dim {
            assert_eq!(
                v[t * kv_heads * dim + c],
                input[t * stride + (q_heads + kv_heads) * dim + c].to_f64()
            );
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn integer_gemms_preserve_pitches_bf16_residuals_and_swiglu_order() {
    let mut h = Harness::new();
    for bits in [4usize, 8] {
        for tile in [128usize, 256] {
            if bits == 8 && tile == 128 {
                continue;
            }
            for mode in ["plain", "resid", "swiglu"] {
                for pad in [0usize, 128] {
                    if tile == 128 && pad != 0 {
                        continue;
                    }
                    let (m, k, n) = (17usize, 128usize, 128usize);
                    let stride = k + pad;
                    let a: Vec<i8> =
                        (0..m * stride).map(|i| ((i * 3 % 15) as i8) - 7).collect();
                    let w: Vec<i8> =
                        (0..n * stride).map(|i| ((i * 7 % 15) as i8) - 7).collect();
                    let pack = |v: &[i8]| -> Vec<u8> {
                        if bits == 8 {
                            v.iter().map(|&x| x as u8).collect()
                        } else {
                            v.chunks_exact(2)
                                .map(|x| (x[0] as u8 & 15) | ((x[1] as u8 & 15) << 4))
                                .collect()
                        }
                    };
                    let mut ws = vec![0.01f32; n];
                    ws[3] = 0.;
                    let mut scales = vec![0.02f32; m];
                    scales[m - 1] = 0.;
                    let residual: Vec<_> =
                        values(m * n, 0.5).into_iter().map(bf16::from_f32).collect();
                    let gates = values(n, 0.3);
                    let full: Vec<f64> = (0..m * n)
                        .map(|i| {
                            let (r, c) = (i / n, i % n);
                            let dot = (0..k)
                                .map(|j| a[r * stride + j] as i32 * w[c * stride + j] as i32)
                                .sum::<i32>();
                            dot as f64 * ws[c] as f64 * scales[r] as f64
                        })
                        .collect();
                    let stem = format!(
                        "gemm_i{bits}{}{}",
                        if mode == "plain" {
                            ""
                        } else if mode == "resid" {
                            "_resid"
                        } else {
                            "_swiglu"
                        },
                        if tile == 256 { "_256" } else { "" }
                    );
                    let config = cfg(&[
                        ("k_size", k),
                        ("n_size", n),
                        ("k_stride", stride),
                        ("m_group", if tile == 256 { 4 } else { 1 }),
                    ]);
                    let mut data = vec![pack(&a), pack(&w), bytes(&ws), bytes(&scales)];
                    let want = if mode == "resid" {
                        data.extend([bytes(&residual), bytes(&gates)]);
                        full.iter()
                            .enumerate()
                            .map(|(i, &x)| {
                                bf16::from_f32(
                                    residual[i].to_f32()
                                        + bf16::from_f32(
                                            gates[i % n] * bf16::from_f32(x as f32).to_f32(),
                                        )
                                        .to_f32(),
                                )
                                .to_f64()
                            })
                            .collect::<Vec<_>>()
                    } else if mode == "swiglu" {
                        data.push(vec![0; m * n]);
                        (0..m * n / 2)
                            .map(|i| {
                                let (r, c) = (i / (n / 2), i % (n / 2));
                                let a = bf16::from_f64(full[r * n + (c / 16) * 32 + c % 16])
                                    .to_f64();
                                let b =
                                    bf16::from_f64(full[r * n + (c / 16) * 32 + c % 16 + 16])
                                        .to_f64();
                                bf16::from_f64(
                                    bf16::from_f64(a / (1. + (-a).exp())).to_f64() * b,
                                )
                                .to_f64()
                            })
                            .collect()
                    } else {
                        data.push(vec![0; m * n * 2]);
                        full.into_iter().map(|x| bf16::from_f64(x).to_f64()).collect()
                    };
                    let out = h.run(
                        &stem,
                        &config,
                        [1, m.div_ceil(tile) as u32, 1],
                        256,
                        &[m as u64],
                        &data,
                    );
                    let actual = halves(&out[4], mode == "resid");
                    assert!(
                        actual.iter().all(|&v| bf16::from_f64(v).to_f64() == v),
                        "{stem} must preserve the model's BF16 output boundary"
                    );
                    close(&actual, &want, 2e-3, 4e-3);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn production_attention_tiles_match_grouped_cpu_softmax() {
    let mut h = Harness::new();
    let (heads, kv, d) = (4usize, 1usize, 128usize);
    for tokens in [17usize, 32, 65] {
        let capacity = (tokens + 79).div_ceil(64) * 64;
        let mut q: Vec<_> =
            values(capacity * heads * d, 0.5).into_iter().map(f16::from_f32).collect();
        let mut k: Vec<_> =
            values(capacity * kv * d, 0.3).into_iter().map(f16::from_f32).collect();
        let mut v: Vec<_> =
            values(capacity * kv * d, 0.7).into_iter().map(f16::from_f32).collect();
        q[tokens * heads * d..].fill(f16::ZERO);
        k[tokens * kv * d..].fill(f16::ZERO);
        v[tokens * kv * d..].fill(f16::ZERO);
        let mut want = vec![0.; tokens * heads * d];
        for t in 0..tokens {
            for head in 0..heads {
                let scores: Vec<_> = (0..tokens)
                    .map(|j| {
                        (0..d)
                            .map(|c| {
                                q[(t * heads + head) * d + c].to_f64() * k[j * d + c].to_f64()
                            })
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum = scores.iter().map(|x| (x - max).exp()).sum::<f64>();
                for (j, score) in scores.iter().enumerate() {
                    for c in 0..d {
                        want[(t * heads + head) * d + c] +=
                            (score - max).exp() / sum * v[j * d + c].to_f64();
                    }
                }
            }
        }
        for query32 in [false, true] {
            let stem = if query32 { "attention_query32" } else { "attention_gqa_lds_f16_wmma" };
            let mut config = cfg(&[
                ("q_stride", heads * d),
                ("kv_stride", kv * d),
                ("tokens", tokens),
                ("token_capacity", capacity),
                ("out_stride", heads * d),
            ]);
            config.push(("scale", (1. / (d as f64).sqrt()).to_string()));
            let vv = if query32 {
                (0..kv * d)
                    .flat_map(|c| (0..capacity).map(move |t| (t, c)))
                    .map(|(t, c)| v[t * kv * d + c])
                    .collect::<Vec<_>>()
            } else {
                v.clone()
            };
            let rows = if query32 { 32 } else { 16 };
            let out = h.run(
                stem,
                &config,
                [tokens.div_ceil(rows) as u32, kv as u32, 1],
                if query32 { 256 } else { 128 },
                &[tokens as u64, 0],
                &[bytes(&q), bytes(&k), bytes(&vv), vec![0; tokens * heads * d * 2]],
            );
            close(&halves(&out[3], false), &want, 2e-2, 2e-2);
        }
    }
}

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn quantized_preparation_matches_the_kronecker_hadamard_and_preserves_padding() {
    let mut h = Harness::new();
    let (tokens, width, stride) = (3usize, 2048usize, 2176usize);
    let h4 = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
    let coefficient = |mut i: usize, mut j: usize| {
        let mut c = 1f64;
        for _ in 0..4 {
            c *= h4[i % 4][j % 4];
            i /= 4;
            j /= 4;
        }
        c / 16.
    };
    let norm_input: Vec<_> =
        values(tokens * width, 1.5).into_iter().map(bf16::from_f32).collect();
    let half_input: Vec<_> =
        values(tokens * width, 0.5).into_iter().map(f16::from_f32).collect();
    let norm = values(width, 0.1);
    let modulation = values(width, 0.2);
    let shift = values(width, 0.15);
    let gate: Vec<_> = values(tokens * stride, 0.7).into_iter().map(f16::from_f32).collect();
    for bits in [4usize, 8] {
        for kind in ["norm", "gated", "plain"] {
            let mut config = cfg(&[("width", width), ("out_stride", stride)]);
            let mut formed = Vec::new();
            for t in 0..tokens {
                let mean_square = norm_input[t * width..(t + 1) * width]
                    .iter()
                    .map(|v| v.to_f64().powi(2))
                    .sum::<f64>()
                    / width as f64;
                for c in 0..width {
                    formed.push(match kind {
                        "norm" => {
                            let rounded = |x: f64| bf16::from_f64(x).to_f64();
                            let normalized = rounded(
                                norm_input[t * width + c].to_f64()
                                    / (mean_square + 1e-5).sqrt()
                                    * (1. + norm[c] as f64),
                            );
                            rounded(
                                rounded(rounded(1. + modulation[c] as f64) * normalized)
                                    + shift[c] as f64,
                            )
                        }
                        "gated" => {
                            let attention =
                                bf16::from_f64(half_input[t * width + c].to_f64()).to_f64();
                            let sigmoid = bf16::from_f64(
                                1. / (1. + (-gate[t * stride + c].to_f64()).exp()),
                            )
                            .to_f64();
                            bf16::from_f64(attention * sigmoid).to_f64()
                        }
                        _ => half_input[t * width + c].to_f64(),
                    });
                }
            }
            let mut data = match kind {
                "norm" => {
                    config.push(("eps", "1e-5".into()));
                    vec![bytes(&norm_input), bytes(&norm), bytes(&modulation), bytes(&shift)]
                }
                "gated" => {
                    config.push(("gate_stride", stride.to_string()));
                    vec![bytes(&half_input), bytes(&gate)]
                }
                _ => vec![bytes(&half_input)],
            };
            // Gate pitch must be a multiple of 256; use a tight gate for this case.
            if kind == "gated" {
                config.retain(|(k, _)| *k != "gate_stride");
                config.push(("gate_stride", (stride + 128).to_string()));
                let mut padded = vec![f16::ZERO; tokens * (stride + 128)];
                for t in 0..tokens {
                    padded[t * (stride + 128)..t * (stride + 128) + width]
                        .copy_from_slice(&gate[t * stride..t * stride + width]);
                }
                data[1] = bytes(&padded);
            }
            let qindex = data.len();
            let row_bytes = stride * bits / 8;
            data.push(vec![0xa5; tokens * row_bytes]);
            data.push(vec![0; tokens * 4]);
            let out = h.run(
                &format!("prepare_{kind}_i{bits}"),
                &config,
                [tokens as u32, 1, 1],
                256,
                &[tokens as u64],
                &data,
            );
            let mut mismatches = 0;
            let levels = if bits == 4 { 7f64 } else { 127. };
            for t in 0..tokens {
                let mut rotated = vec![0.; width];
                for group in 0..width / 256 {
                    for i in 0..256 {
                        rotated[group * 256 + i] = (0..256)
                            .map(|j| formed[t * width + group * 256 + j] * coefficient(i, j))
                            .sum();
                    }
                }
                let scale = rotated.iter().map(|x| x.abs()).fold(1e-30f64, f64::max) / levels;
                let actual_scale =
                    f32::from_le_bytes(out[qindex + 1][t * 4..t * 4 + 4].try_into().unwrap())
                        as f64;
                assert!(
                    (actual_scale - scale).abs() <= scale * 0.002 + 1e-30,
                    "{kind} scale: {actual_scale} vs {scale}"
                );
                for (c, &v) in rotated.iter().enumerate() {
                    let code = if bits == 8 {
                        out[qindex][t * row_bytes + c] as i8 as i32
                    } else {
                        let byte = out[qindex][t * row_bytes + c / 2];
                        let nibble = (byte >> (4 * (c % 2))) & 15;
                        (nibble as i32 ^ 8) - 8
                    };
                    let want = (v / scale).round_ties_even().clamp(-levels, levels) as i32;
                    assert!((code - want).abs() <= 1, "{kind} int{bits}: {code} vs {want}");
                    mismatches += usize::from(code != want);
                }
                assert!(out[qindex][t * row_bytes + width * bits / 8..(t + 1) * row_bytes]
                    .iter()
                    .all(|&v| v == 0xa5));
            }
            assert!(
                mismatches as f64 / ((tokens * width) as f64) < 0.02,
                "{kind} int{bits}: {mismatches} rounding mismatches"
            );
        }
    }
}
