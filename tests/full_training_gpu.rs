//! Independent numerical checks for full-transformer training storage.
use hrx::{BufferPool, Stream};
use krea2::{
    kernels::Scalars,
    ops::{Ops, config},
    training::full::numerics,
};

#[test]
#[ignore = "requires gfx1151"]
fn stochastic_storage_matches_philox_scalar_oracle() {
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for n in [1usize, 31, 256, 259, 4097] {
        let mut values: Vec<f32> =
            (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.00012345).collect();
        if n >= 31 {
            values[..10].copy_from_slice(&[
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                f32::from_bits(0x7fff_ffff),
                f32::from_bits(0xffff_ffff),
                f32::MAX,
                -f32::MAX,
                f32::from_bits(1),
                -f32::from_bits(1),
                -0.0,
            ]);
        }
        let input = stream.allocate(n * 4).unwrap();
        let output = stream.allocate(n * 2).unwrap();
        let clock = stream.allocate(16).unwrap();
        let counter = [7u32, 1, 3, 0x53524e47];
        stream.upload(input.binding(), bytemuck::cast_slice(&values)).unwrap();
        stream.upload(clock.binding(), bytemuck::cast_slice(&counter)).unwrap();
        let key = [0xfedc_ba98u32, 0x7654_3210];
        // SAFETY: full_store writes each of n BF16 elements once; all bindings match its ABI.
        unsafe {
            ops.launch_1d(
                &stream,
                "train_full_store",
                config(&[]),
                &Scalars::new().index(n).index(key[0] as usize).index(key[1] as usize),
                &[input.binding(), output.binding(), clock.binding()],
                n,
            )
            .unwrap();
        }
        let mut actual = vec![0u16; n];
        stream.read_blocking(output.binding(), bytemuck::cast_slice_mut(&mut actual)).unwrap();
        for (i, (&got, &value)) in actual.iter().zip(&values).enumerate() {
            if value.is_nan() {
                assert!(krea2::numerics::to_f32(got).is_nan());
                continue;
            }
            let random = numerics::philox(
                [i as u32, counter[0], counter[1], counter[2]],
                [key[0], key[1] ^ counter[3]],
            )[0];
            assert_eq!(got, numerics::stochastic_bf16(value, random), "element {i} of {n}");
        }
    }
}

#[test]
#[ignore = "requires gfx1151"]
fn quantized_adamw_matches_scalar_updates_and_clears_gradients() {
    use krea2::numerics::{from_f32, to_f32};
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for n in [1usize, 31, 256, 259, 4097] {
        let parts = n.div_ceil(256);
        let weight = stream.allocate(n * 2).unwrap();
        let grad = stream.allocate(n * 2).unwrap();
        let first = stream.allocate(n).unwrap();
        let second = stream.allocate(n).unwrap();
        let scales = stream.allocate(parts * 8).unwrap();
        let maps = stream.allocate(2048).unwrap();
        let clock = stream.allocate(16).unwrap();
        let controls = stream.allocate(32).unwrap();
        let signed = numerics::codebook(true);
        let unsigned = numerics::codebook(false);
        let codes: Vec<_> = signed.into_iter().chain(unsigned).collect();
        stream.upload(maps.binding(), bytemuck::cast_slice(&codes)).unwrap();
        for b in [&first, &second, &scales] {
            stream.fill(b.binding(), 0).unwrap();
        }
        let mut w: Vec<_> = (0..n).map(|i| from_f32(((i % 17) as f32 - 8.0) * 0.03)).collect();
        let mut m = vec![0u8; n];
        let mut v = vec![0u8; n];
        let mut maxes = vec![0f32; parts * 2];
        stream.upload(weight.binding(), bytemuck::cast_slice(&w)).unwrap();
        let key = [0xfedc_ba98u32, 0x7654_3210];
        for step in 1..=8u32 {
            let gradient: Vec<_> = (0..n)
                .map(|i| {
                    from_f32(if i / 256 == 1 {
                        0.0
                    } else {
                        ((i * 13 + step as usize) % 37) as f32 * 0.003 - 0.054
                    })
                })
                .collect();
            let c = [
                0.002f32,
                0.01,
                0.9,
                0.999,
                1e-8,
                0.25,
                (1.0 - 0.9f64.powi(step as i32)) as f32,
                (1.0 - 0.999f64.powi(step as i32)) as f32,
            ];
            let counter = [step, 0, 0, 0x57454947u32];
            stream.upload(grad.binding(), bytemuck::cast_slice(&gradient)).unwrap();
            stream.upload(controls.binding(), bytemuck::cast_slice(&c)).unwrap();
            stream.upload(clock.binding(), bytemuck::cast_slice(&counter)).unwrap();
            // SAFETY: each subgroup owns 256 parameter elements and exactly two block scales.
            unsafe {
                ops.launch(
                    &stream,
                    "train_full_adamw",
                    config(&[("parts", parts)]),
                    &Scalars::new().index(n).index(key[0] as usize).index(key[1] as usize),
                    &[
                        controls.binding(),
                        weight.binding(),
                        grad.binding(),
                        first.binding(),
                        second.binding(),
                        scales.binding(),
                        maps.binding(),
                        clock.binding(),
                    ],
                    parts,
                    1,
                    32,
                )
                .unwrap();
            }
            for block in 0..parts {
                let range = block * 256..((block + 1) * 256).min(n);
                let mut nm = vec![0f32; range.len()];
                let mut nv = nm.clone();
                for (j, i) in range.clone().enumerate() {
                    let g = to_f32(gradient[i]) * c[5];
                    nm[j] = (signed[m[i] as usize] * maxes[block]) * c[2] + g * (1.0 - c[2]);
                    nv[j] = (unsigned[v[i] as usize] * maxes[parts + block]) * c[3]
                        + g * g * (1.0 - c[3]);
                }
                let mm = nm.iter().fold(0f32, |a, b| a.max(b.abs()));
                let vm = nv.iter().fold(0f32, |a, b| a.max(*b));
                maxes[block] = mm;
                maxes[parts + block] = vm;
                for (j, i) in range.enumerate() {
                    let next = to_f32(w[i]) * (1.0 - c[0] * c[1])
                        - c[0] * ((nm[j] / c[6]) / ((nv[j] / c[7]).sqrt() + c[4]));
                    let random =
                        numerics::philox([i as u32, step, 0, 0], [key[0], key[1] ^ counter[3]])
                            [0];
                    w[i] = numerics::stochastic_bf16(next, random);
                    m[i] = numerics::quantize(nm[j], mm, &signed);
                    v[i] = numerics::quantize(nv[j], vm, &unsigned);
                }
            }
            let mut actual = vec![0u16; n];
            let mut am = vec![0u8; n];
            let mut av = am.clone();
            let mut scales_got = vec![0f32; parts * 2];
            stream
                .read_blocking(weight.binding(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            stream.read_blocking(first.binding(), &mut am).unwrap();
            stream.read_blocking(second.binding(), &mut av).unwrap();
            stream
                .read_blocking(scales.binding(), bytemuck::cast_slice_mut(&mut scales_got))
                .unwrap();
            assert_eq!(actual, w, "weights n={n},step={step}");
            assert_eq!(am, m, "first n={n},step={step}");
            assert_eq!(av, v, "second n={n},step={step}");
            for (a, b) in scales_got.iter().zip(&maxes) {
                assert!((a - b).abs() <= 1e-6 * b.abs() + 1e-12, "scale {a} {b}");
            }
            // Rebase the scalar oracle on the actual encoded state: reduction/FMA rounding can differ by one ULP.
            maxes = scales_got;
            stream
                .read_blocking(grad.binding(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            assert!(actual.iter().all(|&b| b == 0));
        }
    }
}

#[test]
#[ignore = "requires gfx1151"]
fn normalization_scale_gradient_matches_cpu_and_accumulates() {
    use krea2::numerics::{from_f32, to_f32};
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    for (rows, cols) in [(3, 31), (17, 128), (5, 2560), (2, 6144)] {
        let x: Vec<_> =
            (0..rows * cols).map(|i| from_f32(((i * 13) % 71) as f32 * 0.01 - 0.3)).collect();
        let dy: Vec<_> =
            (0..rows * cols).map(|i| from_f32(((i * 17) % 43) as f32 * 0.02 - 0.2)).collect();
        let xb = stream.allocate(x.len() * 2).unwrap();
        let gb = stream.allocate(x.len() * 2).unwrap();
        let inv = stream.allocate(rows * 4).unwrap();
        let out = stream.allocate(cols * 4).unwrap();
        stream.upload(xb.binding(), bytemuck::cast_slice(&x)).unwrap();
        stream.upload(gb.binding(), bytemuck::cast_slice(&dy)).unwrap();
        stream.fill(out.binding(), 0).unwrap();
        let cfg = config(&[("rows", rows), ("cols", cols), ("size", rows * cols)]);
        // SAFETY: both kernels receive complete row-major tensors and correctly sized outputs.
        unsafe {
            ops.launch(
                &stream,
                "train_full_norm_inv",
                cfg.clone(),
                &Scalars::new().index(rows),
                &[xb.binding(), inv.binding()],
                rows,
                1,
                32,
            )
            .unwrap();
            for _ in 0..2 {
                ops.launch_1d(
                    &stream,
                    "train_full_norm_scale",
                    cfg.clone(),
                    &Scalars::new().index(cols),
                    &[xb.binding(), gb.binding(), inv.binding(), out.binding()],
                    cols,
                )
                .unwrap();
            }
        }
        let mut actual = vec![0f32; cols];
        stream.read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual)).unwrap();
        let mut expected = vec![0f64; cols];
        for r in 0..rows {
            let inverse = 1.0
                / (x[r * cols..(r + 1) * cols]
                    .iter()
                    .map(|&b| f64::from(to_f32(b)).powi(2))
                    .sum::<f64>()
                    / cols as f64
                    + 1e-5)
                    .sqrt();
            for c in 0..cols {
                expected[c] += 2.0
                    * f64::from(to_f32(x[r * cols + c]))
                    * f64::from(to_f32(dy[r * cols + c]))
                    * inverse;
            }
        }
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((f64::from(a) - b).abs() < 2e-5 * b.abs() + 1e-5, "{a} {b}");
        }
    }
}

#[test]
#[ignore = "requires gfx1151; 1000 optimizer updates for each of three seeds"]
fn quantized_optimizer_converges_close_to_fp32_on_noisy_regression() {
    use krea2::numerics::{from_f32, to_f32};
    let mut stream = Stream::open().unwrap();
    let ops = Ops::new(BufferPool::new());
    // Independent separable least squares: each coordinate has two observations,
    // target +/- 0.01, and a different feature scale. The irreducible MSE is 1e-4.
    let n = 259usize;
    let parts = n.div_ceil(256);
    let weight = stream.allocate(n * 2).unwrap();
    let grad = stream.allocate(n * 2).unwrap();
    let first = stream.allocate(n).unwrap();
    let second = stream.allocate(n).unwrap();
    let scales = stream.allocate(parts * 8).unwrap();
    let maps = stream.allocate(2048).unwrap();
    let clock = stream.allocate(16).unwrap();
    let controls = stream.allocate(32).unwrap();
    let codes: Vec<_> =
        numerics::codebook(true).into_iter().chain(numerics::codebook(false)).collect();
    stream.upload(maps.binding(), bytemuck::cast_slice(&codes)).unwrap();
    for seed in [7u32, 37, 103] {
        for b in [&weight, &first, &second, &scales] {
            stream.fill(b.binding(), 0).unwrap();
        }
        let target: Vec<f32> =
            (0..n).map(|i| (((i * 17 + seed as usize) % 101) as f32 - 50.0) * 0.01).collect();
        let curvature: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32).collect();
        let mut w = vec![0u16; n];
        let mut baseline = vec![0f32; n];
        let mut bm = baseline.clone();
        let mut bv = baseline.clone();
        for step in 1..=1000 {
            stream.read_blocking(weight.binding(), bytemuck::cast_slice_mut(&mut w)).unwrap();
            let gradient: Vec<_> = w
                .iter()
                .enumerate()
                .map(|(i, &b)| from_f32(2.0 * curvature[i] * (to_f32(b) - target[i])))
                .collect();
            let c = [
                0.003f32,
                0.0,
                0.9,
                0.999,
                1e-8,
                1.0,
                (1.0 - f64::from(0.9f32).powi(step)) as f32,
                (1.0 - f64::from(0.999f32).powi(step)) as f32,
            ];
            for i in 0..n {
                let g = 2.0 * curvature[i] * (baseline[i] - target[i]);
                bm[i] = c[2] * bm[i] + (1.0 - c[2]) * g;
                bv[i] = c[3] * bv[i] + (1.0 - c[3]) * g * g;
                baseline[i] -= c[0] * (bm[i] / c[6]) / ((bv[i] / c[7]).sqrt() + c[4]);
            }
            stream.upload(grad.binding(), bytemuck::cast_slice(&gradient)).unwrap();
            stream.upload(controls.binding(), bytemuck::cast_slice(&c)).unwrap();
            stream
                .upload(
                    clock.binding(),
                    bytemuck::cast_slice(&[step as u32, 0, 0, 0x57454947u32]),
                )
                .unwrap();
            // SAFETY: persistent buffers have n elements; each wave owns its moment block and scales.
            unsafe {
                ops.launch(
                    &stream,
                    "train_full_adamw",
                    config(&[("parts", parts)]),
                    &Scalars::new().index(n).index(seed as usize).index(913),
                    &[
                        controls.binding(),
                        weight.binding(),
                        grad.binding(),
                        first.binding(),
                        second.binding(),
                        scales.binding(),
                        maps.binding(),
                        clock.binding(),
                    ],
                    parts,
                    1,
                    32,
                )
                .unwrap();
            }
        }
        stream.read_blocking(weight.binding(), bytemuck::cast_slice_mut(&mut w)).unwrap();
        let objective = |w: &[f32]| {
            w.iter()
                .enumerate()
                .map(|(i, &v)| {
                    f64::from(curvature[i]) * f64::from(v - target[i]).powi(2) + 1e-4
                })
                .sum::<f64>()
                / n as f64
        };
        let loss = objective(&w.into_iter().map(to_f32).collect::<Vec<_>>());
        let reference = objective(&baseline);
        eprintln!("seed {seed}: quantized loss {loss}, FP32 {reference}");
        assert!(
            loss.is_finite() && loss <= reference * 1.05,
            "seed {seed}: {loss} versus {reference}"
        );
        assert!(loss < objective(&vec![0.0; n]) * 0.01, "did not learn the regression");
    }
}
