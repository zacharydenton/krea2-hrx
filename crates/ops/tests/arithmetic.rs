//! Device operations checked against CPU arithmetic and alternate layouts.
//! Scheduler rounding is checked exactly; convolution reduction orders use a
//! tolerance. Tests return early when the GPU or compiler is unavailable.
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::{Binary, Layout, Ops, Pool, Tensor, Unary, Weight};

fn usable() -> bool {
    let compiler = loom::compiler(None);
    let found = std::path::Path::new(&compiler).exists()
        || std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path).any(|entry| entry.join(&compiler).exists())
        });
    found && hrx::try_device().is_ok()
}

fn upload(ops: &Ops, values: &[f32], rows: usize, cols: usize) -> Tensor {
    let bits: Vec<u16> = values.iter().map(|&v| from_f32(v)).collect();
    Tensor::from_slice(ops.pool(), &bits, rows, cols).expect("upload")
}

fn download(tensor: &Tensor) -> Vec<f32> {
    tensor.download().expect("download").into_iter().map(to_f32).collect()
}

#[test]
fn the_euler_step_rounds_where_the_sampler_rounds() {
    if !usable() {
        return;
    }
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
fn guidance_combines_as_krea_defines_it() {
    if !usable() {
        return;
    }
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
fn the_pointwise_and_broadcast_operations_agree_with_the_host() {
    if !usable() {
        return;
    }
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
fn a_convolution_reads_its_weight_in_the_order_the_weight_is_in() {
    if !usable() {
        return;
    }
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
