//! The ops against arithmetic worked out on the CPU.
//!
//! These are the operations whose rounding is part of the model's contract, so
//! they are compared bit for bit rather than within a tolerance. Skipped when
//! there is no GPU or no compiler.
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::{Binary, Ops, Pool, Tensor, Unary};

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
