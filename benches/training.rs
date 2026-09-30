//! Resident training operations, plus an opt-in prepared RAW training step.
use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use hrx::{BufferPool, Stream};
use krea2::{
    lora::Factors,
    numerics::from_f32,
    ops::{Ops, Tensor},
    training::{TrainConfig, Trainer, model::Projection, ops as train, optimizer},
};
use std::{
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};

fn kernel_source(name: &str) -> String {
    std::env::var_os("KREA2_BENCH_KERNEL_DIR")
        .map(|directory| {
            std::fs::read_to_string(Path::new(&directory).join(format!("{name}.loom"))).unwrap()
        })
        .unwrap_or_else(|| krea2::kernels::sources::auxiliary(name).unwrap().to_owned())
}

#[derive(Default)]
struct PairedTimings {
    totals: [Duration; 2],
    pairs: u64,
}

impl PairedTimings {
    // The callback must synchronize each operation before returning.
    fn measure(&mut self, iterations: u64, mut run: impl FnMut(bool)) -> Duration {
        let mut candidate_time = Duration::ZERO;
        for _ in 0..iterations {
            for side in [self.pairs as usize % 2, 1 - self.pairs as usize % 2] {
                let start = Instant::now();
                run(side == 1);
                let elapsed = start.elapsed();
                self.totals[side] += elapsed;
                if side == 1 {
                    candidate_time += elapsed;
                }
            }
            self.pairs += 1;
        }
        candidate_time
    }

    fn report(&self, label: &str) {
        if self.pairs > 0 {
            eprintln!(
                "paired {label}: reference {:.3} ms, candidate {:.3} ms, ratio {:.4} ({} pairs, including warm-up)",
                self.totals[0].as_secs_f64() * 1000.0 / self.pairs as f64,
                self.totals[1].as_secs_f64() * 1000.0 / self.pairs as f64,
                self.totals[1].as_secs_f64() / self.totals[0].as_secs_f64(),
                self.pairs,
            );
        }
    }
}

fn dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/dense");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (m, n, k) in [
        (1043, 6144, 1536),
        (1043, 1536, 6144),
        (1043usize, 6144usize, 6144usize),
        (1043, 16384, 6144),
        (1043, 6144, 16384),
        (4115, 6144, 16384),
        (4115, 16384, 6144),
    ] {
        for (name, tile_m, tile_n) in [
            ("gemm_bf16_bf16_nt", 64usize, 64usize),
            ("gemm_bf16_bf16_nt_wide", 128, 64),
            ("train_gemm", 128, 64),
            ("train_gemm_nn", 128, 64),
        ] {
            let mut prepared = None;
            let mut paired = PairedTimings::default();
            let label = format!("{name}/{m}x{n}x{k}");
            group.bench_function(&label, |b| {
                let (stream, kernel, constants, reference, a, w, out, grid) = prepared
                    .get_or_insert_with(|| {
                        let mut stream = Stream::open().unwrap();
                        let ops = Ops::new(BufferPool::new());
                        let a = tensor(&ops, &mut stream, m, k);
                        let w = tensor(&ops, &mut stream, n, k);
                        let out = ops.tensor(&stream, m, n).unwrap();
                        let grid = [n.div_ceil(tile_n) as u32, m.div_ceil(tile_m) as u32, 1];
                        let mut request =
                            hrx::loom::Specialization::new(format!("krea2_{name}"));
                        for (key, value) in [
                            ("m", m),
                            ("n", n),
                            ("k", k),
                            ("asize", m * k),
                            ("bsize", n * k),
                            ("csize", m * n),
                            ("astride", m * k),
                            ("bstride", n * k),
                            ("grid_x", grid[0] as usize),
                            ("grid_y", grid[1] as usize),
                        ] {
                            request
                                .set_config(format!("krea2.{name}.{key}"), value.to_string());
                        }
                        let reports = std::env::var_os("KREA2_BENCH_REPORT_DIR");
                        if reports.is_some() {
                            request.set_report(hrx::loom::ReportMode::Details);
                        }
                        let compiler = krea2::kernels::compiler(None).unwrap();
                        // Compare candidate Loom sources without rebuilding the trainer.
                        let source = kernel_source(name);
                        let artifact = compiler.module(&source).compile(&request).unwrap();
                        // SAFETY: this benchmark uses each kernel's declared matrix layout and tile geometry.
                        let kernel = unsafe { stream.load_artifact(&artifact).unwrap() };
                        let constants = krea2::kernels::Scalars::new()
                            .index(m)
                            .float(1.0)
                            .pack(name, &kernel)
                            .unwrap();
                        let reference =
                            std::env::var_os("KREA2_BENCH_REFERENCE_DIR").map(|directory| {
                                let source = std::fs::read_to_string(
                                    Path::new(&directory).join(format!("{name}.loom")),
                                )
                                .unwrap();
                                let artifact =
                                    compiler.module(&source).compile(&request).unwrap();
                                // SAFETY: the reference uses the same matrix ABI and tile geometry.
                                let kernel =
                                    unsafe { stream.load_artifact(&artifact).unwrap() };
                                let constants = krea2::kernels::Scalars::new()
                                    .index(m)
                                    .float(1.0)
                                    .pack(name, &kernel)
                                    .unwrap();
                                (kernel, constants)
                            });
                        if let Some(directory) = reports {
                            let directory = std::path::PathBuf::from(directory);
                            std::fs::create_dir_all(&directory).unwrap();
                            let stem = label.replace('/', "-");
                            std::fs::write(
                                directory.join(format!("{stem}-compiler.json")),
                                artifact.report().unwrap().json().to_string(),
                            )
                            .unwrap();
                            let mut graph = stream.graph().unwrap();
                            // SAFETY: resident matrices have the configured extents; the graph writes only out.
                            unsafe {
                                graph
                                    .dispatch(
                                        &[],
                                        &kernel,
                                        grid,
                                        [256, 1, 1],
                                        &constants,
                                        &[
                                            a.binding().unwrap(),
                                            w.binding().unwrap(),
                                            out.binding().unwrap(),
                                        ],
                                    )
                                    .unwrap();
                            }
                            let mut graph =
                                graph.finish_profiled(std::slice::from_ref(&label)).unwrap();
                            for sample in 0..4 {
                                let profile = stream.launch_profiled(&mut graph).unwrap();
                                std::fs::write(
                                    directory.join(format!("{stem}-gpu-{sample}.json")),
                                    serde_json::to_vec(&profile).unwrap(),
                                )
                                .unwrap();
                            }
                        }
                        (stream, kernel, constants, reference, a, w, out, grid)
                    });
                let mut dispatch = |kernel, constants| {
                    // SAFETY: bindings and constants are the same validated resident matrices as above.
                    unsafe {
                        stream
                            .dispatch(
                                kernel,
                                *grid,
                                [256, 1, 1],
                                constants,
                                &[
                                    a.binding().unwrap(),
                                    w.binding().unwrap(),
                                    out.binding().unwrap(),
                                ],
                            )
                            .unwrap();
                    }
                    stream.synchronize().unwrap();
                };
                if let Some((reference, reference_constants)) = reference {
                    b.iter_custom(|iterations| {
                        paired.measure(iterations, |candidate| {
                            if candidate {
                                dispatch(kernel, constants);
                            } else {
                                dispatch(reference, reference_constants);
                            }
                        })
                    });
                } else {
                    b.iter(|| dispatch(kernel, constants));
                }
            });
            paired.report(&label);
        }
    }
    group.finish();
}

fn tensor(ops: &Ops, stream: &mut Stream, rows: usize, cols: usize) -> Tensor {
    let values = (0..rows * cols)
        .map(|i| from_f32(((i % 113) as f32 - 56.0) * 0.01))
        .collect::<Vec<_>>();
    Tensor::from_slice(ops.pool(), stream, &values, rows, cols).unwrap()
}

fn adapter_gradients(c: &mut Criterion) {
    let name = "train_gemm_tn_accumulate";
    let mut group = c.benchmark_group("training/adapter_gradient");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (m, n, k) in [
        (32usize, 6144usize, 1043usize),
        (6144, 32, 1043),
        (32, 16384, 1043),
        (16384, 32, 1043),
        (32, 16384, 4115),
        (16384, 32, 4115),
    ] {
        let mut prepared = None;
        group.bench_function(format!("{m}x{n}x{k}"), |b| {
            let (stream, kernel, constants, a, weight, out) =
                prepared.get_or_insert_with(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let a = tensor(&ops, &mut stream, k, m);
                    let weight = tensor(&ops, &mut stream, k, n);
                    let out = train::FloatTensor::zero(&stream, m, n).unwrap();
                    let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
                    for (key, value) in
                        [("m", m), ("n", n), ("k", k), ("grid_x", n / 32), ("grid_y", m / 32)]
                    {
                        request.set_config(format!("krea2.{name}.{key}"), value.to_string());
                    }
                    request.set_report(hrx::loom::ReportMode::Details);
                    let artifact = krea2::kernels::compiler(None)
                        .unwrap()
                        .module(&kernel_source(name))
                        .compile(&request)
                        .unwrap();
                    if let Some(directory) = std::env::var_os("KREA2_BENCH_REPORT_DIR") {
                        let directory = Path::new(&directory);
                        std::fs::create_dir_all(directory).unwrap();
                        std::fs::write(
                            directory.join(format!("{name}-{m}x{n}x{k}-compiler.json")),
                            artifact.report().unwrap().json().to_string(),
                        )
                        .unwrap();
                    }
                    // SAFETY: the configured input/output matrices match the kernel ABI.
                    let kernel = unsafe { stream.load_artifact(&artifact).unwrap() };
                    let constants = krea2::kernels::Scalars::new()
                        .index(m * n)
                        .float(0.001)
                        .pack(name, &kernel)
                        .unwrap();
                    stream.synchronize().unwrap();
                    (stream, kernel, constants, a, weight, out)
                });
            b.iter(|| {
                // SAFETY: each workgroup accumulates its own 32x32 output tile.
                unsafe {
                    stream
                        .dispatch(
                            kernel,
                            [(n / 32) as u32, (m / 32) as u32, 1],
                            [128, 1, 1],
                            constants,
                            &[a.binding().unwrap(), weight.binding().unwrap(), out.binding()],
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
            });
        });
    }
    group.finish();
}

fn projections(c: &mut Criterion) {
    assert!(!krea2::kernels::native_profile(), "disable profiling for benchmarks");
    let mut group = c.benchmark_group("training/projection");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (tokens, inputs, outputs) in [(1024, 6144, 6144), (4096, 6144, 16384)] {
        for transposed in [true, false] {
            let mut prepared = None;
            let pass =
                if transposed { "forward_backward_transposed" } else { "forward_backward" };
            group.bench_function(format!("{pass}/{tokens}x{inputs}x{outputs}"), |b| {
                let (mut stream, ops, projection, x, grad, base, dx) =
                    prepared.take().unwrap_or_else(|| {
                        let mut stream = Stream::open().unwrap();
                        let ops = Ops::new(BufferPool::new());
                        let rank = 32;
                        let factors = Factors {
                            rank,
                            inputs,
                            outputs,
                            alpha: 32.0,
                            a: vec![0.01; rank * inputs],
                            b: vec![0.01; outputs * rank],
                        };
                        let projection = Projection::new(&ops, &mut stream, &factors).unwrap();
                        let x = tensor(&ops, &mut stream, tokens, inputs);
                        let grad = tensor(&ops, &mut stream, tokens, outputs);
                        let base = tensor(&ops, &mut stream, tokens, outputs);
                        let dx = tensor(&ops, &mut stream, tokens, inputs);
                        (stream, ops, projection, x, grad, base, dx)
                    });
                let mut run = || {
                    projection.a.grad.clear(&stream).unwrap();
                    projection.b.grad.clear(&stream).unwrap();
                    let y = projection.forward(&ops, &stream, &x, &base, 1.0).unwrap();
                    let g = if transposed {
                        projection_backward_transposed(
                            &projection,
                            &ops,
                            &stream,
                            &x,
                            &grad,
                            &dx,
                        )
                    } else {
                        projection.backward(&ops, &stream, &x, &grad, &dx).unwrap()
                    };
                    stream.synchronize().unwrap();
                    black_box((y, g));
                };
                run();
                b.iter(run);
                prepared = Some((stream, ops, projection, x, grad, base, dx));
            });
        }
    }
    group.finish();
}

// Retain the unfused algebra as a runnable baseline for projection optimizations.
fn projection_backward_transposed(
    projection: &Projection,
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    grad: &Tensor,
    base_grad: &Tensor,
) -> Tensor {
    let scale = projection.alpha / projection.a.master.rows() as f32;
    let low = train::matmul(ops, stream, x, &projection.a.value, 1.0).unwrap();
    let gt = train::transpose(ops, stream, grad).unwrap();
    let lt = train::transpose(ops, stream, &low).unwrap();
    let db = train::matmul_float(ops, stream, &gt, &lt, scale).unwrap();
    train::accumulate(ops, stream, &projection.b.grad, &db).unwrap();
    let bt = train::transpose(ops, stream, &projection.b.value).unwrap();
    let dl = train::matmul(ops, stream, grad, &bt, scale).unwrap();
    let dlt = train::transpose(ops, stream, &dl).unwrap();
    let xt = train::transpose(ops, stream, x).unwrap();
    let da = train::matmul_float(ops, stream, &dlt, &xt, 1.0).unwrap();
    train::accumulate(ops, stream, &projection.a.grad, &da).unwrap();
    let at = train::transpose(ops, stream, &projection.a.value).unwrap();
    let dx = train::matmul(ops, stream, &dl, &at, 1.0).unwrap();
    train::add_scaled(ops, stream, base_grad, &dx, 1.0).unwrap()
}

fn attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/attention");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for tokens in [1024, 1043, 1536, 1537, 4096, 4115] {
        for pass in ["forward", "backward", "forward_backward"] {
            let mut prepared = None;
            group.bench_function(format!("{pass}/{tokens}"), |b| {
                let (mut stream, ops, q, k, v, grad) = prepared.take().unwrap_or_else(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let q = tensor(&ops, &mut stream, tokens, 6144);
                    let k = tensor(&ops, &mut stream, tokens, 1536);
                    let v = tensor(&ops, &mut stream, tokens, 1536);
                    let grad = tensor(&ops, &mut stream, tokens, 6144);
                    (stream, ops, q, k, v, grad)
                });
                let cached = (pass == "backward")
                    .then(|| train::attention(&ops, &stream, &q, &k, &v).unwrap());
                stream.synchronize().unwrap();
                let mut run = || {
                    let fresh = (pass != "backward")
                        .then(|| train::attention(&ops, &stream, &q, &k, &v).unwrap());
                    let f = cached.as_ref().or(fresh.as_ref()).unwrap();
                    let gradients = (pass != "forward").then(|| {
                        train::attention_backward(&ops, &stream, &q, &k, &v, f, &grad).unwrap()
                    });
                    stream.synchronize().unwrap();
                    black_box((f, gradients));
                };
                run();
                b.iter(run);
                prepared = Some((stream, ops, q, k, v, grad));
            });
        }
    }
    group.finish();
}

fn fusion_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/fusion_attention");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (batch, sequence) in [(43, 12), (128, 12), (1, 43), (1, 128)] {
        let mut prepared = None;
        group.bench_function(format!("forward_backward/{batch}x{sequence}x2560"), |b| {
            let (mut stream, ops, q, k, v, grad) = prepared.take().unwrap_or_else(|| {
                let mut stream = Stream::open().unwrap();
                let ops = Ops::new(BufferPool::new());
                let q = tensor(&ops, &mut stream, batch * sequence, 2560);
                let k = tensor(&ops, &mut stream, batch * sequence, 2560);
                let v = tensor(&ops, &mut stream, batch * sequence, 2560);
                let grad = tensor(&ops, &mut stream, batch * sequence, 2560);
                (stream, ops, q, k, v, grad)
            });
            let mut run = || {
                let forward =
                    train::attention_batched(&ops, &stream, &q, &k, &v, sequence).unwrap();
                let backward =
                    train::attention_backward(&ops, &stream, &q, &k, &v, &forward, &grad)
                        .unwrap();
                stream.synchronize().unwrap();
                black_box((forward, backward));
            };
            run();
            b.iter(run);
            prepared = Some((stream, ops, q, k, v, grad));
        });
    }
    group.finish();
}

fn frozen_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/frozen_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    let pair_direct = std::env::var_os("KREA2_BENCH_PAIR_FROZEN_DIRECT").is_some();
    for (tokens, inputs, outputs) in [
        (1043, 6144, 1536),
        (1043, 1536, 6144),
        (1043, 6144, 6144),
        (1043, 16384, 6144),
        (1043, 6144, 16384),
        (4115, 16384, 6144),
        (4115, 6144, 16384),
    ] {
        for method in ["transpose", "direct", "cached_transpose"] {
            let mut prepared = None;
            let mut paired = PairedTimings::default();
            let label = format!("{method}/{tokens}x{inputs}x{outputs}");
            group.bench_function(&label, |b| {
                let (mut stream, ops, gradient, weight, cached) =
                    prepared.take().unwrap_or_else(|| {
                        let mut stream = Stream::open().unwrap();
                        let ops = Ops::new(BufferPool::new());
                        let gradient = tensor(&ops, &mut stream, tokens, outputs);
                        let weight = tensor(&ops, &mut stream, outputs, inputs);
                        let cached = (method == "cached_transpose")
                            .then(|| train::transpose(&ops, &stream, &weight).unwrap());
                        stream.synchronize().unwrap();
                        (stream, ops, gradient, weight, cached)
                    });
                let mut run = |candidate: bool| {
                    let dx = if !candidate || method == "direct" {
                        train::matmul_nn(&ops, &stream, &gradient, &weight, 1.0).unwrap()
                    } else if let Some(transpose) = &cached {
                        train::matmul(&ops, &stream, &gradient, transpose, 1.0).unwrap()
                    } else {
                        let transpose = train::transpose(&ops, &stream, &weight).unwrap();
                        train::matmul(&ops, &stream, &gradient, &transpose, 1.0).unwrap()
                    };
                    stream.synchronize().unwrap();
                    black_box(dx);
                };
                run(true);
                if pair_direct && cached.is_some() {
                    run(false);
                    b.iter_custom(|iterations| paired.measure(iterations, &mut run));
                } else {
                    b.iter(|| run(true));
                }
                prepared = Some((stream, ops, gradient, weight, cached));
            });
            paired.report(&label);
        }
    }
    group.finish();
}

fn gated_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/gated_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (tokens, width, sigmoid) in
        [(1043, 6144, true), (1043, 16384, false), (4115, 6144, true), (4115, 16384, false)]
    {
        let mut prepared = None;
        let mut paired = PairedTimings::default();
        let activation_name = if sigmoid { "sigmoid" } else { "silu" };
        let label = format!("{activation_name}/{tokens}x{width}");
        group.bench_function(&label, |b| {
            let (stream, ops, x, activation, value, grad) = prepared.get_or_insert_with(|| {
                let mut stream = Stream::open().unwrap();
                let ops = Ops::new(BufferPool::new());
                let tensors = std::array::from_fn::<_, 4, _>(|_| {
                    tensor(&ops, &mut stream, tokens, width)
                });
                let [x, activation, value, grad] = tensors;
                (stream, ops, x, activation, value, grad)
            });
            let mut run = |candidate| {
                let result = if candidate {
                    train::gated_backward(ops, stream, x, activation, value, grad, sigmoid)
                        .unwrap()
                } else {
                    let dv =
                        ops.binary(stream, grad, activation, krea2::ops::Binary::Mul).unwrap();
                    let product =
                        ops.binary(stream, grad, value, krea2::ops::Binary::Mul).unwrap();
                    let dx =
                        train::activation_backward(ops, stream, x, &product, sigmoid).unwrap();
                    (dx, dv)
                };
                stream.synchronize().unwrap();
                black_box(result);
            };
            run(false);
            run(true);
            b.iter_custom(|iterations| paired.measure(iterations, &mut run));
        });
        paired.report(&label);
    }
    group.finish();
}

fn modulation_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/modulation_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (tokens, width) in [(1, 6144), (1043, 6144), (4115, 6144)] {
        let mut prepared = None;
        let mut paired = PairedTimings::default();
        let label = format!("{tokens}x{width}");
        group.bench_function(&label, |b| {
            let (stream, ops, r, branch, a, normalized, outputs) =
                prepared.get_or_insert_with(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let [r, branch, a, normalized] = std::array::from_fn::<_, 4, _>(|_| {
                        tensor(&ops, &mut stream, tokens, width)
                    });
                    let outputs = std::array::from_fn::<_, 2, _>(|_| {
                        train::FloatTensor::zero(&stream, 6, width).unwrap()
                    });
                    (stream, ops, r, branch, a, normalized, outputs)
                });
            let mut run = |candidate| {
                if candidate {
                    train::modulation_backward(
                        ops,
                        stream,
                        r,
                        branch,
                        a,
                        normalized,
                        &outputs[1],
                        3,
                    )
                    .unwrap();
                } else {
                    let gate = ops.binary(stream, r, branch, krea2::ops::Binary::Mul).unwrap();
                    train::sum_rows_accumulate(ops, stream, &gate, &outputs[0], 5).unwrap();
                    let scale =
                        ops.binary(stream, a, normalized, krea2::ops::Binary::Mul).unwrap();
                    train::sum_rows_accumulate(ops, stream, &scale, &outputs[0], 3).unwrap();
                    train::sum_rows_accumulate(ops, stream, a, &outputs[0], 4).unwrap();
                }
                stream.synchronize().unwrap();
            };
            run(false);
            run(true);
            b.iter_custom(|iterations| paired.measure(iterations, &mut run));
        });
        paired.report(&label);
    }
    group.finish();
}

fn norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/norm_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for rows in [5usize, 1043 * 48, 1043 * 12, 4115 * 48, 4115 * 12] {
        let cols = 128;
        let mut prepared = None;
        let mut paired = PairedTimings::default();
        let label = format!("{rows}x{cols}");
        group.bench_function(&label, |b| {
            let (stream, kernels, x, grad, scale, outputs) =
                prepared.get_or_insert_with(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let x = tensor(&ops, &mut stream, rows, cols);
                    let grad = tensor(&ops, &mut stream, rows, cols);
                    let weights: Vec<_> = (0..cols).map(|i| (i % 113) as f32 * 0.001).collect();
                    let scale =
                        train::FloatTensor::from_slice(&mut stream, 1, cols, &weights).unwrap();
                    let outputs = std::array::from_fn::<_, 2, _>(|_| {
                        ops.tensor(&stream, rows, cols).unwrap()
                    });
                    let compiler = krea2::kernels::compiler(None).unwrap();
                    let reports = std::env::var_os("KREA2_BENCH_REPORT_DIR");
                    let mut kernels = Vec::new();
                    for name in ["train_norm_backward", "train_norm_backward_head"] {
                        let mut request =
                            hrx::loom::Specialization::new(format!("krea2_{name}"));
                        for (key, value) in [
                            ("cols", cols),
                            ("size", rows * cols),
                            ("grid_x", rows),
                            ("grid_y", 1),
                        ] {
                            request
                                .set_config(format!("krea2.{name}.{key}"), value.to_string());
                        }
                        if reports.is_some() {
                            request.set_report(hrx::loom::ReportMode::Details);
                        }
                        let source = if name == "train_norm_backward" {
                            krea2::kernels::sources::auxiliary(name).unwrap().to_owned()
                        } else {
                            kernel_source(name)
                        };
                        let artifact = compiler.module(&source).compile(&request).unwrap();
                        if let Some(directory) = &reports {
                            std::fs::create_dir_all(directory).unwrap();
                            std::fs::write(
                                Path::new(directory)
                                    .join(format!("{name}-{label}-compiler.json")),
                                artifact.report().unwrap().json().to_string(),
                            )
                            .unwrap();
                        }
                        // SAFETY: both kernels have the norm ABI and one wave per row.
                        let kernel = unsafe { stream.load_artifact(&artifact).unwrap() };
                        let constants = krea2::kernels::Scalars::new()
                            .index(rows)
                            .float(1e-5)
                            .pack(name, &kernel)
                            .unwrap();
                        kernels.push((kernel, constants));
                    }
                    (stream, kernels, x, grad, scale, outputs)
                });
            let mut dispatch = |candidate: bool| {
                let side = usize::from(candidate);
                let (kernel, constants) = &kernels[side];
                // SAFETY: all bindings match the specialized extents and declared matrix ABI.
                unsafe {
                    stream
                        .dispatch(
                            kernel,
                            [rows as u32, 1, 1],
                            [32, 1, 1],
                            constants,
                            &[
                                x.binding().unwrap(),
                                grad.binding().unwrap(),
                                scale.binding(),
                                outputs[side].binding().unwrap(),
                            ],
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
            };
            dispatch(false);
            dispatch(true);
            b.iter_custom(|iterations| paired.measure(iterations, &mut dispatch));
            assert_eq!(
                outputs[0].download(stream).unwrap(),
                outputs[1].download(stream).unwrap(),
                "norm reference output: {label}"
            );
        });
        paired.report(&label);
    }
    group.finish();
}

fn modulated_norm_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/modulated_norm_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for rows in [1usize, 1043, 4115] {
        let cols = 6144;
        let mut prepared = None;
        let mut paired = PairedTimings::default();
        let label = format!("{rows}x{cols}");
        group.bench_function(&label, |b| {
            let (stream, ops, x, grad, residual, modulation, scale) = prepared
                .get_or_insert_with(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let [x, grad, residual] = std::array::from_fn::<_, 3, _>(|_| {
                        tensor(&ops, &mut stream, rows, cols)
                    });
                    let modulation = tensor(&ops, &mut stream, 1, cols);
                    let weights: Vec<_> = (0..cols).map(|i| (i % 113) as f32 * 0.001).collect();
                    let scale =
                        train::FloatTensor::from_slice(&mut stream, 1, cols, &weights).unwrap();
                    (stream, ops, x, grad, residual, modulation, scale)
                });
            let mut run = |candidate| {
                let result = if candidate {
                    train::norm_modulated_backward(
                        ops,
                        stream,
                        x,
                        grad,
                        scale.binding(),
                        modulation,
                        residual,
                        1e-5,
                    )
                    .unwrap()
                } else {
                    let one_plus = train::one_plus(ops, stream, modulation).unwrap();
                    let scaled =
                        ops.binary(stream, grad, &one_plus, krea2::ops::Binary::Mul).unwrap();
                    let norm =
                        train::norm_backward(ops, stream, x, &scaled, scale.binding(), 1e-5)
                            .unwrap();
                    train::add_scaled(ops, stream, residual, &norm, 1.0).unwrap()
                };
                stream.synchronize().unwrap();
                black_box(result);
            };
            run(false);
            run(true);
            b.iter_custom(|iterations| paired.measure(iterations, &mut run));
        });
        paired.report(&label);
    }
    group.finish();
}

fn adamw(c: &mut Criterion) {
    let mut prepared = None;
    c.bench_function("training/adamw/rank32x16384", |b| {
        let (mut stream, ops, projection, gradient, config) =
            prepared.take().unwrap_or_else(|| {
                let mut stream = Stream::open().unwrap();
                let ops = Ops::new(BufferPool::new());
                let projection = Projection::new(
                    &ops,
                    &mut stream,
                    &Factors {
                        rank: 32,
                        inputs: 16384,
                        outputs: 6144,
                        alpha: 32.0,
                        a: vec![0.01; 32 * 16384],
                        b: vec![0.01; 32 * 6144],
                    },
                )
                .unwrap();
                let gradient = train::FloatTensor::from_slice(
                    &mut stream,
                    32,
                    16384,
                    &vec![0.001; 32 * 16384],
                )
                .unwrap();
                (stream, ops, projection, gradient, TrainConfig::default())
            });
        let mut run = || {
            // Adam clears gradients; reseed identical nonzero values on-device each iteration.
            stream.copy(projection.a.grad.binding(), gradient.binding()).unwrap();
            optimizer::adamw(&ops, &stream, &projection.a, &config, 1.0, 1.0, 1.0).unwrap();
            stream.synchronize().unwrap();
        };
        run();
        b.iter(run);
        prepared = Some((stream, ops, projection, gradient, config));
    });
}

fn optimizer_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/optimizer");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for prepared_graph in [false, true] {
        let mut prepared = None;
        let name =
            if prepared_graph { "graph_448_parameters" } else { "individual_448_parameters" };
        group.bench_function(name, |b| {
            let (mut stream, ops, projections, mut optimizer, mut reset, config) =
                prepared.take().unwrap_or_else(|| {
                    let mut stream = Stream::open().unwrap();
                    let ops = Ops::new(BufferPool::new());
                    let projections: Vec<_> = (0..28)
                        .flat_map(|_| krea2::lora::PROJECTIONS)
                        .map(|(_, outputs, inputs)| {
                            Projection::new(
                                &ops,
                                &mut stream,
                                &Factors {
                                    rank: 32,
                                    inputs,
                                    outputs,
                                    alpha: 32.0,
                                    a: vec![0.01; 32 * inputs],
                                    b: vec![0.01; outputs * 32],
                                },
                            )
                            .unwrap()
                        })
                        .collect();
                    let parameters: Vec<_> =
                        projections.iter().flat_map(|p| [&p.a, &p.b]).collect();
                    let optimizer =
                        optimizer::PreparedOptimizer::new(&stream, &parameters).unwrap();
                    let mut reset = stream.graph().unwrap();
                    for p in parameters {
                        reset.fill(&[], p.grad.binding(), 0x38).unwrap();
                    }
                    let reset = reset.finish().unwrap();
                    (stream, ops, projections, optimizer, reset, TrainConfig::default())
                });
            let parameters: Vec<_> = projections.iter().flat_map(|p| [&p.a, &p.b]).collect();
            let mut step = 0;
            let mut run = || {
                step += 1;
                stream.launch(&mut reset).unwrap();
                let norm = if prepared_graph {
                    optimizer.update(&mut stream, &config, step).unwrap()
                } else {
                    optimizer::update_parameters(&ops, &mut stream, &parameters, &config, step)
                        .unwrap()
                };
                stream.synchronize().unwrap();
                black_box(norm);
            };
            run();
            b.iter(run);
            prepared = Some((stream, ops, projections, optimizer, reset, config));
        });
    }
    group.finish();
}

fn full_step(c: &mut Criterion) {
    let Some(path) = std::env::var_os("KREA2_TRAIN_BENCH_CONFIG") else { return };
    let mut trainer = None;
    let mut group = c.benchmark_group("training/prepared");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("step", |b| {
        let trainer = trainer.get_or_insert_with(|| {
            let mut config = TrainConfig::read(Path::new(&path)).unwrap();
            config.steps = usize::MAX;
            let mut trainer =
                Trainer::open(config).expect("prepare the external dataset first");
            black_box(trainer.train_step().unwrap().unwrap());
            trainer
        });
        b.iter(|| black_box(trainer.train_step().unwrap().unwrap()));
    });
    group.finish();
}

criterion_group!(
    benches,
    dense,
    adapter_gradients,
    projections,
    attention,
    fusion_attention,
    frozen_backward,
    gated_backward,
    modulation_backward,
    norm_backward,
    modulated_norm_backward,
    adamw,
    optimizer_update,
    full_step
);
criterion_main!(benches);
