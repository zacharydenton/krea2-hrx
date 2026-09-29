//! Resident training operations, plus an opt-in prepared RAW training step.
use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use hrx::{BufferPool, Stream};
use krea2::{
    lora::Factors,
    numerics::from_f32,
    ops::{Ops, Tensor},
    training::{TrainConfig, Trainer, model::Projection, ops as train, optimizer},
};
use std::{hint::black_box, path::Path, time::Duration};

fn dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/dense");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (m, n, k) in [
        (1043usize, 6144usize, 6144usize),
        (1043, 16384, 6144),
        (1043, 6144, 16384),
        (4115, 6144, 16384),
    ] {
        for (name, tile_m, tile_n) in [
            ("gemm_bf16_bf16_nt", 64usize, 64usize),
            ("gemm_bf16_bf16_nt_wide", 128, 64),
            ("train_gemm", 128, 64),
            ("train_gemm_nn", 128, 64),
        ] {
            let mut prepared = None;
            let label = format!("{name}/{m}x{n}x{k}");
            group.bench_function(&label, |b| {
                let (stream, kernel, constants, a, w, out, grid) =
                    prepared.get_or_insert_with(|| {
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
                        let artifact = compiler
                            .module(krea2::kernels::sources::auxiliary(name).unwrap())
                            .compile(&request)
                            .unwrap();
                        // SAFETY: this benchmark uses each kernel's declared matrix layout and tile geometry.
                        let kernel = unsafe { stream.load_artifact(&artifact).unwrap() };
                        let constants = krea2::kernels::Scalars::new()
                            .index(m)
                            .float(1.0)
                            .pack(name, &kernel)
                            .unwrap();
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
                        (stream, kernel, constants, a, w, out, grid)
                    });
                b.iter(|| {
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
                });
            });
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

fn projections(c: &mut Criterion) {
    assert!(!krea2::kernels::native_profile(), "disable profiling for benchmarks");
    let mut group = c.benchmark_group("training/projection");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (tokens, inputs, outputs) in [(1024, 6144, 6144), (4096, 6144, 16384)] {
        let mut prepared = None;
        group.bench_function(format!("forward_backward/{tokens}x{inputs}x{outputs}"), |b| {
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
                let g = projection.backward(&ops, &stream, &x, &grad, &dx).unwrap();
                stream.synchronize().unwrap();
                black_box((y, g));
            };
            run();
            b.iter(run);
            prepared = Some((stream, ops, projection, x, grad, base, dx));
        });
    }
    group.finish();
}

fn attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/attention");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for tokens in [1024, 4096] {
        for backward in [false, true] {
            let mut prepared = None;
            let pass = if backward { "forward_backward" } else { "forward" };
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
                let mut run = || {
                    let f = train::attention(&ops, &stream, &q, &k, &v).unwrap();
                    let gradients = backward.then(|| {
                        train::attention_backward(&ops, &stream, &q, &k, &v, &f, &grad).unwrap()
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

fn frozen_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("training/frozen_backward");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    for (tokens, inputs, outputs) in [
        (1043, 6144, 6144),
        (1043, 16384, 6144),
        (1043, 6144, 16384),
        (4115, 16384, 6144),
        (4115, 6144, 16384),
    ] {
        for direct in [false, true] {
            let mut prepared = None;
            let method = if direct { "direct" } else { "transpose" };
            group.bench_function(format!("{method}/{tokens}x{inputs}x{outputs}"), |b| {
                let (mut stream, ops, gradient, weight) =
                    prepared.take().unwrap_or_else(|| {
                        let mut stream = Stream::open().unwrap();
                        let ops = Ops::new(BufferPool::new());
                        let gradient = tensor(&ops, &mut stream, tokens, outputs);
                        let weight = tensor(&ops, &mut stream, outputs, inputs);
                        (stream, ops, gradient, weight)
                    });
                let mut run = || {
                    let dx = if direct {
                        train::matmul_nn(&ops, &stream, &gradient, &weight, 1.0).unwrap()
                    } else {
                        let transpose = train::transpose(&ops, &stream, &weight).unwrap();
                        train::matmul(&ops, &stream, &gradient, &transpose, 1.0).unwrap()
                    };
                    stream.synchronize().unwrap();
                    black_box(dx);
                };
                run();
                b.iter(run);
                prepared = Some((stream, ops, gradient, weight));
            });
        }
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

criterion_group!(benches, dense, projections, attention, frozen_backward, adamw, full_step);
criterion_main!(benches);
