//! Completed GPU dispatch latency with resident operands; no model weights needed.
use criterion::{Criterion, criterion_group, criterion_main};
use hrx::{Buffer, Constants, Kernel, Stream};
use krea2::kernels::{shape, sources};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn compile(
    stream: &Stream,
    stem: &str,
    config: &[(&str, usize)],
    tokens: usize,
    reference: bool,
) -> Kernel {
    let mut spec = hrx::loom::Specialization::new(format!("krea2_{stem}"));
    for (key, value) in config {
        spec.set_config(format!("krea2.{stem}.{key}"), value.to_string());
    }
    if stem == "gemm_i8_resid_256x256" {
        spec.set_config(
            format!("krea2.{stem}.schedule_rows"),
            usize::from(shape::gemm_schedule_rows(tokens)).to_string(),
        );
    }
    if stem.starts_with("attention") {
        spec.set_config(format!("krea2.{stem}.scale"), "0.08838834764831845");
    }
    let override_path = (!reference).then(|| std::env::var_os("KREA2_BENCH_SOURCE")).flatten();
    let source = override_path.map(|p| std::fs::read_to_string(p).expect("candidate source"));
    let source = source.as_deref().unwrap_or_else(|| sources::block(stem).unwrap());
    let reports = (!reference).then(|| std::env::var_os("KREA2_BENCH_REPORT_DIR")).flatten();
    if reports.is_some() {
        spec.set_report(hrx::loom::ReportMode::Details);
    }
    let compiler = krea2::kernels::compiler(None).expect("compiler");
    let artifact = compiler.module(source).compile(&spec).unwrap();
    if let Some(directory) = reports {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join(format!("{stem}.json")),
            artifact.report().unwrap().json().to_string(),
        )
        .unwrap();
    }
    // SAFETY: callers bind the production ABI with sizes matching this specialization.
    unsafe { stream.load_artifact(&artifact).unwrap() }
}

fn upload(stream: &mut Stream, data: &[u8]) -> Buffer {
    let buffer = stream.allocate(data.len()).unwrap();
    stream.upload(buffer.binding(), data).unwrap();
    buffer
}

fn gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    for (name, family, k, n, residual) in [
        ("gemm_qkvg", "gemm_i8_256", 6144, 15360, false),
        ("gemm_gu", "gemm_i8_swiglu_256", 6144, 32768, false),
        ("gemm_wo", "gemm_i8_resid_256", 6144, 6144, true),
        ("gemm_down", "gemm_i8_resid_256", 16384, 6144, true),
    ] {
        for m in [1043usize, 4095, 4096, 4109, 12301, 16397] {
            let tile = shape::gemm_tile(name);
            let mut prepared = None;
            group.bench_function(format!("{name}/{m}"), |b| {
                let (stream, kernel, constants, buffers, grid) =
                    prepared.get_or_insert_with(|| {
                        let pitch = shape::gemm_pitch(k);
                        let mut stream = Stream::open().unwrap();
                        let kernel = compile(
                            &stream,
                            shape::gemm_source(family, tile),
                            &[
                                ("k_size", k),
                                ("k_stride", pitch),
                                ("n_size", n),
                                ("m_group", shape::GEMM_M_GROUP),
                            ],
                            m,
                            false,
                        );
                        let operand = |rows: usize, salt: usize| -> Vec<u8> {
                            (0..rows * pitch)
                                .map(|i| {
                                    if i % pitch < k {
                                        ((i.wrapping_mul(73) + salt) % 255) as u8
                                    } else {
                                        0
                                    }
                                })
                                .collect()
                        };
                        let buffers = [
                            upload(&mut stream, &operand(m, 17)),
                            upload(&mut stream, &operand(n, 31)),
                            upload(&mut stream, bytemuck::cast_slice(&vec![1e-3f32; n])),
                            upload(&mut stream, bytemuck::cast_slice(&vec![1e-2f32; m])),
                            stream.allocate(m * n * 2).unwrap(),
                            upload(&mut stream, bytemuck::cast_slice(&vec![0.25f32; n])),
                        ];
                        let bindings: Vec<_> = buffers[..if residual { 6 } else { 5 }]
                            .iter()
                            .map(Buffer::binding)
                            .collect();
                        let constants = Constants::indices(&kernel, &[m as u32]).unwrap();
                        let grid =
                            [(n / tile.columns()) as u32, shape::gemm_grid_rows(m) as u32, 1];
                        stream.synchronize().unwrap();
                        if std::env::var_os("KREA2_BENCH_SOURCE").is_some() {
                            let reference = compile(
                                &stream,
                                shape::gemm_source(family, tile),
                                &[
                                    ("k_size", k),
                                    ("k_stride", pitch),
                                    ("n_size", n),
                                    ("m_group", shape::GEMM_M_GROUP),
                                ],
                                m,
                                true,
                            );
                            let mut expected = None;
                            for implementation in [&reference, &kernel] {
                                stream.fill(buffers[4].binding(), 0).unwrap();
                                let args =
                                    Constants::indices(implementation, &[m as u32]).unwrap();
                                // SAFETY: candidate uses the same trusted GEMM ABI and sized owners.
                                unsafe {
                                    stream
                                        .dispatch(
                                            implementation,
                                            grid,
                                            [tile.threads(), 1, 1],
                                            &args,
                                            &bindings,
                                        )
                                        .unwrap();
                                }
                                let mut output =
                                    vec![
                                        0u8;
                                        m * n * if family.contains("swiglu") { 1 } else { 2 }
                                    ];
                                stream
                                    .read_blocking(
                                        buffers[4].binding().slice(0, output.len()).unwrap(),
                                        &mut output,
                                    )
                                    .unwrap();
                                if let Some(expected) = &expected {
                                    assert!(&output == expected, "candidate output differs");
                                } else {
                                    expected = Some(output);
                                }
                            }
                        }
                        (stream, kernel, constants, buffers, grid)
                    });
                let bindings: Vec<_> = buffers[..if residual { 6 } else { 5 }]
                    .iter()
                    .map(Buffer::binding)
                    .collect();
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        // Reset the residual before timing so iterations have identical inputs.
                        stream.fill(buffers[4].binding(), 0).unwrap();
                        stream.synchronize().unwrap();
                        let began = Instant::now();
                        // SAFETY: production grid and ABI, with live owners sized above.
                        unsafe {
                            stream
                                .dispatch(
                                    kernel,
                                    *grid,
                                    [tile.threads(), 1, 1],
                                    constants,
                                    &bindings,
                                )
                                .unwrap();
                        }
                        stream.synchronize().unwrap();
                        elapsed += began.elapsed();
                    }
                    elapsed
                });
            });
        }
    }
    group.finish();
}

fn attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    for tokens in [4109usize, 12301] {
        let rows = shape::attention_rows(tokens);
        let mut prepared = None;
        group.bench_function(tokens.to_string(), |b| {
            let (stream, kernel, constants, buffers) = prepared.get_or_insert_with(|| {
                let capacity = shape::capacity(tokens);
                let mut stream = Stream::open().unwrap();
                let kernel = compile(
                    &stream,
                    shape::attention_source(tokens),
                    &[
                        ("q_stride", 6144),
                        ("kv_stride", 1536),
                        ("out_stride", 6144),
                        ("tokens", tokens),
                        ("token_capacity", capacity),
                    ],
                    tokens,
                    false,
                );
                let mut random = 0x12345678u32;
                let mut buffers = Vec::new();
                for (stride, transposed) in [(6144, false), (1536, false), (1536, true)] {
                    let mut data = vec![half::f16::ZERO; capacity * stride];
                    for token in 0..tokens {
                        for column in 0..stride {
                            random ^= random << 13;
                            random ^= random >> 17;
                            random ^= random << 5;
                            let index = if transposed {
                                column * capacity + token
                            } else {
                                token * stride + column
                            };
                            data[index] =
                                half::f16::from_f32(random as f32 / u32::MAX as f32 - 0.5);
                        }
                    }
                    buffers.push(upload(&mut stream, bytemuck::cast_slice(&data)));
                }
                buffers.push(stream.allocate(tokens * 6144 * 2).unwrap());
                let constants = Constants::indices(&kernel, &[tokens as u32, 0]).unwrap();
                stream.synchronize().unwrap();
                if std::env::var_os("KREA2_BENCH_SOURCE").is_some() {
                    let reference = compile(
                        &stream,
                        shape::attention_source(tokens),
                        &[
                            ("q_stride", 6144),
                            ("kv_stride", 1536),
                            ("out_stride", 6144),
                            ("tokens", tokens),
                            ("token_capacity", capacity),
                        ],
                        tokens,
                        true,
                    );
                    let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
                    let mut expected = None;
                    for implementation in [&reference, &kernel] {
                        stream.fill(buffers[3].binding(), 0xff).unwrap();
                        let args =
                            Constants::indices(implementation, &[tokens as u32, 0]).unwrap();
                        // SAFETY: trusted source with the same attention ABI and padded operands.
                        unsafe {
                            stream
                                .dispatch(
                                    implementation,
                                    [tokens.div_ceil(rows) as u32, 12, 1],
                                    [(rows * 8) as u32, 1, 1],
                                    &args,
                                    &bindings,
                                )
                                .unwrap();
                        }
                        let mut output = vec![0u8; tokens * 6144 * 2];
                        stream.read_blocking(buffers[3].binding(), &mut output).unwrap();
                        if let Some(expected) = &expected {
                            assert!(&output == expected, "candidate output differs");
                        } else {
                            expected = Some(output);
                        }
                    }
                }
                (stream, kernel, constants, buffers)
            });
            let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
            b.iter(|| {
                // SAFETY: padded Q/K, transposed V and output match the selected query tile.
                unsafe {
                    stream
                        .dispatch(
                            kernel,
                            [tokens.div_ceil(rows) as u32, 12, 1],
                            [(rows * 8) as u32, 1, 1],
                            constants,
                            &bindings,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, gemm, attention);
criterion_main!(benches);
