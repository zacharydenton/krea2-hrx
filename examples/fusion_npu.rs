//! Native BF16 qualification, including tails, split reductions and changed inputs.
use anyhow::{ensure, Result};
use half::bf16;
use hrx::Stream;
use krea2::fusion::npu::{Projection, Shape};

fn main() -> Result<()> {
    if let Some(directory) = std::env::args().nth(1) {
        return bench(std::path::Path::new(&directory));
    }
    let residency = hrx::residency::ResidencyManager::new(512 << 20)?;
    for (m, k, n) in [(1, 1, 1), (9, 513, 17), (8, 512, 32), (13, 1025, 31), (9, 513, 2049)] {
        ensure!(
            residency.statistics().reserved_bytes == 0,
            "previous projection retained its budget"
        );
        let mut stream = Stream::open()?.with_memory_budget(residency.budget());
        let shape = Shape { m, k, n };
        let weights: Vec<u16> = (0..n * k)
            .map(|i| bf16::from_f32(((i * 17 % 43) as f32 - 21.) / 16.).to_bits())
            .collect();
        let bias: Vec<u16> =
            (0..n).map(|i| bf16::from_f32((i as f32 - 7.) / 8.).to_bits()).collect();
        let with_bias = n != 1;
        let mut p = Projection::new(
            shape,
            &weights,
            with_bias.then_some(bias.as_slice()),
            &stream,
            None,
        )?;
        let input = stream.allocate(m * k * 2)?;
        let output = stream.allocate(m * n * 2)?;
        let before = p.statistics();
        for replay in 0..3 {
            let values: Vec<u16> = (0..m * k)
                .map(|i| {
                    bf16::from_f32(((i * 13 + replay * 7) % 37) as f32 / 16. - 1.).to_bits()
                })
                .collect();
            stream.upload(input.binding(), bytemuck::cast_slice(&values))?;
            p.execute(&mut stream, input.binding(), output.binding())?;
            let mut actual = vec![0u16; m * n];
            stream.read_blocking(output.binding(), bytemuck::cast_slice_mut(&mut actual))?;
            for row in 0..m {
                for col in 0..n {
                    let sum: f64 = (0..k)
                        .map(|j| {
                            bf16::from_bits(values[row * k + j]).to_f64()
                                * bf16::from_bits(weights[col * k + j]).to_f64()
                        })
                        .sum();
                    let expected = bf16::from_f64(
                        sum + if with_bias { bf16::from_bits(bias[col]).to_f64() } else { 0. },
                    )
                    .to_bits();
                    ensure!(
                        actual[row * n + col] == expected,
                        "{shape:?} replay {replay} [{row},{col}]: {} != {}",
                        bf16::from_bits(actual[row * n + col]),
                        bf16::from_bits(expected)
                    );
                }
            }
        }
        let after = p.statistics();
        ensure!(
            (before.allocations, before.imports) == (after.allocations, after.imports),
            "warm execution allocated or imported"
        );
        println!(
            "{}",
            serde_json::json!({"shape":shape,"replays":3,"exact_f64_oracle":true,"allocations":after.allocations,"imports":after.imports,"storage_bytes":p.storage_bytes(),"source_digest":p.source_digest,"image_digest":p.image_digest})
        );
    }
    ensure!(residency.statistics().reserved_bytes == 0, "final projection retained its budget");
    Ok(())
}

/// Each fresh process alternates completed GPU/NPU stages after ten warmups.
/// Loading/compilation and correctness readback stay outside the timed interval.
fn bench(directory: &std::path::Path) -> Result<()> {
    use krea2::ops::{Ops, Tensor, Weight};
    use std::{sync::Arc, time::Instant};
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("case.json"))?)?;
    let shape: Shape = serde_json::from_value(metadata["shape"].clone())?;
    shape.storage_bytes()?;
    let read = |name: &str| -> Result<Vec<u16>> {
        let bytes = std::fs::read(directory.join(name))?;
        ensure!(
            metadata["digests"][name].as_str() == Some(&hrx::bundle::digest(&bytes)),
            "capture digest mismatch: {name}"
        );
        ensure!(bytes.len() % 2 == 0, "invalid BF16 extent");
        Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    };
    let mut input_values = read("input.bf16")?;
    let weights = read("weights.bf16")?;
    let bias = read("bias.bf16")?;
    ensure!(
        input_values.len() == shape.m * shape.k
            && weights.len() == shape.n * shape.k
            && (bias.is_empty() || bias.len() == shape.n),
        "invalid capture shape"
    );
    let mut stream = Stream::open()?;
    let ops = Ops::new(hrx::BufferPool::new());
    let input = Tensor::from_slice(ops.pool(), &mut stream, &input_values, shape.m, shape.k)?;
    let weight_buffer = Arc::new(stream.allocate(weights.len() * 2)?);
    stream.upload(weight_buffer.binding(), bytemuck::cast_slice(&weights))?;
    let weight = Weight::new(&weight_buffer, 0, vec![shape.n, shape.k], weights.len(), None);
    let bias_buffer = if bias.is_empty() {
        None
    } else {
        let buffer = stream.allocate(bias.len() * 2)?;
        stream.upload(buffer.binding(), bytemuck::cast_slice(&bias))?;
        Some(buffer)
    };
    let bias_view = || bias_buffer.as_ref().map(|b| b.binding());
    let mut p = Projection::new(
        shape,
        &weights,
        (!bias.is_empty()).then_some(bias.as_slice()),
        &stream,
        None,
    )?;
    let mut max_absolute_error = 0f64;
    let mut relative_rms = Vec::new();
    // Reuse the same native graph with a distinct, nonzero input before timing.
    for replay in 0..2 {
        if replay == 1 {
            for (i, v) in input_values.iter_mut().enumerate() {
                *v = bf16::from_f32(
                    bf16::from_bits(*v).to_f32() * -0.75 + (i % 7) as f32 / 128.,
                )
                .to_bits();
            }
        }
        input.upload(&mut stream, &input_values)?;
        let gpu = ops.linear(&stream, &input, &weight, bias_view())?;
        let reference = gpu.download(&mut stream)?;
        let output = ops.tensor(&stream, shape.m, shape.n)?;
        p.execute(&mut stream, input.binding()?, output.binding()?)?;
        let actual = output.download(&mut stream)?;
        ensure!(
            actual.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "nonfinite native output"
        );
        let mut square_error = 0.;
        let mut square_reference = 0.;
        for (&got, &want) in actual.iter().zip(&reference) {
            let got = bf16::from_bits(got).to_f64();
            let want = bf16::from_bits(want).to_f64();
            square_error += (got - want).powi(2);
            square_reference += want.powi(2);
        }
        relative_rms.push((square_error / square_reference.max(f64::MIN_POSITIVE)).sqrt());
        eprintln!(
            "validated replay {replay}: relative RMS vs GPU {}",
            relative_rms.last().unwrap()
        );
        let samples = 1024.min(actual.len());
        for sample in 0..samples {
            let index = sample * (actual.len() - 1) / (samples - 1).max(1);
            let row = index / shape.n;
            let col = index % shape.n;
            let mut expected = 0.;
            let mut magnitude = 0.;
            for j in 0..shape.k {
                let v = bf16::from_bits(input_values[row * shape.k + j]).to_f64()
                    * bf16::from_bits(weights[col * shape.k + j]).to_f64();
                expected += v;
                magnitude += v.abs();
            }
            if !bias.is_empty() {
                expected += bf16::from_bits(bias[col]).to_f64();
            }
            let got = bf16::from_bits(actual[index]).to_f64();
            let error = (got - expected).abs();
            max_absolute_error = max_absolute_error.max(error);
            // BF16 rounding plus conservative FP32 accumulation error bound.
            let gamma = shape.k as f64 * f32::EPSILON as f64;
            let tolerance = expected.abs() / 256. + gamma / (1. - gamma) * magnitude + 1e-7;
            ensure!(error<=tolerance,"f64 oracle replay {replay} index {index}: {got} != {expected}, tolerance {tolerance}");
        }
    }
    // Time the original captured activation, not the changed-input probe.
    input.upload(&mut stream, &read("input.bf16")?)?;
    stream.synchronize()?;
    let mut gpu_samples = Vec::new();
    let mut npu_samples = Vec::new();
    let before = p.statistics();
    for iteration in 0..110 {
        if iteration % 10 == 0 {
            eprintln!("completed stage pairs: {iteration}/110");
        }
        for backend in if iteration % 2 == 0 { [false, true] } else { [true, false] } {
            let start = Instant::now();
            let output = if backend {
                let out = ops.tensor(&stream, shape.m, shape.n)?;
                p.execute(&mut stream, input.binding()?, out.binding()?)?;
                out
            } else {
                ops.linear(&stream, &input, &weight, bias_view())?
            };
            stream.synchronize()?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            drop(output);
            if iteration >= 10 {
                if backend {
                    npu_samples.push(elapsed)
                } else {
                    gpu_samples.push(elapsed)
                }
            }
        }
    }
    let after = p.statistics();
    ensure!(
        (before.allocations, before.imports) == (after.allocations, after.imports),
        "warm native allocations/imports grew"
    );
    let summarize = |samples: &[f64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        serde_json::json!({"median_ms":(sorted[49]+sorted[50])/2.,"p95_ms":sorted[94],"samples_ms":samples})
    };
    println!(
        "{}",
        serde_json::json!({"scope":"completed projection including packing, copies, coherency, NPU, GPU reduction/bias and fences","shape":shape,"gpu":summarize(&gpu_samples),"npu":summarize(&npu_samples),"relative_rms_vs_gpu":relative_rms,"sampled_f64_oracle":true,"max_sampled_absolute_error":max_absolute_error,"storage_bytes":p.storage_bytes(),"allocations":after.allocations,"imports":after.imports,"source_digest":p.source_digest,"image_digest":p.image_digest,"capture":metadata})
    );
    Ok(())
}
