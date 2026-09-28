//! Warm production generation, including text encoding, denoising and VAE.
//! Usage: bench_runtime [--interactive] OUTPUT_RGB [SIZE [COUNT [STEPS]]]
//! SIZE accepts square sides or WIDTHxHEIGHT, comma separated for a size sweep.
//! Interactive mode prints `ready`, then accepts `run` or `quit` on stdin. Keep
//! measurements outside other GPU work; bracket candidate batches with baseline
//! batches to check drift when two resident builds would exceed available memory.
use krea2::pipeline::{Files, Pipeline, Request};
use std::{io::Write, path::Path, time::Instant};
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let interactive = args.peek().is_some_and(|s| s == "--interactive");
    if interactive {
        args.next();
    }
    let output = args.next().expect("output RGB path");
    let size = args.next().unwrap_or_else(|| "256".into());
    let sizes: Vec<(usize, usize)> = size
        .split(',')
        .map(|size| {
            let (width, height) = size.split_once('x').unwrap_or((size, size));
            Ok((width.parse()?, height.parse()?))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(!interactive || sizes.len() == 1, "interactive mode takes one size");
    let count = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(7);
    let steps = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(2);
    for &(width, height) in &sizes {
        anyhow::ensure!(
            [width, height]
                .into_iter()
                .all(|n| (64..=2048).contains(&n) && n.is_multiple_of(16)),
            "dimensions must be multiples of 16 in 64..=2048"
        );
    }
    anyhow::ensure!(count > 0, "sample count must be positive");
    anyhow::ensure!((1..=100).contains(&steps), "steps must be 1..=100");
    let files = Files::of(Path::new("krea2_turbo_int8_convrot")).offline(true).resolve()?;
    let residency = hrx::residency::ResidencyManager::new(64 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    let pipeline = Pipeline::open_in(files, &context, None)?;
    for &(width, height) in &sizes {
        let mut request = Request::new("a red ceramic cup on a wooden table");
        request.width = width;
        request.height = height;
        request.steps = Some(steps);
        request.seed = 37;
        let expected = pipeline.generate(&request, None)?;
        let image_path = if sizes.len() == 1 {
            output.clone()
        } else {
            format!("{output}.{width}x{height}.rgb")
        };
        std::fs::write(&image_path, &expected)?;
        let warm_reserved = residency.statistics().reserved_bytes;
        let warm = context.runtime().statistics();
        let mut samples = Vec::new();
        let mut reserved_samples = Vec::new();
        {
            println!(
                "{}",
                serde_json::json!({"event":"ready", "width":width, "height":height,
            "steps":steps, "warm_reserved_bytes":warm_reserved})
            );
            std::io::stdout().flush()?;
        }
        loop {
            if interactive {
                let mut command = String::new();
                if std::io::stdin().read_line(&mut command)? == 0 || command.trim() == "quit" {
                    break;
                }
                anyhow::ensure!(command.trim() == "run", "expected run or quit");
            } else if samples.len() == count {
                break;
            }
            let start = Instant::now();
            let actual = pipeline.generate(&request, None)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            samples.push(elapsed_ms);
            anyhow::ensure!(actual == expected, "generation replay changed");
            let reserved = residency.statistics().reserved_bytes;
            reserved_samples.push(reserved);
            {
                println!(
                    "{}",
                    serde_json::json!({"event":"sample", "width":width, "height":height, "sample":samples.len(),
                "elapsed_ms":elapsed_ms, "rgb_identical":true,
                "reserved_bytes":reserved})
                );
                std::io::stdout().flush()?;
            }
        }
        let after = context.runtime().statistics();
        let chronological = samples.clone();
        samples.sort_by(f64::total_cmp);
        let count = samples.len();
        let median = if count == 0 {
            0.
        } else if count.is_multiple_of(2) {
            (samples[count / 2 - 1] + samples[count / 2]) * 0.5
        } else {
            samples[count / 2]
        };
        println!(
            "{}",
            serde_json::json!({"scope":"warm full generation", "width":width, "height":height, "steps":steps,
            "median_ms":median, "samples_ms":chronological, "rgb_path":image_path,
            "warm_reserved_bytes":warm_reserved, "tracked_peak_bytes":after.peak_bytes,
            "reserved_bytes_after_samples":reserved_samples,
            "warm_tracked_allocations":after.allocations-warm.allocations})
        );
    }
    drop(pipeline);
    anyhow::ensure!(
        residency.statistics().reserved_bytes == 0,
        "pipeline retained its memory budget"
    );
    Ok(())
}
