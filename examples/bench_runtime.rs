//! Warm production generation, including text encoding, denoising and VAE.
//! Usage: bench_runtime [--interactive] OUTPUT_RGB [SIZE [COUNT [STEPS]]]
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
    let size = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(256);
    let count = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(7);
    let steps = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(2);
    anyhow::ensure!(
        size > 0 && size.is_multiple_of(16),
        "size must be a positive multiple of 16"
    );
    anyhow::ensure!(count > 0, "sample count must be positive");
    anyhow::ensure!((1..=100).contains(&steps), "steps must be 1..=100");
    let files = Files::of(Path::new("krea2_turbo_int8_convrot")).offline(true).resolve()?;
    let residency = hrx::residency::ResidencyManager::new(64 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    let pipeline = Pipeline::open_in(files, &context, None)?;
    let mut request = Request::new("a red ceramic cup on a wooden table");
    request.width = size;
    request.height = size;
    request.steps = Some(steps);
    request.seed = 37;
    let expected = pipeline.generate(&request, None)?;
    std::fs::write(output, &expected)?;
    let warm_reserved = residency.statistics().reserved_bytes;
    let warm = context.runtime().statistics();
    let mut samples = Vec::new();
    if interactive {
        println!(
            "{}",
            serde_json::json!({"event":"ready", "width":size,
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
        anyhow::ensure!(
            residency.statistics().reserved_bytes == warm_reserved,
            "warm residency grew"
        );
        if interactive {
            println!(
                "{}",
                serde_json::json!({"event":"sample", "sample":samples.len(),
                "elapsed_ms":elapsed_ms, "rgb_identical":true})
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
        serde_json::json!({"scope":"warm full generation", "width":size, "height":size, "steps":steps,
            "median_ms":median, "samples_ms":chronological,
            "warm_reserved_bytes":warm_reserved, "tracked_peak_bytes":after.peak_bytes,
            "warm_tracked_allocations":after.allocations-warm.allocations})
    );
    drop(pipeline);
    anyhow::ensure!(
        residency.statistics().reserved_bytes == 0,
        "pipeline retained its memory budget"
    );
    Ok(())
}
