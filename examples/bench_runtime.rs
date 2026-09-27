//! Warm production generation, including text encoding, denoising and VAE.
use krea2::pipeline::{Files, Pipeline, PipelineOptions, Request};
use std::{path::Path, time::Instant};
fn main() -> anyhow::Result<()> {
    let output = std::env::args().nth(1).expect("output RGB path");
    let backend = match std::env::args().nth(2).as_deref() {
        None | Some("gpu") => krea2::fusion::FusionBackend::Gpu,
        Some("npu") => krea2::fusion::FusionBackend::Npu,
        Some("auto") => krea2::fusion::FusionBackend::Auto,
        _ => anyhow::bail!("expected gpu|npu|auto"),
    };
    let size = std::env::args().nth(3).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(256);
    let count = std::env::args().nth(4).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(7);
    anyhow::ensure!(
        size > 0 && size.is_multiple_of(16),
        "size must be a positive multiple of 16"
    );
    anyhow::ensure!(count > 0, "sample count must be positive");
    let files = Files::of(Path::new("krea2_turbo_int8_convrot")).offline(true).resolve()?;
    let residency = hrx::residency::ResidencyManager::new(64 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    let pipeline =
        Pipeline::open_in(files, &context, None, PipelineOptions { fusion_backend: backend })?;
    let mut request = Request::new("a red ceramic cup on a wooden table");
    request.width = size;
    request.height = size;
    request.steps = Some(2);
    request.seed = 37;
    let expected = pipeline.generate(&request, None)?;
    std::fs::write(output, &expected)?;
    let warm_reserved = residency.statistics().reserved_bytes;
    let warm = context.runtime().statistics();
    let mut samples = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        let actual = pipeline.generate(&request, None)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
        anyhow::ensure!(actual == expected, "generation replay changed");
        anyhow::ensure!(
            residency.statistics().reserved_bytes == warm_reserved,
            "warm residency grew"
        );
    }
    let after = context.runtime().statistics();
    let chronological = samples.clone();
    samples.sort_by(f64::total_cmp);
    let median = if count.is_multiple_of(2) {
        (samples[count / 2 - 1] + samples[count / 2]) * 0.5
    } else {
        samples[count / 2]
    };
    println!(
        "{}",
        serde_json::json!({"scope":"warm 2-step full generation", "width":size, "height":size,
            "backend":format!("{backend:?}"), "selection":pipeline.fusion_selection(),
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
