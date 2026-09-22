//! Warm production generation, including text encoding, denoising and VAE.
use krea2::pipeline::{Files, Pipeline, PipelineOptions, Request};
use std::{path::Path, time::Instant};
fn main() -> anyhow::Result<()> {
    let output = std::env::args().nth(1).expect("output RGB path");
    let files = Files::of(Path::new("krea2_turbo_int8_convrot")).offline(true).resolve()?;
    let residency = hrx::residency::ResidencyManager::new(64 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    let pipeline = Pipeline::open_in(
        files,
        &context,
        None,
        PipelineOptions { fusion_backend: krea2::fusion::FusionBackend::Gpu },
    )?;
    let mut request = Request::new("a red ceramic cup on a wooden table");
    request.width = 256;
    request.height = 256;
    request.steps = Some(2);
    request.seed = 37;
    let expected = pipeline.generate(&request, None)?;
    std::fs::write(output, &expected)?;
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        let actual = pipeline.generate(&request, None)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
        anyhow::ensure!(actual == expected, "generation replay changed");
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"scope":"warm 256x256, 2-step full generation", "median_ms":samples[3], "samples_ms":samples})
    );
    Ok(())
}
