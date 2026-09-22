//! Paired production-pipeline fixture capture and verification. Record with the
//! pre-migration implementation, then compare without overwriting the fixture.
use anyhow::{ensure, Context, Result};
use krea2::pipeline::{Files, Pipeline, PipelineOptions, Request};
use std::{io::Write, path::Path, time::Instant};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    ensure!(
        args.len() == 3 && matches!(args[1].as_str(), "record" | "compare"),
        "usage: qualify_pipeline record|compare DIRECTORY"
    );
    let directory = Path::new(&args[2]);
    let record = args[1] == "record";
    let check = |name: &str, bytes: &[u8]| -> Result<()> {
        let path = directory.join(name);
        if record {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?
                .write_all(bytes)?;
        } else {
            let expected = std::fs::read(&path).with_context(|| path.display().to_string())?;
            if expected != bytes {
                std::fs::write(path.with_extension("actual"), bytes)?;
            }
            ensure!(
                expected == bytes,
                "{} differs from the pre-migration pipeline",
                path.display()
            );
        }
        eprintln!(
            "{} {name}: {} bytes",
            if record { "recorded" } else { "matched" },
            bytes.len()
        );
        Ok(())
    };
    let start = Instant::now();
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
    eprintln!("opened pipeline in {:.3}s", start.elapsed().as_secs_f64());
    for (i, (width, height, tokens)) in
        [(64, 64, 3), (128, 64, 3), (64, 128, 3), (64, 64, 5), (64, 64, 3)]
            .into_iter()
            .enumerate()
    {
        let text =
            (0..tokens * 12 * 2560).map(|i| ((i % 97) as f32 - 48.) / 97.).collect::<Vec<_>>();
        let latents = (0..(width / 16) * (height / 16) * 64)
            .map(|i| ((i % 61) as f32 - 30.) / 31.)
            .collect::<Vec<_>>();
        let start = Instant::now();
        let velocity = pipeline.transformer(&text, tokens, &latents, width, height, 0.75)?;
        ensure!(velocity.iter().all(|v| v.is_finite()), "nonfinite velocity");
        check(
            &format!("transformer-{i}-{width}x{height}-t{tokens}.f32"),
            bytemuck::cast_slice(&velocity),
        )?;
        eprintln!("transformer case {i}: {:.3}s", start.elapsed().as_secs_f64());
    }
    let mut request = Request::new("a red ceramic cup on a wooden table");
    request.width = 64;
    request.height = 64;
    request.steps = Some(2);
    request.seed = 37;
    let rgb = pipeline.generate(&request, None)?;
    check("generated.rgb", &rgb)?;
    let warm = pipeline.context().runtime().statistics();
    let warm_reserved = residency.statistics().reserved_bytes;
    for _ in 0..3 {
        ensure!(pipeline.generate(&request, None)? == rgb, "generation replay changed");
    }
    let replayed = pipeline.context().runtime().statistics();
    ensure!(residency.statistics().reserved_bytes == warm_reserved, "warm native storage grew");
    ensure!(warm.allocations == replayed.allocations, "warm tracked allocations grew");
    ensure!(warm.live_bytes == replayed.live_bytes, "warm tracked memory grew");
    ensure!(
        warm.native_graphs_prepared == replayed.native_graphs_prepared
            && warm.copy_streams_created == replayed.copy_streams_created,
        "warm native graphs or copy streams grew"
    );
    let cancelled = pipeline.generate(&request, Some(&mut |_, _, _| false));
    ensure!(cancelled.is_err_and(|e| e.0 == "cancelled"), "cancellation not observed");
    ensure!(pipeline.generate(&request, None)? == rgb, "retry after cancellation changed");
    eprintln!(
        "warm tracked bytes: {}; allocations: {}; cancellation/retry matched",
        replayed.live_bytes, replayed.allocations
    );
    eprintln!("production pipeline qualification completed");
    eprintln!("warm native + tracked reserved bytes: {warm_reserved}");
    drop(pipeline);
    ensure!(residency.statistics().reserved_bytes == 0, "pipeline retained budget after drop");
    Ok(())
}
