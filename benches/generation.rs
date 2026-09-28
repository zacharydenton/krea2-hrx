//! Warm, complete Turbo generation; requires cached model weights and gfx1151.
use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use krea2::pipeline::{Files, Pipeline, Request};
use std::{hint::black_box, path::Path, time::Duration};

fn generation(c: &mut Criterion) {
    assert!(!krea2::kernels::native_profile(), "disable profiling for benchmarks");
    let mut pipeline = None;
    let mut group = c.benchmark_group("generation");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));
    for (width, height) in [(256, 256), (1024, 1024), (1536, 2048)] {
        let mut warmed = false;
        group.bench_function(format!("{width}x{height}"), |b| {
            let pipeline = pipeline.get_or_insert_with(|| {
                let files = Files::of(Path::new("krea2_turbo_int8_convrot"))
                    .offline(true)
                    .resolve()
                    .expect("cache the Turbo, BF16 text encoder and VAE weights");
                Pipeline::open(files, None).expect("open pipeline")
            });
            let mut request = Request::new("a red ceramic cup on a wooden table");
            request.width = width;
            request.height = height;
            request.steps = Some(8);
            request.guidance = Some(0.0);
            request.seed = 37;
            if !warmed {
                black_box(pipeline.generate(&request, None).expect("warm generation"));
                warmed = true;
            }
            b.iter(|| black_box(pipeline.generate(&request, None).expect("generation")));
        });
    }
    group.finish();
}

criterion_group!(benches, generation);
criterion_main!(benches);
