//! CPU chunk blending and complete streamed checkpoint averaging, without GPU setup.
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krea2::training::{
    TrainConfig,
    full::artifacts::{self, Averaging, Digest, Kind, Manifest},
};
use std::{
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

fn fixture(root: &Path, step: usize, count: usize) -> PathBuf {
    let path = root.join(format!("step-{step}"));
    std::fs::create_dir(&path).unwrap();
    let mut header = serde_json::to_vec(&serde_json::json!({
        "a":{"dtype":"BF16","shape":[count],"data_offsets":[0,count*2]},
        "b":{"dtype":"F32","shape":[count],"data_offsets":[count*2,count*6]}
    }))
    .unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    for _ in 0..count {
        bytes.extend_from_slice(&half::bf16::from_f32(step as f32).to_bits().to_le_bytes());
    }
    for _ in 0..count {
        bytes.extend_from_slice(&(step as f32).to_le_bytes());
    }
    let model =
        Digest { bytes: bytes.len() as u64, blake3: blake3::hash(&bytes).to_hex().to_string() };
    let mut file = std::fs::File::create_new(path.join("model.safetensors")).unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    let manifest = Manifest {
        version: 1,
        kind: Kind::Model,
        run_id: "criterion".into(),
        training_fingerprint: "criterion".into(),
        software: "criterion".into(),
        config: TrainConfig::full_preset(),
        step,
        pinned: false,
        model,
        sources: Vec::new(),
        averaging: None,
    };
    std::fs::write(path.join("artifact.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    path
}

fn checkpoint(c: &mut Criterion) {
    let count = 1 << 20;
    let mut group = c.benchmark_group("checkpoint/blend");
    for bf16 in [false, true] {
        let bytes: Vec<_> = (0..count)
            .flat_map(|i| {
                let x = (i % 127) as f32 / 128.0;
                if bf16 {
                    half::bf16::from_f32(x).to_bits().to_le_bytes().to_vec()
                } else {
                    x.to_le_bytes().to_vec()
                }
            })
            .collect();
        let mut acc = vec![0.0; count];
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(if bf16 { "bf16" } else { "f32" }, |b| {
            b.iter(|| {
                artifacts::blend_chunk(
                    black_box(&mut acc),
                    black_box(&bytes),
                    bf16,
                    black_box(0.25),
                )
                .unwrap();
            })
        });
    }
    group.finish();

    // Use KREA2_CHECKPOINT_BENCH_DIR to measure a particular output filesystem.
    // Sources stay in the OS cache. Publication/fsync/hash/encoding are timed;
    // fixture construction and deletion are outside the measured interval.
    let directory = std::env::var_os("KREA2_CHECKPOINT_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::tempdir_in(directory).unwrap();
    let inputs = [fixture(dir.path(), 1, count), fixture(dir.path(), 2, count)];
    let output = dir.path().join("average");
    let mut group = c.benchmark_group("checkpoint/stream");
    group.sample_size(10).measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes((count * 6 * 2) as u64));
    group.bench_function("two_mixed_snapshots_warm_cache", |b| {
        b.iter_custom(|n| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..n {
                let start = Instant::now();
                black_box(artifacts::average(&inputs, &output, Averaging::Uniform).unwrap());
                elapsed += start.elapsed();
                std::fs::remove_dir_all(&output).unwrap();
            }
            elapsed
        })
    });
    group.finish();
}
criterion_group!(benches, checkpoint);
criterion_main!(benches);
