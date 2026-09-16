//! Compilation and fresh-process stage measurements for scripts/qualify-fusion.py.
use krea2::fusion::{
    npu,
    qualification::{self as q, Case, Record, Samples},
};
use std::{fs, path::Path, sync::Arc, time::Instant};
fn bf16(bytes: &[u8], index: usize) -> f64 {
    let bits = u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
    f32::from_bits(u32::from(bits) << 16) as f64
}

// Sample across the entire result using an independent scalar f64 dot product.
// Small synthetic cases check every element, including ragged M and bias.
fn check_oracle(
    case: &Case,
    input: &[u8],
    weights: &[u8],
    bias: &[u8],
    output: &[u8],
) -> anyhow::Result<()> {
    let shape = &case.shape;
    let count = shape.m * shape.n;
    let stride = if count * shape.k <= 1_000_000 { 1 } else { count.div_ceil(128) };
    let mut squared = 0.0;
    let mut norm = 0.0;
    for index in (0..count).step_by(stride) {
        let (row, col) = (index / shape.n, index % shape.n);
        let dot: f64 = (0..shape.k)
            .map(|k| bf16(input, row * shape.k + k) * bf16(weights, col * shape.k + k))
            .sum();
        let expected = dot + if shape.bias { bf16(bias, col) } else { 0.0 };
        let actual = bf16(output, index);
        anyhow::ensure!(actual.is_finite() && expected.is_finite(), "nonfinite oracle result");
        squared += (actual - expected).powi(2);
        norm += expected.powi(2);
    }
    let error = (squared / norm.max(f64::MIN_POSITIVE)).sqrt();
    anyhow::ensure!(error <= 0.01, "independent f64 oracle relative RMS: {error}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: fusion_worker compile CASE TOOLCHAIN | measure CASE RECORD OUTPUT"
    );
    if args[1] == "status" {
        let record: Record = serde_json::from_slice(&fs::read(&args[2])?)?;
        record.verify(
            &record.case.shape,
            &record.case.checkpoint,
            &record.case.weights,
            &record.case.bias,
        )?;
        println!(
            "{}",
            if record.qualified() {
                "NPU qualified"
            } else {
                "GPU selected: qualification did not establish a win"
            }
        );
        return Ok(());
    }
    let directory = Path::new(&args[2]);
    let case = Case::read(directory)?;
    if args[1] == "compile" {
        anyhow::ensure!(args.len() == 4, "compile CASE TOOLCHAIN");
        let record = npu::compile(Path::new(&args[3]), case)?;
        let path = q::directory()?.join(format!("{}.json", record.case.shape.name()));
        q::save(&path, &record)?;
        println!("{}", path.display());
        return Ok(());
    }
    anyhow::ensure!(args[1] == "measure" && args.len() == 5, "measure CASE RECORD OUTPUT");
    let record: Record = serde_json::from_slice(&fs::read(&args[3])?)?;
    let weights = fs::read(directory.join("weights.bin"))?;
    let biases = fs::read(directory.join("bias.bin"))?;
    let input = fs::read(directory.join("input.bin"))?;
    record.verify(&case.shape, &case.checkpoint, &case.weights, &case.bias)?;
    anyhow::ensure!(
        record.case.input == case.input,
        "captured input differs from the compiled case"
    );
    // The explicit qualification command supplies artifacts from its own compile step.
    let mut stream = hrx::Stream::open()?;
    let pilot = unsafe { npu::Pilot::load(&record, &weights, &biases, &stream) }?;
    let ops = krea2::ops::Ops::new(hrx::BufferPool::new());
    let x = ops.tensor(&stream, case.shape.m, case.shape.k)?;
    stream.upload(x.binding()?, &input)?;
    let storage = Arc::new(stream.allocate(weights.len())?);
    stream.upload(storage.binding(), &weights)?;
    let w = krea2::ops::Weight::new(
        &storage,
        0,
        vec![case.shape.n, case.shape.k],
        case.shape.n * case.shape.k,
        None,
    );
    let bias = stream.allocate(biases.len().max(2))?;
    stream.upload(bias.binding(), &biases)?;
    let b = case.shape.bias.then(|| bias.binding());
    let y = ops.tensor(&stream, case.shape.m, case.shape.n)?;
    for _ in 0..10 {
        let _ = ops.linear(&stream, &x, &w, b)?;
        stream.synchronize()?;
        pilot.execute(&mut stream, x.binding()?, y.binding()?)?;
    }
    let mut checked = vec![0; case.shape.m * case.shape.n * 2];
    // Mutate inputs between graph replays to catch stale cache visibility and
    // incomplete writes; restore the captured activation before timing.
    for variant in 0..3 {
        let changed: Vec<u8> = input
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|bytes| {
                let mut value = u16::from_le_bytes(*bytes);
                if variant == 1 {
                    value ^= 0x8000;
                }
                if variant == 2 {
                    value = 0;
                }
                value.to_le_bytes()
            })
            .collect();
        stream.upload(x.binding()?, &changed)?;
        pilot.execute(&mut stream, x.binding()?, y.binding()?)?;
        stream.read_blocking(y.binding()?, &mut checked)?;
        check_oracle(&case, &changed, &weights, &biases, &checked)?;
    }
    stream.upload(x.binding()?, &input)?;
    stream.synchronize()?;
    let before = pilot.statistics();
    let mut samples = Samples { gpu: vec![], npu: vec![], correct: true };
    for round in 0..100 {
        for backend in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let start = Instant::now();
            if backend == 0 {
                let _ = ops.linear(&stream, &x, &w, b)?;
                stream.synchronize()?;
                samples.gpu.push(start.elapsed().as_secs_f64());
            } else {
                let timed_output = ops.tensor(&stream, case.shape.m, case.shape.n)?;
                pilot.execute(&mut stream, x.binding()?, timed_output.binding()?)?;
                samples.npu.push(start.elapsed().as_secs_f64());
            }
        }
    }
    pilot.execute(&mut stream, x.binding()?, y.binding()?)?;
    let reference = ops.linear(&stream, &x, &w, b)?;
    let mut gpu = vec![0; case.shape.m * case.shape.n * 2];
    let mut output = gpu.clone();
    stream.read_blocking(reference.binding()?, &mut gpu)?;
    stream.read_blocking(y.binding()?, &mut output)?;
    let mut squared = 0f64;
    let mut norm = 0f64;
    for (a, b) in gpu.as_chunks::<2>().0.iter().zip(output.as_chunks::<2>().0.iter()) {
        let a = f32::from_bits(u32::from(u16::from_le_bytes(*a)) << 16) as f64;
        let b = f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16) as f64;
        if !a.is_finite() || !b.is_finite() {
            samples.correct = false;
        }
        squared += (a - b) * (a - b);
        norm += a * a;
    }
    samples.correct &= (squared / norm.max(f64::MIN_POSITIVE)).sqrt() <= 0.01;
    let after = pilot.statistics();
    anyhow::ensure!(
        before.allocations == after.allocations && before.imports == after.imports,
        "pilot allocated during replay"
    );
    eprintln!("resident={} peak={} replay_allocations={} replay_imports={} handoff_copy_bytes={} cache_maintenance_bytes={}",
        pilot.bytes, after.peak_bytes, after.allocations - before.allocations,
        after.imports - before.imports, (case.shape.m * (case.shape.k + case.shape.n) * 2) * 101,
        after.cache_maintenance_bytes - before.cache_maintenance_bytes);
    let mut gpu = samples.gpu.clone();
    gpu.sort_by(f64::total_cmp);
    let mut npu = samples.npu.clone();
    npu.sort_by(f64::total_cmp);
    eprintln!(
        "completed stage: GPU median={:.3}ms p95={:.3}ms; NPU median={:.3}ms p95={:.3}ms",
        gpu[49] * 1e3,
        gpu[94] * 1e3,
        npu[49] * 1e3,
        npu[94] * 1e3
    );
    q::save(Path::new(&args[4]), &samples)?;
    anyhow::ensure!(samples.correct, "NPU output differs from GPU beyond 1% relative RMS");
    Ok(())
}
