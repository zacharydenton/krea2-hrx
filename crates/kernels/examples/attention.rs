//! Compare attention sources on identical inputs, retaining compiler reports and code.
//! cargo run --release -p krea2-kernels --example attention -- TOKENS OUT_DIR SOURCE...
use half::f16;
use hrx::{Buffer, Constants, Stream};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let tokens: usize = args.next().ok_or("expected TOKENS OUT_DIR SOURCE...")?.parse()?;
    let out = PathBuf::from(args.next().ok_or("expected output directory")?);
    let sources: Vec<_> = args.map(PathBuf::from).collect();
    assert!(!sources.is_empty() && (16..=16896).contains(&tokens));
    std::fs::create_dir_all(&out)?;
    let compiler = kernels::compiler(None)?;
    let mut stream = Stream::open()?;
    let capacity = (tokens + 16).div_ceil(64) * 64;
    let stem = "attention_gqa_lds_f16_wmma";
    let mut spec = hrx::loom::Specialization::new(format!("krea2_{stem}"));
    for (key, value) in [
        ("q_stride", 6144),
        ("kv_stride", 1536),
        ("out_stride", 6144),
        ("tokens", tokens),
        ("token_capacity", capacity),
    ] {
        spec.config.insert(format!("krea2.{stem}.{key}"), value.to_string());
    }
    spec.config.insert(format!("krea2.{stem}.scale"), "0.08838834764831845".into());
    spec.report = true;
    let mut loaded = Vec::new();
    for (i, path) in sources.iter().enumerate() {
        let artifact = compiler
            .module(&std::fs::read_to_string(path)?)
            .compile(&spec, &kernels::cache_root()?)?;
        std::fs::write(out.join(format!("{i}.hsaco")), artifact.bytes())?;
        if let Some(report) = artifact.report() {
            std::fs::write(out.join(format!("{i}.json")), report.to_string())?;
        }
        let diagnostics = artifact
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(out.join(format!("{i}.diagnostics")), &diagnostics)?;
        println!(
            "{i}: {} compiler={} spills={}",
            path.display(),
            compiler.identity(),
            artifact.diagnostics().iter().filter(|d| d.code == "BACKEND/009").count()
        );
        // Safety: caller-selected Loom sources are trusted native code, with the
        // production attention ABI and the exact specialization declared above.
        let kernel = unsafe { stream.load_artifact(&artifact)? };
        let constants = Constants::indices(&kernel, &[tokens as u32, 0])?;
        loaded.push((kernel, constants));
    }
    let mut buffers = Vec::new();
    let mut random = 0x12345678u32;
    for (stride, scale) in [(6144, 0.5), (1536, 0.3), (1536, 0.7)] {
        let mut data = vec![f16::ZERO; capacity * stride];
        for value in &mut data[..tokens * stride] {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            *value = f16::from_f32((random as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        let buffer = stream.allocate(data.len() * 2)?;
        stream.upload(buffer.binding(), bytemuck::cast_slice(&data))?;
        buffers.push(buffer);
    }
    buffers.push(stream.allocate(tokens * 6144 * 2)?);
    let bindings: Vec<_> = buffers.iter().map(Buffer::binding).collect();
    let grid = [tokens.div_ceil(16) as u32, 12, 1];
    let mut baseline: Option<Vec<u8>> = None;
    for (i, (kernel, constants)) in loaded.iter().enumerate() {
        // Safety: bindings are sized and initialized for all valid rows and zero
        // headroom, and each workgroup owns sixteen query rows and four heads.
        unsafe {
            stream.dispatch(kernel, grid, [128, 1, 1], constants, &bindings)?;
        }
        let bytes = stream.read_queued(bindings[3])?.wait(&mut stream)?;
        if let Some(ref original) = baseline {
            let different = bytes
                .chunks_exact(2)
                .zip(original.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            println!("{i}: differing output elements {different}/{}", tokens * 6144);
            assert!(&bytes == original, "candidate changed attention output");
        } else {
            baseline = Some(bytes);
        }
    }
    const BATCH: usize = 10;
    let mut samples = vec![Vec::new(); loaded.len()];
    for round in 0..8 {
        // Rotate the order to reduce clock/thermal drift between candidates.
        for offset in 0..loaded.len() {
            let i = (round + offset) % loaded.len();
            let (kernel, constants) = &loaded[i];
            stream.synchronize()?;
            let began = Instant::now();
            for _ in 0..BATCH {
                unsafe {
                    stream.dispatch(kernel, grid, [128, 1, 1], constants, &bindings)?;
                }
            }
            stream.synchronize()?;
            if round > 0 {
                samples[i].push(began.elapsed().as_secs_f64() * 1000. / BATCH as f64);
            }
        }
    }
    for (i, sample) in samples.iter_mut().enumerate() {
        sample.sort_by(f64::total_cmp);
        println!(
            "{i}: tokens={tokens} median_ms={:.6} samples_ms={sample:?}",
            sample[sample.len() / 2]
        );
    }
    Ok(())
}
