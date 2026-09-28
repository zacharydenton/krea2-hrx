//! Interleaved GPU-clock comparison of plain INT8 preparation sources.
//! Usage: prepare_ab TOKENS WIDTH ROUNDS BASELINE.loom CANDIDATE.loom [...]
//! Checks every output byte, including scales and untouched row padding.
use anyhow::{Context, Result, ensure};
use hrx::{Constants, Stream};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tokens: usize = args.next().context("tokens")?.parse()?;
    let width: usize = args.next().context("width")?.parse()?;
    let rounds: usize = args.next().context("rounds")?.parse()?;
    let paths: Vec<_> = args.collect();
    ensure!(tokens > 0 && tokens <= 16896, "tokens must be 1..=16896");
    ensure!(
        (2048..=32768).contains(&width) && width.is_multiple_of(2048),
        "width must be a multiple of 2048 in 2048..=32768"
    );
    ensure!(rounds > 0 && paths.len() >= 2, "need rounds and at least two sources");
    let stride = krea2::kernels::shape::gemm_pitch(width);
    let mut stream = Stream::open()?;
    let compiler = krea2::kernels::compiler(None)?;
    let input: Vec<u16> = (0..tokens * width)
        .map(|i| {
            // Include zero rows, signs, subnormals and a wide finite exponent range.
            if (i / width).is_multiple_of(17) {
                0
            } else {
                let x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
                ((x >> 32) as u16 & 0x83ff) | (((x >> 48) % 24) as u16 * 1024)
            }
        })
        .collect();
    let h = stream.allocate(input.len() * 2)?;
    stream.upload(h.binding(), bytemuck::cast_slice(&input))?;
    let q = stream.allocate(tokens * stride)?;
    let scale = stream.allocate(tokens * 4)?;
    let bindings = [h.binding(), q.binding(), scale.binding()];
    let mut graphs = Vec::new();
    let mut artifacts = Vec::new();
    let mut reference = None;
    const REPEATS: usize = 5;
    for path in &paths {
        let source = std::fs::read_to_string(path)?;
        let mut request = hrx::loom::Specialization::new("krea2_prepare_plain_i8");
        request.set_config("krea2.prepare_plain_i8.width", width.to_string());
        request.set_config("krea2.prepare_plain_i8.out_stride", stride.to_string());
        request.set_report(hrx::loom::ReportMode::Summary);
        let artifact = compiler.module(&source).compile(&request)?;
        artifacts.push(artifact.path().display().to_string());
        // SAFETY: every binding is sized from the specialization above.
        let kernel = unsafe { stream.load_artifact(&artifact)? };
        let constants = Constants::indices(&kernel, &[tokens as u32])?;
        stream.fill(q.binding(), 0xa5)?;
        stream.fill(scale.binding(), 0)?;
        // SAFETY: the kernel serves one row per 256-thread workgroup.
        unsafe {
            stream.dispatch(
                &kernel,
                [tokens as u32, 1, 1],
                [256, 1, 1],
                &constants,
                &bindings,
            )?;
        }
        let mut bytes = vec![0; tokens * stride];
        let mut scales = vec![0; tokens * 4];
        stream.read_blocking(q.binding(), &mut bytes)?;
        stream.read_blocking(scale.binding(), &mut scales)?;
        for row in bytes.chunks_exact(stride) {
            ensure!(row[width..].iter().all(|&v| v == 0xa5), "{path}: padding overwritten");
        }
        let output = (bytes, scales);
        if let Some(expected) = &reference {
            ensure!(expected == &output, "{path}: output differs from {}", paths[0]);
        } else {
            reference = Some(output);
        }
        let mut graph = stream.graph()?;
        let mut last = None;
        for _ in 0..REPEATS {
            let after: Vec<_> = last.into_iter().collect();
            // SAFETY: same dimensions and live buffers as the checked dispatch.
            last = Some(unsafe {
                graph.dispatch(
                    &after,
                    &kernel,
                    [tokens as u32, 1, 1],
                    [256, 1, 1],
                    &constants,
                    &bindings,
                )
            }?);
        }
        graphs.push(
            graph.finish_profiled(&(0..REPEATS).map(|i| i.to_string()).collect::<Vec<_>>())?,
        );
    }
    let mut samples = vec![Vec::new(); paths.len()];
    for round in 0..rounds + 2 {
        for offset in 0..paths.len() {
            let arm = (round + offset) % paths.len();
            let profile = stream.launch_profiled(&mut graphs[arm])?;
            if round >= 2 {
                samples[arm].extend(profile.intervals.iter().map(|i| {
                    (i.end_tick - i.start_tick) as f64 * 1000. / profile.frequency_hz as f64
                }));
            }
        }
    }
    // Verify the same resident buffers after repeated graph execution too.
    // These launches and readbacks are outside the measured samples.
    for (path, graph) in paths.iter().zip(&mut graphs) {
        stream.launch_profiled(graph)?;
        let mut bytes = vec![0; tokens * stride];
        let mut scales = vec![0; tokens * 4];
        stream.read_blocking(q.binding(), &mut bytes)?;
        stream.read_blocking(scale.binding(), &mut scales)?;
        ensure!(reference.as_ref() == Some(&(bytes, scales)), "{path}: replay output differs");
    }
    let results: Vec<_> = paths.iter().zip(samples).zip(artifacts).map(|((path, samples), artifact)| {
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.
        } else { sorted[mid] };
        serde_json::json!({"source":path, "artifact":artifact, "median_ms":median, "samples_ms":samples})
    }).collect();
    println!(
        "{}",
        serde_json::json!({"tokens":tokens, "width":width, "stride":stride,
        "rounds":rounds, "repeats":REPEATS, "outputs_identical":true, "results":results})
    );
    Ok(())
}
