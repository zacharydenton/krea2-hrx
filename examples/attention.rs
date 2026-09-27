//! Compare attention sources on identical inputs with interleaved GPU-clock
//! timing against a register-only fp16 WMMA peak from the same rounds,
//! retaining each kernel's code object, compiler report and diagnostics.
//!
//! ```sh
//! cargo run --release --example attention -- TOKENS OUT_DIR [--rounds N] [--tolerance T] SOURCE...
//! ```
//!
//! Sources use the production ABI of `kernels/attention_gqa_lds_f16_wmma.loom`,
//! V transposed to `[kv_heads * 128][capacity]`, and are also reported with the
//! cost of the `sage_transpose` producing it. A `natural:` prefix binds V as
//! `[capacity][kv_heads * 128]` instead, for kernels that stage it themselves.
//! The first is the reference: others must match it byte for byte, or, with
//! `--tolerance`, stay within that absolute difference, for candidates that
//! legitimately reorder floating-point work. Device timestamps need a native
//! bridge with HRX's profiling exports, as in hrx-rs 0.8.10's bundle; see
//! `docs/graph-recording.md`.
use anyhow::{Context, Result, bail, ensure};
use half::f16;
use hrx::{Buffer, Constants, Stream};
use std::path::PathBuf;

const HEADS: usize = 48;
const KV_HEADS: usize = 12;
const DIM: usize = 128;
const REPEATS: usize = 5;
const PEAK_ITERATIONS: usize = 4096;
const PEAK_WORKGROUPS: usize = 40 * 8;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tokens: usize = args.next().context("expected TOKENS OUT_DIR SOURCE...")?.parse()?;
    let out = PathBuf::from(args.next().context("expected an output directory")?);
    let (mut rounds, mut tolerance, mut sources) = (11usize, None::<f32>, Vec::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rounds" => rounds = args.next().context("--rounds needs a count")?.parse()?,
            "--tolerance" => {
                tolerance = Some(args.next().context("--tolerance needs a value")?.parse()?)
            }
            flag if flag.starts_with("--") => bail!("unknown option {flag}"),
            _ => match arg.strip_prefix("natural:") {
                Some(path) => sources.push((PathBuf::from(path), false)),
                None => sources.push((PathBuf::from(arg), true)),
            },
        }
    }
    ensure!(
        !sources.is_empty() && (16..=16896).contains(&tokens),
        "expected TOKENS OUT_DIR SOURCE..."
    );
    std::fs::create_dir_all(&out)?;
    let compiler = krea2::kernels::compiler(None)?;
    let mut stream = Stream::open()?;
    let capacity = (tokens + 16).div_ceil(64) * 64;
    let stem = "attention_gqa_lds_f16_wmma";
    let mut spec = hrx::loom::Specialization::new(format!("krea2_{stem}"));
    for (key, value) in [
        ("q_stride", HEADS * DIM),
        ("kv_stride", KV_HEADS * DIM),
        ("out_stride", HEADS * DIM),
        ("tokens", tokens),
        ("token_capacity", capacity),
    ] {
        spec.set_config(format!("krea2.{stem}.{key}"), value.to_string());
    }
    spec.set_config(format!("krea2.{stem}.scale"), "0.08838834764831845");
    spec.set_report(hrx::loom::ReportMode::Summary);
    let mut loaded = Vec::new();
    for (i, (path, _)) in sources.iter().enumerate() {
        let artifact = compiler.module(&std::fs::read_to_string(path)?).compile(&spec)?;
        std::fs::write(out.join(format!("{i}.hsaco")), artifact.bytes())?;
        let mut resources = String::new();
        if let Some(report) = artifact.report() {
            std::fs::write(out.join(format!("{i}.json")), report.json().to_string())?;
            let t = &report.json()["target_resources"];
            resources = format!(
                "occupancy {}%, {} VGPRs, limited by {}",
                t["occupancy_percent"],
                t["vector"]["final"]["register_count"],
                t["limiting_resource"].as_str().unwrap_or("?")
            );
        }
        let diagnostics: Vec<&str> =
            artifact.diagnostics().iter().map(|d| d.message.as_str()).collect();
        std::fs::write(out.join(format!("{i}.diagnostics")), diagnostics.join("\n"))?;
        let spills = artifact.diagnostics().iter().filter(|d| d.code == "BACKEND/009").count();
        println!("{i}: {}: {resources}, {spills} spills", path.display());
        // SAFETY: caller-selected Loom sources are trusted native code with the
        // production attention ABI and the specialization declared above.
        let kernel = unsafe { stream.load_artifact(&artifact)? };
        let constants = Constants::indices(&kernel, &[tokens as u32, 0])?;
        loaded.push((kernel, constants));
    }

    let mut buffers = Vec::new();
    let mut random = 0x12345678u32;
    let mut v = Vec::new();
    for (stride, scale) in [(HEADS * DIM, 0.5), (KV_HEADS * DIM, 0.3), (KV_HEADS * DIM, 0.7)] {
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
        v = data;
    }
    let stride = KV_HEADS * DIM;
    let transposed: Vec<f16> =
        (0..stride * capacity).map(|i| v[(i % capacity) * stride + i / capacity]).collect();
    let v_transposed = stream.allocate(transposed.len() * 2)?;
    stream.upload(v_transposed.binding(), bytemuck::cast_slice(&transposed))?;
    buffers.push(stream.allocate(tokens * HEADS * DIM * 2)?);
    let natural: Vec<_> = buffers.iter().map(Buffer::binding).collect();
    let mut swapped = natural.clone();
    swapped[2] = v_transposed.binding();
    let bindings = |(_, vt): &(PathBuf, bool)| if *vt { &swapped[..] } else { &natural[..] };
    let grid = [tokens.div_ceil(16) as u32, KV_HEADS as u32, 1];

    let mut baseline: Option<Vec<f16>> = None;
    for (i, ((kernel, constants), source)) in loaded.iter().zip(&sources).enumerate() {
        // SAFETY: bindings are sized and initialized for all valid rows and zero
        // headroom, and each workgroup owns sixteen query rows and four heads.
        unsafe { stream.dispatch(kernel, grid, [128, 1, 1], constants, bindings(source))? };
        let bytes = stream.read(natural[3])?.wait(&mut stream)?;
        let values: Vec<f16> = bytemuck::cast_slice(&bytes).to_vec();
        let Some(reference) = &baseline else {
            baseline = Some(values);
            continue;
        };
        let differing =
            values.iter().zip(reference).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let worst = values
            .iter()
            .zip(reference)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{i}: differing elements {differing}/{}, max |difference| {worst:e}",
            values.len()
        );
        match tolerance {
            None => ensure!(differing == 0, "candidate {i} changed attention output"),
            Some(limit) => ensure!(worst <= limit, "candidate {i} exceeds tolerance {limit}"),
        }
    }

    let peak_source = include_str!("../experiments/peak_f16.loom");
    let mut peak_spec = hrx::loom::Specialization::new("krea2_peak_f16");
    peak_spec.set_config("krea2.peak_f16.iterations", PEAK_ITERATIONS.to_string());
    let peak_artifact = compiler.module(peak_source).compile(&peak_spec)?;
    // SAFETY: the peak kernel writes 128 x 16 floats into `peak_out`.
    let peak = unsafe { stream.load_artifact(&peak_artifact)? };
    let peak_constants = Constants::indices(&peak, &[PEAK_WORKGROUPS as u32])?;
    let peak_out = stream.allocate(128 * 16 * 4)?;
    let peak_bindings = [peak_out.binding()];

    let mut transpose_spec = hrx::loom::Specialization::new("krea2_sage_transpose");
    transpose_spec.set_config("krea2.sage_transpose.width", (KV_HEADS * DIM).to_string());
    transpose_spec.set_config("krea2.sage_transpose.row_capacity", capacity.to_string());
    let transpose_artifact = compiler
        .module(include_str!("../kernels/native/sage_transpose.loom"))
        .compile(&transpose_spec)?;
    // SAFETY: the transpose reads `tokens` rows of V and writes V^T, both sized above.
    let transpose = unsafe { stream.load_artifact(&transpose_artifact)? };
    let transpose_constants = Constants::indices(&transpose, &[tokens as u32])?;
    let transpose_bindings = [natural[2], swapped[2]];
    let transpose_grid = [tokens.div_ceil(32) as u32, (KV_HEADS * DIM / 32) as u32, 1];

    let arms = loaded
        .iter()
        .zip(&sources)
        .map(|((kernel, constants), source)| {
            (kernel, constants, grid, [128u32, 1, 1], bindings(source))
        })
        .chain([
            (
                &peak,
                &peak_constants,
                [PEAK_WORKGROUPS as u32, 1, 1],
                [256, 1, 1],
                &peak_bindings[..],
            ),
            (
                &transpose,
                &transpose_constants,
                transpose_grid,
                [256, 1, 1],
                &transpose_bindings[..],
            ),
        ]);
    let mut graphs = Vec::new();
    for (kernel, constants, grid, block, bindings) in arms {
        let mut graph = stream.graph()?;
        let mut last = None;
        for _ in 0..REPEATS {
            let after: Vec<hrx::Node> = last.into_iter().collect();
            // SAFETY: as the eager dispatches above.
            last = Some(unsafe {
                graph.dispatch(&after, kernel, grid, block, constants, bindings)
            }?);
        }
        let labels: Vec<String> = (0..REPEATS).map(|i| i.to_string()).collect();
        graphs.push(
            graph.finish_profiled(&labels).context(
                "device timestamps need a native bridge with HRX's profiling exports",
            )?,
        );
    }
    let mut samples = vec![Vec::new(); graphs.len()];
    for round in 0..rounds + 2 {
        for offset in 0..graphs.len() {
            let arm = (round + offset) % graphs.len();
            let profile = stream.launch_profiled(&mut graphs[arm])?;
            if round >= 2 {
                let ms = 1e3 / profile.frequency_hz as f64;
                samples[arm].extend(
                    profile.intervals.iter().map(|i| (i.end_tick - i.start_tick) as f64 * ms),
                );
            }
        }
    }
    let medians: Vec<f64> = samples
        .into_iter()
        .map(|mut s| {
            s.sort_by(f64::total_cmp);
            s[s.len() / 2]
        })
        .collect();
    let peak_flops = (PEAK_WORKGROUPS * 8 * PEAK_ITERATIONS * 8 * 16 * 16 * 16 * 2) as f64;
    let peak_tflops = peak_flops / medians[loaded.len()] / 1e9;
    let transposing = medians[loaded.len() + 1];
    println!(
        "register-only fp16 peak: {peak_tflops:.1} TFLOP/s; V transpose {transposing:.3} ms"
    );
    // QK^T and PV, each 2 * tokens^2 * dim per query head.
    let flops = (4 * tokens * tokens * DIM * HEADS) as f64;
    let total = |i: usize| medians[i] + if sources[i].1 { transposing } else { 0.0 };
    for (i, median) in medians[..loaded.len()].iter().enumerate() {
        let tflops = flops / median / 1e9;
        println!(
            "{i}: tokens={tokens} {median:.3} ms  {tflops:.1} TFLOP/s  {:.0}% of peak  x{:.3}{}",
            100.0 * tflops / peak_tflops,
            medians[0] / median,
            if sources[i].1 {
                format!(", x{:.3} with the transpose", total(0) / total(i))
            } else {
                String::new()
            }
        );
    }
    Ok(())
}
