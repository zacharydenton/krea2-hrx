//! Interleaved GPU-clock A/B of int8 GEMM kernel sources, measured against a
//! register-only WMMA peak run in the same rounds, so a shared GPU affects
//! every arm alike.
//!
//! ```sh
//! cargo run --release --example gemm_ab -- kernels/gemm_i8_256.loom candidate.loom
//! cargo run --release --example gemm_ab -- --shape 4115,16384,6144 --rounds 21 a.loom b.loom
//! cargo run --release --example gemm_ab -- --m-group 2 kernels/gemm_i8_256.loom
//! ```
//!
//! Every source must have the plain GEMM ABI of `kernels/gemm_i8_256.loom`
//! (`m_size`; `a`, `w`, `scale`, `a_scale`, `c`) and its `k_size`, `k_stride`,
//! `n_size` and `m_group` configuration; `--m-group` sets the raster group
//! (default 4, production's).
//! A source launching 512-thread workgroups is taken to be the 256x256 tile
//! (`gemm_i8_resid_256x256.loom`'s shape) and gets half as many column tiles. Outputs must match the first source
//! byte for byte unless `--no-check` is given, for diagnostic variants that
//! deliberately skip work. Each kernel's compiled artifact path is printed for
//! `llvm-objdump -d --mcpu=gfx1151`.
//!
//! Device timestamps need a native bridge with HRX's optional profiling
//! exports, as in hrx-rs 0.8.10's bundle; see `docs/graph-recording.md`.
use anyhow::{Context, Result, bail, ensure};
use hrx::{Constants, Kernel, Stream};

/// The four transformer GEMMs at 1024x1024: qkv|gate, gate|up, wo, down.
const SHAPES: [(usize, usize, usize); 4] =
    [(4115, 6144, 15360), (4115, 6144, 32768), (4115, 6144, 6144), (4115, 16384, 6144)];
const REPEATS: usize = 5;
const PEAK_ITERATIONS: usize = 4096;
const PEAK_WORKGROUPS: usize = 40 * 8;

struct Options {
    shapes: Vec<(usize, usize, usize)>,
    rounds: usize,
    m_group: usize,
    check: bool,
    sources: Vec<String>,
}

fn options() -> Result<Options> {
    let mut options = Options {
        shapes: Vec::new(),
        rounds: 11,
        m_group: 4,
        check: true,
        sources: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--shape" => {
                let value = args.next().context("--shape needs M,K,N")?;
                let dims: Vec<usize> =
                    value.split(',').map(str::parse).collect::<Result<_, _>>()?;
                ensure!(dims.len() == 3, "--shape needs M,K,N");
                options.shapes.push((dims[0], dims[1], dims[2]));
            }
            "--rounds" => {
                options.rounds = args.next().context("--rounds needs a count")?.parse()?
            }
            "--m-group" => {
                options.m_group = args.next().context("--m-group needs 1 to 4")?.parse()?
            }
            "--no-check" => options.check = false,
            flag if flag.starts_with("--") => bail!("unknown option {flag}"),
            _ => options.sources.push(arg),
        }
    }
    ensure!(!options.sources.is_empty(), "give at least one kernel source");
    if options.shapes.is_empty() {
        options.shapes = SHAPES.to_vec();
    }
    Ok(options)
}

/// Operand row pitch, as `krea2::kernels::shape::gemm_pitch` lays it out.
fn pitch(k: usize) -> usize {
    krea2::kernels::shape::gemm_pitch(k)
}

fn compile(
    stream: &Stream,
    compiler: &hrx::loom::Compiler,
    source: &str,
    config: &[(&str, usize)],
) -> Result<(Kernel, String)> {
    let symbol = source
        .split("export(\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .context("no export in the source")?
        .to_string();
    let prefix = symbol.trim_start_matches("krea2_");
    let mut request = hrx::loom::Specialization::new(&symbol);
    for (key, value) in config {
        request.set_config(format!("krea2.{prefix}.{key}"), value.to_string());
    }
    request.set_report(hrx::loom::ReportMode::Summary);
    let artifact = compiler.module(source).compile(&request)?;
    let resources = artifact
        .report()
        .map(|report| {
            let t = &report.json()["target_resources"];
            format!(
                "occupancy {}%, {} VGPRs, limited by {}",
                t["occupancy_percent"],
                t["vector"]["final"]["register_count"],
                t["limiting_resource"].as_str().unwrap_or("?")
            )
        })
        .unwrap_or_default();
    let spills = artifact.diagnostics().iter().filter(|d| d.code == "BACKEND/009").count();
    let note = format!("{resources}, {spills} spills, {}", artifact.path().display());
    // SAFETY: the caller binds operands sized from the same configuration.
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    Ok((kernel, note))
}

fn main() -> Result<()> {
    let options = options()?;
    let mut stream = Stream::open()?;
    let compiler = krea2::kernels::compiler(None)?;
    let sources: Vec<String> =
        options.sources.iter().map(std::fs::read_to_string).collect::<Result<_, _>>()?;
    let (peak, _) = compile(
        &stream,
        &compiler,
        include_str!("../experiments/peak_i8.loom"),
        &[("iterations", PEAK_ITERATIONS)],
    )?;
    let peak_constants = Constants::indices(&peak, &[PEAK_WORKGROUPS as u32])?;
    let peak_out = stream.allocate(128 * 16 * 4)?;

    for &(m, k, n) in &options.shapes {
        let stride = pitch(k);
        let bytes = |len: usize, mut s: u64| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s % 255) as i32 - 127) as i8 as u8
                })
                .collect()
        };
        let upload = |stream: &mut Stream, data: &[u8]| -> Result<hrx::Buffer> {
            let buffer = stream.allocate(data.len())?;
            stream.upload(buffer.binding(), data)?;
            Ok(buffer)
        };
        let a = upload(&mut stream, &bytes(m * stride, 1))?;
        let w = upload(&mut stream, &bytes(n * stride, 2))?;
        let scale = upload(&mut stream, bytemuck::cast_slice(&vec![1e-3f32; n]))?;
        let a_scale = upload(&mut stream, bytemuck::cast_slice(&vec![1e-2f32; m]))?;
        let c = stream.allocate(m * n * 2)?;
        let bindings =
            [a.binding(), w.binding(), scale.binding(), a_scale.binding(), c.binding()];
        let grid = [(n / 128) as u32, m.div_ceil(256) as u32, 1];
        let config =
            [("k_size", k), ("k_stride", stride), ("n_size", n), ("m_group", options.m_group)];

        println!("{m}x{k}x{n}:");
        let mut kernels = Vec::new();
        for (path, source) in options.sources.iter().zip(&sources) {
            let (kernel, note) = compile(&stream, &compiler, source, &config)?;
            println!("  {path}: {note}");
            let constants = Constants::indices(&kernel, &[m as u32])?;
            // The 256x256 tile launches sixteen waves over twice the columns.
            let wide = source.contains("workgroup_size(%c512");
            let (grid, block) = if wide {
                ([(n / 256) as u32, m.div_ceil(256) as u32, 1], [512u32, 1, 1])
            } else {
                (grid, [256u32, 1, 1])
            };
            kernels.push((kernel, constants, grid, block));
        }

        let mut reference: Option<Vec<u8>> = None;
        for ((kernel, constants, grid, block), path) in kernels.iter().zip(&options.sources) {
            stream.fill(c.binding(), 0)?;
            // SAFETY: operands are sized from the configuration compiled above.
            unsafe { stream.dispatch(kernel, *grid, *block, constants, &bindings) }?;
            let mut out = vec![0u8; m * n * 2];
            stream.read_blocking(c.binding(), &mut out)?;
            match &reference {
                None => reference = Some(out),
                Some(expected) if options.check => {
                    ensure!(&out == expected, "{path} differs from {}", options.sources[0]);
                }
                Some(_) => {}
            }
        }

        let mut graphs = Vec::new();
        let arms = kernels.iter().map(|(kernel, constants, grid, block)| {
            (kernel, constants, *grid, *block, &bindings[..])
        });
        let peak_bindings = [peak_out.binding()];
        let peak_arm = (
            &peak,
            &peak_constants,
            [PEAK_WORKGROUPS as u32, 1, 1],
            [256u32, 1, 1],
            &peak_bindings[..],
        );
        for (kernel, constants, grid, block, bindings) in arms.chain([peak_arm]) {
            let mut graph = stream.graph()?;
            let mut last = None;
            for _ in 0..REPEATS {
                let after: Vec<hrx::Node> = last.into_iter().collect();
                // SAFETY: as the eager dispatch above.
                last = Some(unsafe {
                    graph.dispatch(&after, kernel, grid, block, constants, bindings)
                }?);
            }
            let labels: Vec<String> = (0..REPEATS).map(|i| i.to_string()).collect();
            graphs.push(graph.finish_profiled(&labels).context(
                "device timestamps need a native bridge with HRX's profiling exports",
            )?);
        }

        let mut samples = vec![Vec::new(); graphs.len()];
        for round in 0..options.rounds + 2 {
            for offset in 0..graphs.len() {
                let arm = (round + offset) % graphs.len();
                let profile = stream.launch_profiled(&mut graphs[arm])?;
                if round < 2 {
                    continue;
                }
                let ms = 1e3 / profile.frequency_hz as f64;
                samples[arm].extend(
                    profile.intervals.iter().map(|i| (i.end_tick - i.start_tick) as f64 * ms),
                );
            }
        }
        let medians: Vec<f64> = samples
            .into_iter()
            .map(|mut s| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            })
            .collect();
        let peak_ops = (PEAK_WORKGROUPS * 8 * PEAK_ITERATIONS * 8 * 16 * 16 * 16 * 2) as f64;
        let peak_tops = peak_ops / medians[kernels.len()] / 1e9;
        println!("  register-only peak: {peak_tops:.1} TOPS");
        let ops = 2.0 * (m * k * n) as f64;
        for (path, median) in options.sources.iter().zip(&medians) {
            let tops = ops / median / 1e9;
            println!(
                "  {path}: {median:.3} ms  {tops:.1} TOPS  {:.0}% of peak  x{:.3}",
                100.0 * tops / peak_tops,
                medians[0] / median
            );
        }
    }
    Ok(())
}
