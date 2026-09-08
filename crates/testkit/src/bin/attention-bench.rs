//! `BASE_HSACO BASE_SYMBOL CAND_HSACO CAND_SYMBOL TOKENS CAPACITY QTILES
//! ROUNDS ORACLE_ROWS INPUT_DIR [TRANSPOSE_HSACO]`
//!
//! Paired fp16 attention on resident inputs. Python supplies an independent
//! CPU oracle for selected query rows; every output is compared with the
//! baseline too, before and after the timed rounds.
use std::path::Path;
use std::time::Instant;

use half::f16;
use hrx::{device, Args, Buffer, Kernel};

const HEADS: usize = 48;
const KV_HEADS: usize = 12;
const DIM: usize = 128;

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 11 && a.len() != 12 {
        return Err("usage: attention-bench BASE_HSACO BASE_SYMBOL CAND_HSACO CAND_SYMBOL \
                    TOKENS CAPACITY QTILES ROUNDS ORACLE_ROWS INPUT_DIR [TRANSPOSE_HSACO]"
            .into());
    }
    let (tokens, capacity): (usize, usize) = (a[5].parse()?, a[6].parse()?);
    let (qtiles, rounds, oracle_rows): (usize, usize, usize) =
        (a[7].parse()?, a[8].parse()?, a[9].parse()?);
    if !(16..=65536).contains(&tokens)
        || capacity < tokens + 16
        || capacity > 65600
        || !capacity.is_multiple_of(64)
        || ![1, 2, 4].contains(&qtiles)
        || !(10..=10000).contains(&rounds)
        || oracle_rows < 1
        || oracle_rows > tokens
    {
        return Err("invalid benchmark dimensions".into());
    }
    let transposed = a.len() == 12;
    let input = Path::new(&a[10]);

    let q: Vec<f16> = read(input, "q.bin", capacity * HEADS * DIM)?;
    let k: Vec<f16> = read(input, "k.bin", capacity * KV_HEADS * DIM)?;
    let v: Vec<f16> = read(input, "v.bin", k.len())?;
    let rows: Vec<u32> = read(input, "rows.bin", oracle_rows)?;
    let want: Vec<f32> = read(input, "want.bin", oracle_rows * HEADS * DIM)?;
    if rows.iter().any(|&row| row as usize >= tokens) {
        return Err("oracle row outside sequence".into());
    }

    let kernels =
        [Kernel::load(Path::new(&a[1]), &a[2])?, Kernel::load(Path::new(&a[3]), &a[4])?];
    let transpose = match transposed {
        true => Some(Kernel::load(Path::new(&a[11]), "krea2_sage_transpose")?),
        false => None,
    };
    let q_device = upload(&q)?;
    let k_device = upload(&k)?;
    let v_device = upload(&v)?;
    let vt_device = device().allocate(if transposed { v.len() * 2 } else { 2 })?;
    if transposed {
        device().zero(vt_device.ptr(), v.len() * 2)?;
    }

    let count = tokens * HEADS * DIM;
    let outputs = [device().allocate(count * 2)?, device().allocate(count * 2)?];
    // Poison the output to catch missing rows or channels in publication.
    let poison = vec![f16::NAN.to_bits(); count];
    let mut args = Vec::new();
    for (which, output) in outputs.iter().enumerate() {
        device().write(output.ptr(), &poison)?;
        // token_count is an 8-byte Loom index. KV heads belong to the grid,
        // not the launch arguments (putting them here corrupts its high word).
        let mut blob = Args::new();
        blob.i64(tokens as i64)
            .ptr(q_device.ptr())
            .ptr(k_device.ptr())
            .ptr(if which == 1 && transposed { vt_device.ptr() } else { v_device.ptr() })
            .ptr(output.ptr());
        args.push(blob);
    }
    let mut transpose_args = Args::new();
    transpose_args.i64(tokens as i64).ptr(v_device.ptr()).ptr(vt_device.ptr());

    let launch = |which: usize| -> Result<(), hrx::Error> {
        if which == 1 {
            if let Some(transpose) = &transpose {
                transpose.launch_2d(
                    tokens.div_ceil(32) as u32,
                    (KV_HEADS * DIM / 32) as u32,
                    256,
                    &transpose_args,
                )?;
            }
        }
        let tiles = if which == 1 { qtiles } else { 1 };
        kernels[which].launch_2d(
            tokens.div_ceil(16 * tiles) as u32,
            KV_HEADS as u32,
            (128 * tiles) as u32,
            &args[which],
        )
    };
    let check = || -> Result<(), Box<dyn std::error::Error>> {
        let mut actual = [vec![0u16; count], vec![0u16; count]];
        for which in 0..2 {
            device().read(&mut actual[which], outputs[which].ptr())?;
            let mut oracle = Deviation::default();
            for (index, &row) in rows.iter().enumerate() {
                for channel in 0..HEADS * DIM {
                    oracle.add(
                        f16::from_bits(actual[which][row as usize * HEADS * DIM + channel])
                            .to_f32(),
                        want[index * HEADS * DIM + channel],
                    )?;
                }
            }
            oracle.check(if which == 1 {
                "candidate vs CPU oracle"
            } else {
                "baseline vs CPU oracle"
            })?;
        }
        let mut full = Deviation::default();
        let [reference, candidate] = &actual;
        for (&candidate, &reference) in candidate.iter().zip(reference) {
            full.add(f16::from_bits(candidate).to_f32(), f16::from_bits(reference).to_f32())?;
        }
        full.check("candidate vs baseline, all outputs")
    };

    launch(0)?;
    launch(1)?;
    device().synchronize()?;
    check()?;
    for _ in 0..5 {
        launch(0)?;
        launch(1)?;
    }
    device().synchronize()?;

    let mut times = [Vec::with_capacity(rounds), Vec::with_capacity(rounds)];
    let mut ratios = Vec::with_capacity(rounds);
    for round in 0..rounds {
        for order in 0..2 {
            let which = order ^ (round % 2);
            let start = Instant::now();
            launch(which)?;
            device().synchronize()?;
            times[which].push(start.elapsed().as_secs_f64() * 1e3);
        }
        ratios.push(times[0][round] / times[1][round]);
    }
    check()?;

    ratios.sort_by(f64::total_cmp);
    print!(
        "{{\"correct\":true,\"tokens\":{tokens},\"rounds\":{rounds},\
         \"paired_speedup\":{{\"p10\":{:.6},\"median\":{:.6},\"p90\":{:.6}}}",
        ratios[rounds / 10],
        ratios[rounds / 2],
        ratios[rounds * 9 / 10]
    );
    for (which, name) in [(0, "baseline"), (1, "candidate")] {
        times[which].sort_by(f64::total_cmp);
        print!(
            ",\"{name}_ms\":{{\"min\":{:.6},\"p10\":{:.6},\"median\":{:.6},\"p90\":{:.6}}}",
            times[which][0],
            times[which][rounds / 10],
            times[which][rounds / 2],
            times[which][rounds * 9 / 10]
        );
    }
    println!("}}");
    Ok(())
}

/// How far one output is from another, absolutely and in relative RMS.
struct Deviation {
    squared: f64,
    reference_squared: f64,
    maximum: f64,
    close: bool,
}

impl Default for Deviation {
    fn default() -> Self {
        // Nothing compared yet is trivially close.
        Deviation { squared: 0.0, reference_squared: 0.0, maximum: 0.0, close: true }
    }
}

impl Deviation {
    fn add(&mut self, actual: f32, expected: f32) -> Result<(), Box<dyn std::error::Error>> {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("nonfinite attention output".into());
        }
        let delta = f64::from(actual) - f64::from(expected);
        self.squared += delta * delta;
        self.reference_squared += f64::from(expected) * f64::from(expected);
        self.maximum = self.maximum.max(delta.abs());
        self.close &= delta.abs() <= 0.002 + 0.002 * f64::from(expected.abs());
        Ok(())
    }

    fn relative_rms(&self) -> f64 {
        (self.squared / self.reference_squared.max(1e-30)).sqrt()
    }

    fn check(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!(
            "{name}: max_abs={:.6} relative_rms={:.6}",
            self.maximum,
            self.relative_rms()
        );
        if !self.close || self.relative_rms() > 0.002 {
            return Err(format!("{name} failed accuracy check").into());
        }
        Ok(())
    }
}

/// A fixture of exactly `count` elements.
fn read<T: bytemuck::Pod>(
    directory: &Path,
    name: &str,
    count: usize,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let path = directory.join(name);
    let bytes = std::fs::read(&path)?;
    if bytes.len() != count * std::mem::size_of::<T>() {
        return Err(format!("wrong input size: {}", path.display()).into());
    }
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn upload(values: &[f16]) -> Result<Buffer, hrx::Error> {
    let bits: Vec<u16> = values.iter().map(|value| value.to_bits()).collect();
    let buffer = device().allocate(bits.len() * 2)?;
    device().write(buffer.ptr(), &bits)?;
    Ok(buffer)
}
