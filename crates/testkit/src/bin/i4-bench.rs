//! `i4-bench BASELINE.hsaco CANDIDATE.hsaco key=value...`
//!
//! Paired, interleaved timing of two INT4 GEMM kernels on the same resident
//! inputs, for any of the three epilogues (plain, resid, swiglu). Both sides
//! see identical operands and the same number of residual updates; outputs are
//! compared bit for bit before and after timing whenever the operands agree —
//! they differ only when the two sides are given different K, which is how the
//! operand-pitch falsifier runs one kernel at two row pitches.
//!
//! ```text
//!   mode=plain|resid|swiglu   symbol=... symbol_cand=...
//!   m= n= k= [k_cand=] [k_stride= k_stride_cand=]   (k_stride: operand row pitch)
//!   tile=128|256 tile_cand= group= group_cand= [grid_group= grid_group_cand=] rounds=
//! ```
//!
//! The launch grid is `n/128 x ceil(ceil(m/tile)/group)*group`, the padded
//! raster form; a kernel that shortens its own tail takes `group=1`.
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use hrx::{device, Args, Buffer, Kernel};

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
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 {
        return Err("usage: i4-bench BASELINE CANDIDATE key=value...".into());
    }
    let mut options: BTreeMap<&str, &str> = BTreeMap::new();
    for item in &arguments[3..] {
        match item.split_once('=') {
            Some((key, value)) if !key.is_empty() => options.insert(key, value),
            _ => return Err(format!("expected key=value: {item}").into()),
        };
    }
    let text = |key: &str, fallback: &'static str| -> String {
        options.get(key).map_or_else(|| fallback.to_string(), |value| (*value).to_string())
    };
    let number = |key: &str, fallback: i64| -> Result<i64, std::num::ParseIntError> {
        match options.get(key) {
            Some(value) => value.parse(),
            None => Ok(fallback),
        }
    };

    let mode = text("mode", "plain");
    if !["plain", "resid", "swiglu"].contains(&mode.as_str()) {
        return Err("mode must be plain, resid or swiglu".into());
    }
    let m = number("m", 4115)? as usize;
    let n = number("n", 6144)? as usize;
    let rounds = number("rounds", 80)? as usize;
    let k = [number("k", 6144)? as usize, 0];
    let k = [k[0], number("k_cand", k[0] as i64)? as usize];
    let stride = [number("k_stride", k[0] as i64)? as usize, 0];
    let stride = [stride[0], number("k_stride_cand", k[1] as i64)? as usize];
    let tile = [number("tile", 128)? as usize, 0];
    let tile = [tile[0], number("tile_cand", tile[0] as i64)? as usize];
    let group = [number("group", 4)? as usize, 0];
    let group = [group[0], number("group_cand", group[0] as i64)? as usize];
    // Grid rows are padded to a multiple of grid_group; a kernel that shortens
    // its own raster tail takes grid_group=1 with its full m_group config.
    let grid_group = [number("grid_group", group[0] as i64)? as usize, 0];
    let grid_group = [grid_group[0], number("grid_group_cand", group[1] as i64)? as usize];
    let baseline_symbol = text("symbol", "krea2_gemm_i4");
    let candidate_symbol = match options.get("symbol_cand") {
        Some(value) => (*value).to_string(),
        None => baseline_symbol.clone(),
    };
    let symbol = [baseline_symbol, candidate_symbol];

    if !(1..=16896).contains(&m)
        || !(128..=32768).contains(&n)
        || !n.is_multiple_of(128)
        || !(10..=10000).contains(&rounds)
    {
        return Err("invalid benchmark dimensions".into());
    }
    for side in 0..2 {
        if !(128..=65536).contains(&k[side])
            || !k[side].is_multiple_of(128)
            || stride[side] < k[side]
            || !stride[side].is_multiple_of(128)
            || (tile[side] != 128 && tile[side] != 256)
            || !(1..=4).contains(&group[side])
            || !(1..=4).contains(&grid_group[side])
        {
            return Err("invalid kernel shape".into());
        }
    }
    let same_operands = k[0] == k[1] && stride[0] == stride[1];
    let kernels = [
        Kernel::load(Path::new(&arguments[1]), &symbol[0])?,
        Kernel::load(Path::new(&arguments[2]), &symbol[1])?,
    ];

    // Operands per side: identical bytes when the pitches agree, else each
    // side gets its own rows, from the same seed, so the K prefix matches.
    let mut operands = Vec::new();
    for &pitch in stride.iter().take(if same_operands { 1 } else { 2 }) {
        let mut random = Xorshift::new(42);
        operands
            .push((packed(&mut random, m * pitch / 2)?, packed(&mut random, n * pitch / 2)?));
    }
    let mut random = Xorshift::new(42);
    let weight_scales = scales(&mut random, n, false)?;
    let activation_scales = scales(&mut random, m, false)?;
    let gate = scales(&mut random, n, true)?;

    let columns = if mode == "swiglu" { n / 2 } else { n };
    let output = [device().allocate(m * columns * 2)?, device().allocate(m * columns * 2)?];
    // A residual epilogue reads what is already there, so it starts at values
    // near one rather than at whatever the allocator held.
    let initial: Vec<u16> = (0..m * columns)
        .map(|_| (0x3800 + random.next() % 2048 + (random.next() % 2) * 0x8000) as u16)
        .collect();
    let mut args = Vec::new();
    for (side, buffer) in output.iter().enumerate() {
        device().write(buffer.ptr(), &initial)?;
        let operand = if same_operands { 0 } else { side };
        let mut blob = Args::new();
        blob.i32(m as i32)
            .ptr(operands[operand].0.ptr())
            .ptr(operands[operand].1.ptr())
            .ptr(weight_scales.ptr())
            .ptr(activation_scales.ptr())
            .ptr(buffer.ptr());
        if mode == "resid" {
            blob.ptr(gate.ptr());
        }
        args.push(blob);
    }
    let launch = |side: usize| -> Result<(), hrx::Error> {
        let tiles = m.div_ceil(tile[side]);
        let rows = tiles.div_ceil(grid_group[side]) * grid_group[side];
        kernels[side].launch_2d((n / 128) as u32, rows as u32, 256, &args[side])
    };
    let compare = || -> Result<(), Box<dyn std::error::Error>> {
        if !same_operands {
            return Ok(());
        }
        let mut reference = vec![0u16; initial.len()];
        let mut candidate = vec![0u16; initial.len()];
        device().read(&mut reference, output[0].ptr())?;
        device().read(&mut candidate, output[1].ptr())?;
        if reference != candidate {
            return Err("candidate differs from baseline".into());
        }
        Ok(())
    };

    launch(0)?;
    launch(1)?;
    compare()?;
    for _ in 0..5 {
        launch(0)?;
        launch(1)?;
    }
    device().synchronize()?;
    let mut times = [Vec::with_capacity(rounds), Vec::with_capacity(rounds)];
    for round in 0..rounds {
        for order in 0..2 {
            let side = order ^ (round % 2);
            let start = Instant::now();
            launch(side)?;
            device().synchronize()?;
            times[side].push(start.elapsed().as_secs_f64() * 1e3);
        }
    }
    compare()?;

    let mut ratios: Vec<f64> =
        (0..rounds).map(|round| times[0][round] / times[1][round]).collect();
    ratios.sort_by(f64::total_cmp);
    print!(
        "{{\"mode\":\"{mode}\",\"exact\":{},\"m\":{m},\"n\":{n},\"rounds\":{rounds},\
         \"paired_median_speedup\":{:.6}",
        if same_operands { "true" } else { "null" },
        ratios[rounds / 2]
    );
    for (side, name) in [(0, "baseline"), (1, "candidate")] {
        times[side].sort_by(f64::total_cmp);
        let median = times[side][rounds / 2];
        let tops = 2.0 * m as f64 * n as f64 * k[side] as f64 / (median * 1e-3) / 1e12;
        print!(
            ",\"{name}\":{{\"k\":{},\"k_stride\":{},\"tile\":{},\"group\":{},\
             \"min_ms\":{:.6},\"p10_ms\":{:.6},\"median_ms\":{median:.6},\
             \"p90_ms\":{:.6},\"median_tops\":{tops:.3}}}",
            k[side],
            stride[side],
            tile[side],
            group[side],
            times[side][0],
            times[side][rounds / 10],
            times[side][rounds * 9 / 10]
        );
    }
    println!("}}");
    Ok(())
}

/// Packed int4 nibbles: the values are noise, only their distribution matters.
fn packed(random: &mut Xorshift, bytes: usize) -> Result<Buffer, hrx::Error> {
    let values: Vec<u8> = (0..bytes).map(|_| random.next() as u8).collect();
    let buffer = device().allocate(bytes)?;
    device().copy_from_host(buffer.ptr(), &values)?;
    Ok(buffer)
}

fn scales(random: &mut Xorshift, count: usize, signed: bool) -> Result<Buffer, hrx::Error> {
    let values: Vec<f32> = (0..count)
        .map(|_| {
            let magnitude = (random.next() % 1000) as i64 - if signed { 500 } else { 0 };
            magnitude as f32 * 0.00001
        })
        .collect();
    let buffer = device().allocate(count * 4)?;
    device().write(buffer.ptr(), &values)?;
    Ok(buffer)
}

/// A seeded stream. The C++ used `std::mt19937`; nothing here depends on the
/// exact sequence, only on it being the same for both sides.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Xorshift {
        Xorshift(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 >> 32
    }
}
