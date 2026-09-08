//! `BASELINE CANDIDATE SYMBOL M N K TILE_M TILE_N ROUNDS [BASE_SYMBOL BASE_TILE_M BASE_TILE_N]`
//!
//! Interleaved bf16 GEMMs on identical resident buffers. Timing includes HRX
//! submission and synchronization; correctness is checked before timing, and
//! the two kernels alternate so drift affects both equally.
use std::path::Path;
use std::time::Instant;

use hrx::{device, Args, Buffer, Kernel};
use krea2_numerics::from_f32_carrying;

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
    if a.len() != 10 && a.len() != 13 {
        return Err("usage: gemm-bench BASELINE CANDIDATE SYMBOL M N K TILE_M TILE_N \
                    ROUNDS [BASE_SYMBOL BASE_TILE_M BASE_TILE_N]"
            .into());
    }
    let (m, n, k): (usize, usize, usize) = (a[4].parse()?, a[5].parse()?, a[6].parse()?);
    let (tile, columns): (usize, usize) = (a[7].parse()?, a[8].parse()?);
    let rounds: usize = a[9].parse()?;
    // The baseline defaults to the plain 64x64 kernel; a same-tile baseline
    // isolates an operand-path change from tile selection.
    let base_symbol = if a.len() == 13 { a[10].as_str() } else { "krea2_gemm_bf16_bf16_nt" };
    let (base_tile, base_columns): (usize, usize) = match a.len() {
        13 => (a[11].parse()?, a[12].parse()?),
        _ => (64, 64),
    };
    let tile_ok = |t: usize| t == 64 || t == 128;
    if m < 1
        || n < 1
        || k < 1
        || !(10..=10000).contains(&rounds)
        || ![tile, columns, base_tile, base_columns].into_iter().all(tile_ok)
        || [m * n, m * k, n * k].into_iter().any(|size| size > 268435456)
    {
        return Err("invalid benchmark dimensions".into());
    }

    let kernels =
        [Kernel::load(Path::new(&a[1]), base_symbol)?, Kernel::load(Path::new(&a[2]), &a[3])?];
    let left = operand(m * k, 1)?;
    let right = operand(n * k, 2)?;
    let bytes = m * n * 2;
    let output = [device().allocate(bytes)?, device().allocate(bytes)?];
    let args: Vec<Args> = output
        .iter()
        .map(|buffer| {
            let mut args = Args::new();
            args.i32(m as i32).f32(1.0).ptr(left.ptr()).ptr(right.ptr()).ptr(buffer.ptr());
            args
        })
        .collect();
    let launch = |which: usize| -> Result<(), hrx::Error> {
        let (rows, cols) = if which == 1 { (tile, columns) } else { (base_tile, base_columns) };
        kernels[which].launch_2d(
            n.div_ceil(cols) as u32,
            m.div_ceil(rows) as u32,
            256,
            &args[which],
        )
    };

    launch(0)?;
    launch(1)?;
    let mut reference = vec![0u16; m * n];
    let mut candidate = vec![0u16; m * n];
    device().read(&mut reference, output[0].ptr())?;
    device().read(&mut candidate, output[1].ptr())?;
    if reference != candidate {
        return Err("candidate differs from baseline".into());
    }

    for _ in 0..5 {
        launch(0)?;
        launch(1)?;
    }
    device().synchronize()?;
    let mut times = [Vec::with_capacity(rounds), Vec::with_capacity(rounds)];
    for round in 0..rounds {
        for order in 0..2 {
            // Alternate which runs first, so submission order cannot favour one.
            let which = order ^ (round % 2);
            let start = Instant::now();
            launch(which)?;
            device().synchronize()?;
            times[which].push(start.elapsed().as_secs_f64() * 1e3);
        }
    }

    let mut ratios: Vec<f64> =
        (0..rounds).map(|round| times[0][round] / times[1][round]).collect();
    ratios.sort_by(f64::total_cmp);
    print!(
        "{{\"exact\":true,\"m\":{m},\"n\":{n},\"k\":{k},\"rounds\":{rounds},\
         \"paired_median_speedup\":{:.6}",
        ratios[rounds / 2]
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

/// A resident bf16 operand of pseudorandom values in [-4, 4).
fn operand(count: usize, seed: u64) -> Result<Buffer, hrx::Error> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let values: Vec<u16> = (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            from_f32_carrying(((state >> 40) as i32 % 8192 - 4096) as f32 / 1024.0)
        })
        .collect();
    let buffer = device().allocate(count * 2)?;
    device().write(buffer.ptr(), &values)?;
    Ok(buffer)
}
