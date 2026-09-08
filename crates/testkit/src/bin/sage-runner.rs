//! `HSACO SYMBOL TOKENS HEADS KV CAP DIR REPEAT`: one smoothed attention
//! kernel with its preparation, timed and dumped.
//!
//! `tests/test_sage_attention.py` compares the output against a Torch oracle
//! and reports the split between preparation and attention.
use std::path::Path;
use std::time::Instant;

use hrx::{device, Args, Buffer, Kernel};
use krea2_session::Sage;

fn main() -> std::process::ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    match run(&arguments) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let [_, hsaco, symbol, tokens, heads, kv, capacity, directory, repeat] = arguments else {
        return Err("usage: sage-runner HSACO SYMBOL TOKENS HEADS KV CAP DIR REPEAT".into());
    };
    let (tokens, heads): (usize, usize) = (tokens.parse()?, heads.parse()?);
    let (kv, capacity, repeat): (usize, usize, usize) =
        (kv.parse()?, capacity.parse()?, repeat.parse()?);
    // krea2_attention_sage_i{4,8}_fast[_prefetch]: the code width selects the
    // preparation, the suffix the wave count.
    let (bits, tail) = match symbol.strip_prefix("krea2_attention_sage_i4_fast") {
        Some(tail) => (4, tail),
        None => match symbol.strip_prefix("krea2_attention_sage_i8_fast") {
            Some(tail) => (8, tail),
            None => return Err("invalid dimensions".into()),
        },
    };
    let waves = match tail {
        "" => 8,
        "_prefetch" => 4,
        _ => return Err("invalid dimensions".into()),
    };
    if tokens < 16
        || capacity < tokens + 16
        || !capacity.is_multiple_of(32)
        || heads != 4 * kv
        || kv < 1
        || repeat < 1
    {
        return Err("invalid dimensions".into());
    }

    let directory = Path::new(directory);
    let q = read(directory, "q.bin", capacity * heads * 128 * 2)?;
    let k = read(directory, "k.bin", capacity * kv * 128 * 2)?;
    let v = read(directory, "v.bin", capacity * kv * 128 * 2)?;
    let out = device().allocate(tokens * heads * 128 * 2)?;

    let sage = Sage::new(tokens, capacity, heads, kv, bits)?;
    let kernel = Kernel::load(Path::new(hsaco), symbol)?;
    let mut args = Args::new();
    args.i32(tokens as i32)
        .i32(kv as i32)
        .ptr(sage.q4.ptr())
        .ptr(sage.k4.ptr())
        .ptr(sage.v_transposed.ptr())
        .ptr(sage.q_scale.ptr())
        .ptr(sage.k_scale.ptr())
        .ptr(sage.correction.ptr())
        .ptr(out.ptr());
    let rows = 16 * (waves / 4);
    let launch =
        || kernel.launch_2d(tokens.div_ceil(rows) as u32, kv as u32, waves as u32 * 32, &args);

    let prepare = |q: &Buffer, k: &Buffer, v: &Buffer| sage.run(q.ptr(), k.ptr(), v.ptr());
    prepare(&q, &k, &v)?;
    launch()?;
    device().synchronize()?;

    // Synchronized wall time includes HRX submission overhead.
    let (mut prepare_ms, mut attention_ms) = (0.0, 0.0);
    for _ in 0..repeat {
        let a = Instant::now();
        prepare(&q, &k, &v)?;
        device().synchronize()?;
        let b = Instant::now();
        launch()?;
        device().synchronize()?;
        let c = Instant::now();
        prepare_ms += (b - a).as_secs_f64() * 1e3;
        attention_ms += (c - b).as_secs_f64() * 1e3;
    }

    let mut result = vec![0u16; tokens * heads * 128];
    device().read(&mut result, out.ptr())?;
    std::fs::write(directory.join("out.bin"), bytemuck::cast_slice(&result))?;
    let repeat = repeat as f64;
    println!(
        "{{\"prepare_ms\":{},\"attention_ms\":{},\"total_ms\":{}}}",
        prepare_ms / repeat,
        attention_ms / repeat,
        (prepare_ms + attention_ms) / repeat
    );
    Ok(())
}

/// A fixture file of exactly `bytes`, uploaded.
fn read(
    directory: &Path,
    name: &str,
    bytes: usize,
) -> Result<Buffer, Box<dyn std::error::Error>> {
    let contents = std::fs::read(directory.join(name))?;
    if contents.len() != bytes {
        return Err("wrong input file size".into());
    }
    let buffer = device().allocate(bytes)?;
    device().copy_from_host(buffer.ptr(), &contents)?;
    Ok(buffer)
}
