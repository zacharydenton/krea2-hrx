//! `loomrun` — launch one Loom-compiled kernel on the GPU and dump its buffers.
//!
//! The Loom AMDGPU ABI packs kernarg in declaration order: scalars by value at
//! their natural alignment, then one 8-byte device pointer per `buffer`
//! operand. Arguments are given on the command line in that same order.
//!
//! ```text
//! loomrun --hsaco k.hsaco --kernel name --grid 201 --block 256 \
//!         --i32 201 --in x.bin --in gamma.bin --in beta.bin --out y.bin:308736
//! ```
//!
//! `tools/kernel_test.py` drives this to compare a kernel against Torch.
use std::path::{Path, PathBuf};
use std::time::Instant;

use hrx::{device, Args, Buffer, Kernel};

/// A buffer operand: uploaded, downloaded, or both.
enum Direction {
    In,
    Out,
    InOut,
}

struct Operand {
    direction: Direction,
    path: PathBuf,
    bytes: usize,
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("{message}");
            std::process::ExitCode::from(64)
        }
        Err(Failure::Failed(message)) => {
            eprintln!("{message}");
            std::process::ExitCode::FAILURE
        }
    }
}

enum Failure {
    Usage(String),
    Failed(String),
}

impl<E: std::fmt::Display> From<E> for Failure {
    fn from(error: E) -> Self {
        Failure::Failed(error.to_string())
    }
}

fn run() -> Result<(), Failure> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut hsaco: Option<String> = None;
    let mut kernel: Option<String> = None;
    let (mut grid, mut block) = ([1u32; 3], [1u32; 3]);
    let mut repeat = 1usize;
    let mut verbose = false;
    // Scalars and operands share one order: the kernarg blob is built in it.
    let mut args = Args::new();
    let mut operands: Vec<Operand> = Vec::new();
    let mut scalars: Vec<(usize, Scalar)> = Vec::new();

    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        let mut value = || -> Result<String, Failure> {
            index += 1;
            arguments
                .get(index)
                .cloned()
                .ok_or_else(|| Failure::Usage(format!("{option} needs a value")))
        };
        match option {
            "--hsaco" => hsaco = Some(value()?),
            "--kernel" => kernel = Some(value()?),
            "--grid" => grid = triple(&value()?),
            "--block" => block = triple(&value()?),
            "--repeat" => repeat = value()?.parse().unwrap_or(0),
            "--verbose" => verbose = true,
            "--i32" => {
                let parsed = value()?.parse().unwrap_or(0);
                scalars.push((operands.len() + scalars.len(), Scalar::I32(parsed)));
            }
            "--f32" => {
                let parsed = value()?.parse().unwrap_or(0.0);
                scalars.push((operands.len() + scalars.len(), Scalar::F32(parsed)));
            }
            "--in" | "--inout" => {
                let direction = if option == "--in" { Direction::In } else { Direction::InOut };
                operands.push(Operand { direction, path: value()?.into(), bytes: 0 });
            }
            "--out" => {
                // --out path:bytes
                let spec = value()?;
                let (path, bytes) = spec
                    .rsplit_once(':')
                    .ok_or_else(|| Failure::Usage("--out wants path:bytes".into()))?;
                let bytes = bytes
                    .parse()
                    .map_err(|_| Failure::Usage("--out wants path:bytes".into()))?;
                operands.push(Operand { direction: Direction::Out, path: path.into(), bytes });
            }
            other => return Err(Failure::Usage(format!("unknown option {other}"))),
        }
        index += 1;
    }
    let (Some(hsaco), Some(kernel)) = (hsaco, kernel) else {
        return Err(Failure::Usage("need --hsaco and --kernel".into()));
    };
    if repeat < 1 {
        return Err(Failure::Failed("repeat must be positive".into()));
    }

    let function = Kernel::load(Path::new(&hsaco), &kernel)?;
    // Rebuild the declaration order: scalars carry the position they were
    // parsed at, operands fill the rest in their own order.
    let mut scalars = scalars.into_iter().peekable();
    let mut buffers: Vec<(Buffer, &Operand)> = Vec::new();
    let mut next_operand = operands.iter();
    for position in 0..operands.len() + scalars.len() {
        match scalars.peek() {
            Some((at, scalar)) if *at == position => {
                match scalar {
                    Scalar::I32(value) => args.i32(*value),
                    Scalar::F32(value) => args.f32(*value),
                };
                scalars.next();
            }
            _ => {
                let operand = next_operand.next().expect("one operand per position");
                let contents = match operand.direction {
                    Direction::Out => None,
                    _ => Some(std::fs::read(&operand.path).map_err(|e| {
                        Failure::Failed(format!("cannot open {}: {e}", operand.path.display()))
                    })?),
                };
                let bytes = contents.as_ref().map_or(operand.bytes, Vec::len);
                let buffer = device().allocate(bytes.max(1))?;
                device().zero(buffer.ptr(), bytes.max(4))?;
                if let Some(contents) = &contents {
                    device().copy_from_host(buffer.ptr(), contents)?;
                }
                args.ptr(buffer.ptr());
                buffers.push((buffer, operand));
            }
        }
    }
    if verbose {
        eprintln!("kernarg_size={} grid={grid:?} block={block:?}", args.as_bytes().len());
    }

    // A single in-place launch must not be repeated for warm-up.
    if repeat > 1 {
        function.launch(grid, block, &args)?;
        device().synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..repeat {
        function.launch(grid, block, &args)?;
    }
    device().synchronize()?;
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "{{\"launches\": {repeat}, \"total_ms\": {elapsed:.6}, \"per_launch_us\": {:.3}}}",
        1000.0 * elapsed / repeat as f64
    );

    for (buffer, operand) in &buffers {
        if matches!(operand.direction, Direction::In) {
            continue;
        }
        let bytes = match operand.direction {
            Direction::Out => operand.bytes,
            _ => buffer.len(),
        };
        let mut contents = vec![0u8; bytes];
        device().copy_to_host(&mut contents, buffer.ptr())?;
        std::fs::write(&operand.path, &contents).map_err(|e| {
            Failure::Failed(format!("cannot write {}: {e}", operand.path.display()))
        })?;
    }
    Ok(())
}

enum Scalar {
    I32(i32),
    F32(f32),
}

/// `a`, `a,b` or `a,b,c`, the rest left at one.
fn triple(text: &str) -> [u32; 3] {
    let mut values = [1u32; 3];
    for (slot, field) in values.iter_mut().zip(text.split(',')) {
        *slot = field.trim().parse().unwrap_or(1);
    }
    values
}
