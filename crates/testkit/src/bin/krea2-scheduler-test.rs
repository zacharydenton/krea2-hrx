//! `<directory>`: the production Euler step over a fixture, and the sigmas it
//! used, so `tests/test_native_regressions.py` can compare both against CUDA
//! diffusers. No models: this is the scheduler alone.
use std::path::Path;

use krea2_numerics::{from_f32, to_f32};
use krea2_ops::{Ops, Pool, Tensor};
use krea2_pipeline::schedule;

const ROWS: usize = 5050;
const WIDTH: usize = 256;

fn main() -> std::process::ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, directory] = &arguments[..] else {
        eprintln!("usage: krea2-scheduler-test <directory>");
        return std::process::ExitCode::from(2);
    };
    match run(Path::new(directory)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let ops = Ops::new(Pool::new());
    let read = |name: &str| -> Result<Tensor, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(root.join(name))?;
        let values: Vec<u16> =
            bytemuck::cast_slice::<u8, f32>(&bytes).iter().map(|&v| from_f32(v)).collect();
        if values.len() != ROWS * WIDTH {
            return Err("cannot read scheduler fixture".into());
        }
        Ok(Tensor::from_slice(ops.pool(), &values, ROWS, WIDTH)?)
    };
    let samples = read("samples.bin")?;
    let velocity = read("velocity.bin")?;

    // Every step count from 1 to 100, laid end to end: 5,050 steps.
    let mut sigmas = vec![0f32; ROWS];
    let mut row = 0;
    for steps in 1..=100 {
        for step in 0..steps {
            sigmas[row] = schedule::sigma(step, steps, 1.15);
            let delta = schedule::sigma(step + 1, steps, 1.15) - sigmas[row];
            ops.euler_step(
                &samples.view(1, WIDTH, row * WIDTH)?,
                &velocity.view(1, WIDTH, row * WIDTH)?,
                delta,
            )?;
            row += 1;
        }
    }
    std::fs::write(root.join("sigmas.bin"), bytemuck::cast_slice(&sigmas))?;
    let result: Vec<f32> = samples.download()?.into_iter().map(to_f32).collect();
    std::fs::write(root.join("result.bin"), bytemuck::cast_slice(&result))?;
    Ok(())
}
