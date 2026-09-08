//! The sigma grid for every step count from 1 to 100, at each image size,
//! written to stdout as float32.
//!
//! `tests/test_schedule.py` compares it against diffusers on the CPU. No GPU,
//! no models: the scheduler alone.
use std::io::Write;

use krea2_pipeline::schedule;

/// 589 tokens (304x496) exposes the early float32 rounding of Raw's mu; zero
/// stands for Turbo, whose shift is fixed.
const SIZES: [usize; 7] = [0, 16, 256, 589, 4096, 6400, 16384];

fn main() -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for tokens in SIZES {
        let mu = if tokens == 0 { 1.15 } else { schedule::dynamic_mu(tokens) };
        for steps in 1..=100 {
            for step in 0..=steps {
                out.write_all(&schedule::sigma(step, steps, mu).to_le_bytes())?;
            }
        }
    }
    out.flush()
}
