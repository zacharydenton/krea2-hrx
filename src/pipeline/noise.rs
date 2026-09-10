//! Seeded standard-normal initial latents using ChaCha8.
//! Native seeds do not reproduce Torch's noise; reference comparisons must supply
//! identical initial latents.
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// `count` standard normal samples, reproducible for a seed on any machine.
pub fn latents(seed: u64, count: usize) -> Vec<f32> {
    let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    StandardNormal.sample_iter(&mut generator).take(count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_latents_and_a_different_one_does_not() {
        let first = latents(7, 4096);
        assert_eq!(first, latents(7, 4096));
        assert_ne!(first, latents(8, 4096));
        // A prefix of a longer draw, so a larger image starts the same way.
        assert_eq!(first, latents(7, 8192)[..4096]);
    }

    #[test]
    fn the_draw_is_standard_normal() {
        let values = latents(11, 1 << 16);
        let mean = values.iter().sum::<f32>() as f64 / values.len() as f64;
        let variance = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>()
            / values.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
