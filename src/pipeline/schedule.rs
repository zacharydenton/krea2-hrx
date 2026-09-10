//! diffusers' exponential time shift of `linspace(1, 1/steps)` sigmas.
//!
//! Turbo (the distilled checkpoint) uses the fixed `mu = 1.15`; Raw computes
//! it from the image's token count. The float boundaries here are diffusers':
//! it casts the linspace to float32 before NumPy's shift operations, and a
//! tiny sigma difference can cross a bf16 tie in the Euler delta.

/// Flux's `calculate_shift` with Krea 2's scheduler config: base 256 tokens at
/// 0.5, 6400 tokens at 1.15, linear between (1024² = 4096 tokens: 0.906).
pub fn dynamic_mu(image_tokens: usize) -> f64 {
    let slope = (1.15 - 0.5) / (6400.0 - 256.0);
    image_tokens as f64 * slope + (0.5 - slope * 256.0)
}

/// The sigma at `step` of `steps`; the last one is zero.
pub fn sigma(step: usize, steps: usize, mu: f64) -> f32 {
    if step == steps {
        return 0.0;
    }
    let raw = (1.0 - step as f64 / steps as f64) as f32;
    // mu stays in double through exp, as Python keeps it; rounding it first
    // can move a later bf16 Euler delta across a rounding boundary.
    let shift = mu.exp() as f32;
    let offset = 1.0 / raw - 1.0;
    shift / (shift + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_turbo_curve_starts_at_one_and_ends_at_zero() {
        assert_eq!(sigma(0, 8, 1.15), 1.0);
        assert_eq!(sigma(8, 8, 1.15), 0.0);
        // Strictly decreasing, which is what makes every Euler delta negative.
        for step in 0..8 {
            assert!(sigma(step, 8, 1.15) > sigma(step + 1, 8, 1.15), "step {step}");
        }
    }

    #[test]
    fn raws_shift_follows_the_image_size() {
        // The three points the rule is defined by.
        assert!((dynamic_mu(256) - 0.5).abs() < 1e-12);
        assert!((dynamic_mu(6400) - 1.15).abs() < 1e-12);
        assert!((dynamic_mu(4096) - 0.906).abs() < 1e-3);
    }
}
