//! Scalar references for the full-training optimizer's persistent representation.

/// Elements sharing one FP32 maximum in each quantized moment.
pub const BLOCK: usize = 256;

/// Philox4x32-10; the counter is independent of scheduling and data sampling.
pub fn philox(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let a = 0xd251_1f53u64 * u64::from(counter[0]);
        let b = 0xcd9e_8d57u64 * u64::from(counter[2]);
        counter = [
            (b >> 32) as u32 ^ counter[1] ^ key[0],
            b as u32,
            (a >> 32) as u32 ^ counter[3] ^ key[1],
            a as u32,
        ];
        key[0] = key[0].wrapping_add(0x9e37_79b9);
        key[1] = key[1].wrapping_add(0xbb67_ae85);
    }
    counter
}

/// Unbiased rounding to BF16 using sixteen independent random bits.
/// Nonfinite values retain their class; callers reject them before updating.
pub fn stochastic_bf16(value: f32, random: u32) -> u16 {
    let bits = value.to_bits();
    if !value.is_finite() {
        return ((bits >> 16) as u16) | if value.is_nan() { 0x40 } else { 0 };
    }
    (bits.wrapping_add(random & 0xffff) >> 16) as u16
}

/// Dynamic 8-bit quantization map, following bitsandbytes' seven-exponent map.
/// Signed moments and nonnegative second moments use separate sorted maps.
pub fn codebook(signed: bool) -> [f32; 256] {
    let mut values = Vec::with_capacity(256);
    for exponent in 0..7 {
        let intervals = 1usize << (exponent + usize::from(!signed));
        let scale = 10.0f32.powi(exponent as i32 - 6);
        for i in 0..intervals {
            let left = 0.1 + 0.9 * i as f32 / intervals as f32;
            let right = 0.1 + 0.9 * (i + 1) as f32 / intervals as f32;
            let v = (left + right) * 0.5 * scale;
            values.push(v);
            if signed {
                values.push(-v);
            }
        }
    }
    values.extend([0.0, 1.0]);
    values.sort_by(f32::total_cmp);
    values.try_into().expect("256 codebook entries")
}

/// Nearest code, with ties assigned to the lower code. Input must be finite.
pub fn quantize(value: f32, maximum: f32, map: &[f32; 256]) -> u8 {
    let x = if maximum == 0.0 { 0.0 } else { value / maximum };
    let upper = map.partition_point(|v| *v < x).min(255);
    let lower = upper.saturating_sub(1);
    if (x - map[lower]).abs() <= (map[upper] - x).abs() { lower as u8 } else { upper as u8 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_matches_random123_known_answer() {
        assert_eq!(philox([0; 4], [0; 2]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
    }

    #[test]
    fn stochastic_rounding_is_unbiased_over_all_random_bits() {
        for value in [1.001, -1.001, 0.0, -0.0, 1e-20, -1e20] {
            let mean = (0..65536)
                .map(|r| f64::from(f32::from_bits(u32::from(stochastic_bf16(value, r)) << 16)))
                .sum::<f64>()
                / 65536.0;
            assert!((mean - f64::from(value)).abs() <= f64::from(value).abs() * 1e-12);
        }
        assert!(f32::from_bits(u32::from(stochastic_bf16(f32::NAN, 0)) << 16).is_nan());
    }

    #[test]
    fn codebooks_preserve_zero_order_and_representable_values() {
        for signed in [false, true] {
            let map = codebook(signed);
            assert!(map.windows(2).all(|p| p[0] < p[1]));
            assert_eq!(map[255], 1.0);
            assert_eq!(map[quantize(0.0, 0.0, &map) as usize], 0.0);
            for (i, &value) in map.iter().enumerate() {
                assert_eq!(quantize(value, 1.0, &map) as usize, i);
            }
        }
    }
}
