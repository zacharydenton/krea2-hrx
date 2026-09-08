//! Host-side bf16 and fp8 E4M3 conversions.
//!
//! Activation and checkpoint conversions both round finite values to nearest,
//! ties to even, but preserve distinct NaN handling for numerical compatibility.
#![forbid(unsafe_code)]

/// bf16 bits: the top half of an f32, rounded.
pub type Bits = u16;

/// bf16 → f32. Exact: bf16 is the high half of an f32.
pub fn to_f32(bits: Bits) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 → bf16, round to nearest even, **quieting NaN**.
///
/// Used for activations and intermediate pipeline values; preserves NaNs.
pub fn from_f32(value: f32) -> Bits {
    let u = value.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN: keep the sign and payload, set the quiet bit.
        ((u >> 16) | 0x40) as Bits
    } else {
        ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as Bits
    }
}

/// f32 → bf16, round to nearest even, **without the NaN branch**.
///
/// Used for checkpoint conversion and fp8 dequantization. The rounding carry
/// can change a NaN's exponent: `0x7FFF_FFFF` becomes negative zero (`0x8000`).
/// This behavior is retained for checkpoint compatibility.
pub fn from_f32_carrying(value: f32) -> Bits {
    let u = value.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as Bits
}

/// fp8 E4M3 (`torch.float8_e4m3fn`) → f32, as the `_scaled` text encoders store
/// it: one NaN encoding, subnormals at 2^-9 steps, no infinities.
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = byte >> 7;
    let exponent = (byte >> 3) & 15;
    let mantissa = byte & 7;
    let value = if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-9) // m/8 * 2^-6
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values a rounding bug would most likely land on: ties, the boundary
    /// where rounding carries into the exponent, subnormals, and every class of
    /// NaN payload.
    const ADVERSARIAL: &[u32] = &[
        0x0000_0000,
        0x8000_0000, // zeros
        0x0000_0001,
        0x8000_0001, // smallest subnormals
        0x3f80_0000,
        0xbf80_0000, // ±1
        0x3f80_8000,
        0x3f81_8000, // exact ties, up and down
        0x3f80_7fff,
        0x3f80_8001, // either side of a tie
        0x7f7f_ffff,
        0xff7f_ffff, // largest finite
        0x7f80_0000,
        0xff80_0000, // ±inf
        0x7f80_0001,
        0x7fbf_ffff, // signalling NaN
        0x7fc0_0000,
        0x7fff_ffff, // quiet NaN, and the carrying one
        0xffff_ffff,
        0xffc0_0000, // negative NaN
    ];

    #[test]
    fn round_trips_through_bf16_are_exact_for_representable_values() {
        for value in [0.0f32, 1.0, -2.5, 65536.0, f32::MIN_POSITIVE] {
            assert_eq!(to_f32(from_f32(value)), value, "{value} did not round-trip");
        }
    }

    /// `half` is an independent implementation of the same rounding; it must
    /// agree everywhere, NaN payloads included.
    #[test]
    fn the_quieting_conversion_matches_the_half_crate() {
        for &bits in ADVERSARIAL {
            let value = f32::from_bits(bits);
            assert_eq!(
                from_f32(value),
                half::bf16::from_f32(value).to_bits(),
                "{bits:#010x} ({value})"
            );
        }
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..2_000_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = f32::from_bits(state as u32);
            assert_eq!(
                from_f32(value),
                half::bf16::from_f32(value).to_bits(),
                "{:#010x}",
                state as u32
            );
        }
    }

    /// The two conversions differ only on NaN — and there they really do
    /// differ, which is why both exist.
    #[test]
    fn the_carrying_conversion_differs_only_on_nan() {
        for &bits in ADVERSARIAL {
            let value = f32::from_bits(bits);
            if value.is_nan() {
                continue;
            }
            assert_eq!(from_f32(value), from_f32_carrying(value), "{bits:#010x}");
        }
        // The carry reaches the exponent: the largest NaN becomes negative zero.
        assert_eq!(from_f32_carrying(f32::from_bits(0x7fff_ffff)), 0x8000);
        assert_eq!(from_f32(f32::from_bits(0x7fff_ffff)), 0x7fff);
    }

    /// Every f32, against `half`, for the conversion that has an independent
    /// implementation. Minutes, so it is not in the default run:
    /// `cargo test --release -p krea2-numerics -- --ignored`.
    #[test]
    #[ignore = "sweeps all 2^32 f32 patterns"]
    fn the_quieting_conversion_matches_half_over_every_f32() {
        for bits in 0..=u32::MAX {
            let value = f32::from_bits(bits);
            assert_eq!(from_f32(value), half::bf16::from_f32(value).to_bits(), "{bits:#010x}");
        }
    }

    /// Every encoding against `torch.float8_e4m3fn`, captured as f32 bits so
    /// the oracle is PyTorch's table rather than a second copy of the formula
    /// this file implements.
    #[rustfmt::skip]
    const TORCH_FLOAT8_E4M3FN: [u32; 256] = [
        0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
        0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
        0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
        0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
        0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
        0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
        0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
        0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
        0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
        0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
        0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
        0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
        0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
        0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
        0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
        0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7ff00000,
        0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
        0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
        0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
        0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
        0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
        0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
        0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
        0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
        0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
        0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
        0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
        0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
        0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
        0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
        0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
        0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0xfff00000,
    ];

    #[test]
    fn fp8_decodes_the_whole_table_as_torch_does() {
        for (byte, &expected) in TORCH_FLOAT8_E4M3FN.iter().enumerate() {
            let decoded = fp8_e4m3_to_f32(byte as u8);
            let expected = f32::from_bits(expected);
            if expected.is_nan() {
                assert!(decoded.is_nan(), "{byte:#04x} should be NaN, got {decoded}");
            } else {
                assert_eq!(
                    decoded.to_bits(),
                    expected.to_bits(),
                    "{byte:#04x}: {decoded} is not {expected}"
                );
            }
        }
    }
}
