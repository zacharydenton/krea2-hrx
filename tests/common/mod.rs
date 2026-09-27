//! Helpers shared by the integration tests. Each test binary uses a subset.
#![allow(dead_code)]

use std::path::Path;

/// Every element finite and within `abs + rel * |expected|` of its reference.
pub fn close(actual: &[f64], expected: &[f64], abs: f64, rel: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= abs + rel * b.abs(),
            "element {i}: {a} vs {b}"
        );
    }
}

/// A safetensors file: the header's length, the header, then `payload`.
pub fn safetensors(path: &Path, header: &str, payload: &[u8]) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).expect("writing the safetensors fixture");
}
