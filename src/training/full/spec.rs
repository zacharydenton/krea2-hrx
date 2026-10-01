//! Complete RAW parameter inventory, independent of checkpoint ordering.
use crate::{
    Error, Result,
    checkpoint::{Checkpoint, DType},
    lora::Targets,
};
use std::collections::BTreeMap;

/// Logical shape and optimizer representation of one trainable tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    /// Checkpoint shape, preserved on export.
    pub shape: Vec<usize>,
    /// Number of scalar parameters.
    pub count: usize,
    /// Small parameters retain FP32 values, gradients and moments.
    pub small: bool,
}

/// The complete supported Krea 2 transformer schema (430 learned tensors).
pub fn inventory() -> BTreeMap<String, ParameterSpec> {
    let mut shapes = BTreeMap::new();
    for (name, o, i) in Targets::All.layers() {
        shapes.insert(format!("{name}.weight"), vec![o, i]);
    }
    for block in 0..28 {
        let p = format!("blocks.{block}");
        for (suffix, width) in [
            ("prenorm.scale", 6144),
            ("postnorm.scale", 6144),
            ("attn.qknorm.qnorm.scale", 128),
            ("attn.qknorm.knorm.scale", 128),
            ("mod.lin", 36864),
        ] {
            shapes.insert(format!("{p}.{suffix}"), vec![width]);
        }
    }
    for tower in ["layerwise_blocks", "refiner_blocks"] {
        for block in 0..2 {
            for (suffix, width) in [
                ("prenorm.scale", 2560),
                ("postnorm.scale", 2560),
                ("attn.qknorm.qnorm.scale", 128),
                ("attn.qknorm.knorm.scale", 128),
            ] {
                shapes.insert(format!("txtfusion.{tower}.{block}.{suffix}"), vec![width]);
            }
        }
    }
    for (name, width) in [
        ("first.bias", 6144),
        ("last.linear.bias", 64),
        ("tmlp.0.bias", 6144),
        ("tmlp.2.bias", 6144),
        ("tproj.1.bias", 36864),
        ("txtmlp.1.bias", 6144),
        ("txtmlp.3.bias", 6144),
        ("txtmlp.0.scale", 2560),
        ("last.norm.scale", 6144),
    ] {
        shapes.insert(name.into(), vec![width]);
    }
    shapes.insert("last.modulation.lin".into(), vec![2, 6144]);
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let count = shape.iter().product();
            let small = !name.ends_with(".weight") || count < 4096;
            (name, ParameterSpec { shape, count, small })
        })
        .collect()
}

/// Validate all names, dimensions and floating-point storage before allocation.
pub fn validate(file: &Checkpoint) -> Result<BTreeMap<String, ParameterSpec>> {
    let schema = inventory();
    if file.names().count() != schema.len() {
        return Err(Error::invalid(
            "full tuning requires the complete 430-tensor RAW checkpoint",
        ));
    }
    for name in file.names() {
        let spec = schema.get(name).ok_or_else(|| {
            Error::invalid(format!("unsupported full-training tensor {name}"))
        })?;
        let tensor = file.get(name)?;
        if tensor.shape != spec.shape || !matches!(tensor.dtype, DType::BF16 | DType::F32) {
            return Err(Error::invalid(format!("full-training shape/dtype mismatch: {name}")));
        }
    }
    Ok(schema)
}

/// Persistent device bytes: weights, gradients, moments, scales, and small copies.
pub fn persistent_bytes(schema: &BTreeMap<String, ParameterSpec>) -> usize {
    schema
        .values()
        .map(|s| {
            if s.small {
                s.count * 18
            } else {
                s.count * 6 + s.count.div_ceil(super::numerics::BLOCK) * 8
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_inventory_and_memory_bound() {
        let schema = inventory();
        assert_eq!(schema.len(), 430);
        assert_eq!(schema.values().map(|s| s.count).sum::<usize>(), 12_820_073_036);
        assert_eq!(schema.values().filter(|s| !s.small).count(), 263);
        let gib = persistent_bytes(&schema) as f64 / (1u64 << 30) as f64;
        assert!((72.0..73.0).contains(&gib), "{gib}");
    }
}
