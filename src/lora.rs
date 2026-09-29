//! Original-basis LoRA factors and interoperable safetensors serialization.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::checkpoint::{Checkpoint, DType};
use crate::numerics::{from_f32, to_f32};
use crate::{Error, Result};

/// The eight trainable projections in each main transformer block.
pub const PROJECTIONS: [(&str, usize, usize); 8] = [
    ("attn.wq", 6144, 6144),
    ("attn.wk", 1536, 6144),
    ("attn.wv", 1536, 6144),
    ("attn.gate", 6144, 6144),
    ("attn.wo", 6144, 6144),
    ("mlp.gate", 16384, 6144),
    ("mlp.up", 16384, 6144),
    ("mlp.down", 6144, 16384),
];

/// Which original-basis DiT linear layers receive adapters. Qwen stays frozen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Targets {
    /// The eight projections in each of the 28 main blocks (224 total).
    #[default]
    MainBlocks,
    /// All 264 DiT linears, including text fusion and time/text/image projections.
    All,
}

impl Targets {
    /// Canonical names and `[outputs, inputs]`, in deterministic initialization order.
    pub fn layers(self) -> Vec<(String, usize, usize)> {
        let mut layers = Vec::new();
        for block in 0..28 {
            for (name, outputs, inputs) in PROJECTIONS {
                layers.push((format!("blocks.{block}.{name}"), outputs, inputs));
            }
        }
        if self == Self::All {
            for tower in ["layerwise_blocks", "refiner_blocks"] {
                for block in 0..2 {
                    for (name, _, _) in PROJECTIONS {
                        let (outputs, inputs) = match name {
                            "mlp.gate" | "mlp.up" => (6912, 2560),
                            "mlp.down" => (2560, 6912),
                            _ => (2560, 2560),
                        };
                        layers.push((
                            format!("txtfusion.{tower}.{block}.{name}"),
                            outputs,
                            inputs,
                        ));
                    }
                }
            }
            for (name, outputs, inputs) in [
                ("first", 6144, 64),
                ("last.linear", 64, 6144),
                ("tmlp.0", 6144, 256),
                ("tmlp.2", 6144, 6144),
                ("tproj.1", 36864, 6144),
                ("txtmlp.1", 6144, 2560),
                ("txtmlp.3", 6144, 6144),
                ("txtfusion.projector", 1, 12),
            ] {
                layers.push((name.into(), outputs, inputs));
            }
        }
        layers
    }
}

/// Row-major master factors: A is `[rank, inputs]`, B is `[outputs, rank]`.
#[derive(Clone, Debug)]
pub struct Factors {
    /// Input features.
    pub inputs: usize,
    /// Output features.
    pub outputs: usize,
    /// Low-rank dimension.
    pub rank: usize,
    /// Numerator of the `alpha / rank` multiplier.
    pub alpha: f32,
    /// Down projection in the original model basis.
    pub a: Vec<f32>,
    /// Up projection in the original model basis.
    pub b: Vec<f32>,
}

impl Factors {
    /// Validate shapes and finite values before upload or serialization.
    pub fn validate(&self) -> Result<()> {
        if !(1..=128).contains(&self.rank)
            || self.inputs == 0
            || self.outputs == 0
            || self.rank.checked_mul(self.inputs) != Some(self.a.len())
            || self.rank.checked_mul(self.outputs) != Some(self.b.len())
            || !self.alpha.is_finite()
            || self.alpha <= 0.0
            || self.a.iter().chain(&self.b).any(|v| !to_f32(from_f32(*v)).is_finite())
        {
            return Err(Error::invalid("invalid LoRA dimensions, alpha, or factor values"));
        }
        Ok(())
    }
}

/// A single adapter, keyed by unwrapped native projection names.
#[derive(Clone, Debug, Default)]
pub struct Adapter {
    /// For example, `blocks.0.attn.wq`.
    pub layers: BTreeMap<String, Factors>,
}

fn dimensions(name: &str) -> Result<(usize, usize)> {
    Targets::All
        .layers()
        .into_iter()
        .find(|(key, _, _)| key == name)
        .map(|(_, outputs, inputs)| (outputs, inputs))
        .ok_or_else(|| Error::invalid(format!("unsupported LoRA target {name}")))
}

impl Adapter {
    /// Validate every target, shape and value, including manually constructed adapters.
    pub fn validate(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(Error::invalid("empty LoRA adapter"));
        }
        for (name, factors) in &self.layers {
            if dimensions(name)? != (factors.outputs, factors.inputs) {
                return Err(Error::invalid(format!("LoRA shape mismatch for {name}")));
            }
            factors.validate()?;
        }
        Ok(())
    }

    /// Initialize every main-block projection with Gaussian A and zero B.
    pub fn initialize(rank: usize, alpha: f32, seed: u64) -> Result<Self> {
        Self::initialize_targets(rank, alpha, seed, Targets::MainBlocks)
    }

    /// Initialize the selected target profile; main-block RNG ordering is preserved.
    pub fn initialize_targets(
        rank: usize,
        alpha: f32,
        seed: u64,
        targets: Targets,
    ) -> Result<Self> {
        if !(1..=128).contains(&rank) || !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error::invalid("LoRA rank must be 1..=128 and alpha positive"));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut layers = BTreeMap::new();
        for (name, outputs, inputs) in targets.layers() {
            let a = (0..rank * inputs)
                .map(|_| {
                    let value: f32 = StandardNormal.sample(&mut rng);
                    value / rank as f32
                })
                .collect();
            layers.insert(
                name,
                Factors { inputs, outputs, rank, alpha, a, b: vec![0.0; outputs * rank] },
            );
        }
        Ok(Self { layers })
    }

    /// Load canonical ComfyUI/PEFT factors. Unknown keys are errors, never ignored.
    pub fn load(path: &Path) -> Result<Self> {
        let file = Checkpoint::open(path)?;
        let mut names = BTreeMap::<String, String>::new();
        for key in file.names() {
            let canonical = key.strip_prefix("diffusion_model.").unwrap_or(key);
            let target = canonical
                .strip_suffix(".lora_A.weight")
                .or_else(|| canonical.strip_suffix(".lora_B.weight"))
                .or_else(|| canonical.strip_suffix(".alpha"))
                .ok_or_else(|| Error::invalid(format!("unsupported LoRA tensor {key}")))?;
            dimensions(target)?;
            if names.insert(canonical.to_owned(), key.to_owned()).is_some() {
                return Err(Error::invalid(format!("duplicate LoRA tensor {canonical}")));
            }
        }
        let mut layers = BTreeMap::new();
        for name in names.keys().filter_map(|key| key.strip_suffix(".lora_A.weight")) {
            let (outputs, inputs) = dimensions(name)?;
            let a = file.get(&names[&format!("{name}.lora_A.weight")])?;
            let b_key = names
                .get(&format!("{name}.lora_B.weight"))
                .ok_or_else(|| Error::invalid(format!("missing B factor for {name}")))?;
            let b = file.get(b_key)?;
            if a.shape.len() != 2 || b.shape != [outputs, a.shape[0]] || a.shape[1] != inputs {
                return Err(Error::invalid(format!("LoRA shape mismatch for {name}")));
            }
            let rank = a.shape[0];
            let alpha = match names.get(&format!("{name}.alpha")) {
                Some(key) => {
                    let tensor = file.get(key)?;
                    if !tensor.shape.is_empty() && tensor.shape != [1] {
                        return Err(Error::invalid(format!(
                            "non-scalar LoRA alpha for {name}"
                        )));
                    }
                    let values = floats(tensor)?;
                    if values.len() != 1 {
                        return Err(Error::invalid("invalid LoRA alpha"));
                    }
                    values[0]
                }
                None => rank as f32,
            };
            let factors =
                Factors { inputs, outputs, rank, alpha, a: floats(a)?, b: floats(b)? };
            factors.validate()?;
            layers.insert(name.to_owned(), factors);
        }
        for key in names.keys() {
            let target = key
                .strip_suffix(".lora_A.weight")
                .or_else(|| key.strip_suffix(".lora_B.weight"))
                .or_else(|| key.strip_suffix(".alpha"))
                .expect("validated suffix");
            if !layers.contains_key(target) {
                return Err(Error::invalid(format!("missing A factor for {target}")));
            }
        }
        if layers.is_empty() {
            return Err(Error::invalid("empty LoRA adapter"));
        }
        Ok(Self { layers })
    }

    /// Atomically export BF16 factors with explicit alpha; no base weights are included.
    pub fn save(&self, path: &Path, metadata: BTreeMap<String, String>) -> Result<()> {
        self.validate()?;
        let mut tensors = BTreeMap::new();
        for (name, f) in &self.layers {
            for (suffix, shape, values) in [
                ("lora_A.weight", vec![f.rank, f.inputs], &f.a),
                ("lora_B.weight", vec![f.outputs, f.rank], &f.b),
            ] {
                tensors.insert(
                    format!("diffusion_model.{name}.{suffix}"),
                    SavedTensor {
                        dtype: "BF16",
                        shape,
                        bytes: values.iter().flat_map(|v| from_f32(*v).to_le_bytes()).collect(),
                    },
                );
            }
            tensors.insert(
                format!("diffusion_model.{name}.alpha"),
                SavedTensor {
                    dtype: "F32",
                    shape: vec![],
                    bytes: f.alpha.to_le_bytes().to_vec(),
                },
            );
        }
        save_tensors(path, tensors, metadata)
    }
}

pub(crate) fn floats(t: crate::checkpoint::Tensor<'_>) -> Result<Vec<f32>> {
    match t.dtype {
        DType::BF16 => Ok(t
            .bytes
            .chunks_exact(2)
            .map(|v| to_f32(u16::from_le_bytes([v[0], v[1]])))
            .collect()),
        DType::F32 => Ok(t
            .bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect()),
        _ => Err(Error::invalid("LoRA tensors must be BF16 or F32")),
    }
}

#[derive(Clone)]
pub(crate) struct SavedTensor {
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub(crate) fn io(error: impl std::fmt::Display) -> Error {
    Error::invalid(error.to_string())
}

pub(crate) fn save_tensors(
    path: &Path,
    tensors: BTreeMap<String, SavedTensor>,
    metadata: BTreeMap<String, String>,
) -> Result<()> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!(metadata));
    let mut offset = 0usize;
    for (name, tensor) in &tensors {
        let end = offset
            .checked_add(tensor.bytes.len())
            .ok_or_else(|| Error::invalid("checkpoint too large"))?;
        header.insert(name.clone(), serde_json::json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end]}));
        offset = end;
    }
    let mut bytes = serde_json::to_vec(&header).map_err(io)?;
    bytes.resize(bytes.len().div_ceil(8) * 8, b' ');
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io)?;
    let result = (|| {
        file.write_all(&(bytes.len() as u64).to_le_bytes()).map_err(io)?;
        file.write_all(&bytes).map_err(io)?;
        for tensor in tensors.values() {
            file.write_all(&tensor.bytes).map_err(io)?;
        }
        file.sync_all().map_err(io)?;
        std::fs::rename(&temporary, path).map_err(io)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_target_profile_is_complete_and_preserves_main_initialization() {
        let main = Adapter::initialize(1, 1.0, 37).unwrap();
        let all = Adapter::initialize_targets(1, 1.0, 37, Targets::All).unwrap();
        assert_eq!(main.layers.len(), 224);
        assert_eq!(all.layers.len(), 264);
        all.validate().unwrap();
        for (name, factors) in main.layers {
            assert_eq!(factors.a, all.layers[&name].a);
            assert_eq!(factors.b, all.layers[&name].b);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("all.safetensors");
        all.save(&path, BTreeMap::new()).unwrap();
        let loaded = Adapter::load(&path).unwrap();
        assert_eq!(loaded.layers.len(), 264);
        for (name, factors) in loaded.layers {
            assert_eq!((factors.outputs, factors.inputs), dimensions(&name).unwrap());
            assert_eq!(
                factors.a,
                all.layers[&name].a.iter().map(|v| to_f32(from_f32(*v))).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn export_round_trips_original_basis_factors_and_alpha() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("character.safetensors");
        let f = Factors {
            inputs: 6144,
            outputs: 1536,
            rank: 2,
            alpha: 1.0,
            a: vec![0.125; 2 * 6144],
            b: vec![-0.25; 1536 * 2],
        };
        let adapter = Adapter { layers: [("blocks.0.attn.wk".into(), f)].into() };
        adapter.save(&path, BTreeMap::new()).unwrap();
        let loaded = Adapter::load(&path).unwrap();
        let f = &loaded.layers["blocks.0.attn.wk"];
        assert_eq!(f.alpha, 1.0);
        assert_eq!(f.a, adapter.layers["blocks.0.attn.wk"].a);
        assert_eq!(f.b, adapter.layers["blocks.0.attn.wk"].b);
    }

    #[test]
    fn malformed_or_unpaired_factors_are_rejected() {
        for name in [
            "blocks.28.attn.wq",
            "blocks.00.attn.wq",
            "txtfusion.layerwise_blocks.2.attn.wq",
            "blocks.0.attn.qknorm",
        ] {
            assert!(dimensions(name).is_err());
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.safetensors");
        save_tensors(
            &path,
            [(
                "diffusion_model.blocks.0.attn.wk.lora_B.weight".into(),
                SavedTensor { dtype: "BF16", shape: vec![1536, 1], bytes: vec![0; 1536 * 2] },
            )]
            .into(),
            BTreeMap::new(),
        )
        .unwrap();
        assert!(Adapter::load(&path).unwrap_err().to_string().contains("missing A"));
    }

    #[test]
    fn manually_constructed_adapters_reject_unknown_targets_and_bf16_overflow() {
        let factors = Factors {
            inputs: 6144,
            outputs: 1536,
            rank: 1,
            alpha: 1.0,
            a: vec![0.125; 6144],
            b: vec![0.0; 1536],
        };
        let bad = Adapter { layers: [("blocks.0.typo".into(), factors.clone())].into() };
        assert!(bad.validate().is_err());
        let mut bad = factors;
        bad.a[0] = f32::MAX;
        assert!(bad.a[0].is_finite());
        assert!(bad.validate().is_err());
    }

    #[test]
    fn a_conflicting_temporary_file_is_never_removed_or_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("state.safetensors");
        let temporary = destination.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temporary, b"existing writer").unwrap();
        assert!(save_tensors(&destination, BTreeMap::new(), BTreeMap::new()).is_err());
        assert_eq!(std::fs::read(&temporary).unwrap(), b"existing writer");
        assert!(!destination.exists());
    }
}
