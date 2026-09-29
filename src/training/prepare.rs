//! Model-bound disk caches; GPU components are loaded one phase at a time.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{
    PreparedDataset, TrainConfig,
    dataset::{self, Sample},
    vae::Encoder,
};
use crate::checkpoint::{Checkpoint, DType};
use crate::lora::{SavedTensor, io, save_tensors};
use crate::models::{Models, hub};
use crate::numerics::from_f32;
use crate::ops::{Ops, Tensor};
use crate::{Error, Result};
use hrx::{BufferPool, Stream};

pub(crate) fn components(c: &TrainConfig) -> Result<(PathBuf, PathBuf)> {
    let text = match &c.text_encoder {
        Some(p) => p.clone(),
        None => hub::file("text_encoders/qwen3vl_4b_bf16.safetensors", c.offline)?,
    };
    let vae = match &c.vae {
        Some(p) => p.clone(),
        None => hub::file("vae/qwen_image_vae.safetensors", c.offline)?,
    };
    Ok((text, vae))
}

pub(crate) fn software_hash() -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(preprocessing_hash().as_bytes());
    for source in [
        include_str!("trainer.rs"),
        include_str!("model.rs"),
        include_str!("optimizer.rs"),
        include_str!("ops.rs"),
        include_str!("config.rs"),
        include_str!("../lora.rs"),
        include_str!("../pipeline/mod.rs"),
        include_str!("../pipeline/schedule.rs"),
    ] {
        hash.update(source.as_bytes());
    }
    for (name, source) in
        crate::kernels::sources::AUXILIARY.iter().chain(crate::kernels::sources::BLOCK)
    {
        hash.update(name.as_bytes());
        hash.update(source.as_bytes());
    }
    hash.finalize().to_hex().to_string()
}

// Cache identity covers preprocessing only. Backward and optimizer changes must
// invalidate resumable state without forcing unchanged images/text to be encoded again.
fn preprocessing_hash() -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    for source in [
        include_str!("dataset.rs"),
        include_str!("vae.rs"),
        include_str!("prepare.rs"),
        include_str!("../models/graph.rs"),
        include_str!("../models/weights.rs"),
        include_str!("../ops/mod.rs"),
        include_str!("../ops/tensor.rs"),
        include_str!("../numerics.rs"),
        include_str!("../tokenizer.rs"),
        include_str!("../kernels/mod.rs"),
        include_str!("../../Cargo.lock"),
    ] {
        hash.update(source.as_bytes());
    }
    for (name, source) in crate::kernels::sources::AUXILIARY.iter().filter(|(name, _)| {
        (!name.starts_with("train_") || *name == "train_stride_two")
            && *name != "lora_transport"
    }) {
        hash.update(name.as_bytes());
        hash.update(source.as_bytes());
    }
    hash.finalize().to_hex().to_string()
}

pub(crate) fn fingerprint(
    c: &TrainConfig,
    d: &PreparedDataset,
    text: &Path,
    vae: &Path,
) -> Result<String> {
    let identities =
        [dataset::file_hash(&c.model)?, dataset::file_hash(text)?, dataset::file_hash(vae)?];
    let mut h = blake3::Hasher::new();
    h.update(b"krea2-prepared-v1");
    h.update(&serde_json::to_vec(&identities).map_err(io)?);
    h.update(include_bytes!("../../assets/tokenizer.json"));
    h.update(preprocessing_hash().as_bytes());
    h.update(crate::kernels::compiler(None)?.identity().as_bytes());
    for s in &d.samples {
        h.update(s.key.as_bytes());
    }
    Ok(h.finalize().to_hex().to_string())
}

/// Write JSON atomically in the same directory as its destination.
pub(crate) fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(value).map_err(io)?;
    let mut f =
        std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(io)?;
    let result = (|| {
        f.write_all(&bytes).map_err(io)?;
        f.sync_all().map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    result
}

pub(crate) fn cache_path(c: &TrainConfig, s: &Sample, kind: &str) -> PathBuf {
    c.output.join("cache").join(format!("{}.{kind}.safetensors", s.key))
}

fn valid_cache(path: &Path, name: &str, cols: usize, rows: Option<usize>) -> bool {
    Checkpoint::open(path)
        .and_then(|f| {
            let t = f.get(name)?;
            if t.dtype != DType::BF16
                || t.shape.len() != 2
                || t.shape[1] != cols
                || t.shape[0] == 0
                || rows.is_some_and(|r| t.shape[0] != r)
            {
                return Err(Error::invalid("invalid prepared tensor"));
            }
            Ok(())
        })
        .is_ok()
}

fn save_bf16(path: &Path, name: &str, tensor: &Tensor, stream: &mut Stream) -> Result<()> {
    let bits = tensor.download(stream)?;
    save_tensors(
        path,
        [(
            name.into(),
            SavedTensor {
                dtype: "BF16",
                shape: vec![tensor.rows(), tensor.cols()],
                bytes: bits.into_iter().flat_map(u16::to_le_bytes).collect(),
            },
        )]
        .into(),
        BTreeMap::new(),
    )
}

/// Prepare posterior moments and frozen RAW text conditioning, then release all models.
pub fn prepare(c: &TrainConfig) -> Result<PreparedDataset> {
    let mut data = PreparedDataset::scan(c)?;
    let (text, vae) = components(c)?;
    // Check RAW block storage before uploading the auxiliary graph.
    let raw = Checkpoint::open(&c.model)?;
    if raw.get("blocks.0.attn.wq.weight")?.dtype != DType::BF16 {
        return Err(Error::invalid("training requires an original-basis RAW BF16 checkpoint"));
    }
    data.fingerprint = fingerprint(c, &data, &text, &vae)?;
    let manifest = c.output.join("prepared.json");
    if manifest.is_file() {
        let existing: PreparedDataset =
            serde_json::from_slice(&std::fs::read(&manifest).map_err(io)?).map_err(io)?;
        if existing.fingerprint != data.fingerprint {
            return Err(Error::invalid(
                "prepared cache belongs to different data/models/software; choose a fresh output directory",
            ));
        }
    }
    std::fs::create_dir_all(c.output.join("cache")).map_err(io)?;
    // Publish the identity before any entries; individual tensors are atomically replaced.
    write_json(&manifest, &data)?;
    data.contact_sheet(&c.output.join("crops.png"))?;
    let need_latents = data.samples.iter().any(|s| {
        !valid_cache(
            &cache_path(c, s, "posterior"),
            "posterior",
            32,
            Some(s.width / 8 * (s.height / 8)),
        )
    });
    if need_latents {
        let budget = hrx::residency::ResidencyManager::new(c.memory_gib * (1usize << 30))?;
        let mut stream = Stream::open()?.with_memory_budget(budget.budget());
        let ops = Ops::new(BufferPool::new());
        let encoder = Encoder::load(&mut stream, &vae)?;
        for (index, s) in data.samples.iter().enumerate() {
            let path = cache_path(c, s, "posterior");
            if valid_cache(&path, "posterior", 32, Some(s.width / 8 * (s.height / 8))) {
                continue;
            }
            eprintln!(
                "encoding image {}/{}: {}",
                index + 1,
                data.samples.len(),
                s.image.display()
            );
            let rgb = dataset::crop(s)?;
            let pixels: Vec<u16> =
                rgb.as_raw().iter().map(|v| from_f32(f32::from(*v) / 127.5 - 1.0)).collect();
            let input =
                Tensor::from_slice(ops.pool(), &mut stream, &pixels, s.width * s.height, 3)?;
            let posterior = encoder.encode(&ops, &mut stream, &input, s.height, s.width)?;
            save_bf16(&path, "posterior", &posterior, &mut stream)?;
        }
        stream.synchronize()?;
    }
    let need_text = data
        .samples
        .iter()
        .any(|s| !valid_cache(&cache_path(c, s, "text"), "conditioning", 6144, None));
    if need_text {
        let budget = hrx::residency::ResidencyManager::new(c.memory_gib * (1usize << 30))?;
        let mut stream = Stream::open()?.with_memory_budget(budget.budget());
        let models = Models::load_parts(&mut stream, &c.model, Some(&text), None, None, None)?;
        for (index, s) in data.samples.iter().enumerate() {
            let path = cache_path(c, s, "text");
            if valid_cache(&path, "conditioning", 6144, None) {
                continue;
            }
            let ids = models.tokenizer.prompt(&s.caption)?;
            if ids.len() > 512 {
                return Err(Error::invalid(format!(
                    "{}: caption exceeds 512 tokens",
                    s.image.display()
                )));
            }
            eprintln!("encoding caption {}/{}", index + 1, data.samples.len());
            let taps = models.encode(&mut stream, &ids)?;
            let conditioning = models.text_fusion(&mut stream, &taps)?;
            save_bf16(&path, "conditioning", &conditioning, &mut stream)?;
        }
        stream.synchronize()?;
    }
    Ok(data)
}

/// Verify cache identity and completeness without GPU allocation.
pub(crate) fn load(c: &TrainConfig) -> Result<PreparedDataset> {
    let path = c.output.join("prepared.json");
    let data: PreparedDataset =
        serde_json::from_slice(&std::fs::read(path).map_err(io)?).map_err(io)?;
    let current = PreparedDataset::scan(c)?;
    let (text, vae) = components(c)?;
    if data.version != 1
        || data.samples != current.samples
        || data.fingerprint != fingerprint(c, &current, &text, &vae)?
    {
        return Err(Error::invalid(
            "stale prepared dataset; rerun preparation in a fresh output directory",
        ));
    }
    for s in &data.samples {
        if !valid_cache(
            &cache_path(c, s, "posterior"),
            "posterior",
            32,
            Some(s.width / 8 * (s.height / 8)),
        ) || !valid_cache(&cache_path(c, s, "text"), "conditioning", 6144, None)
        {
            return Err(Error::invalid("incomplete prepared dataset; run prepare first"));
        }
    }
    Ok(data)
}
