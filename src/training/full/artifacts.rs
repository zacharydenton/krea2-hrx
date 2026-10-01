//! Immutable RAW model exports, independent retention and bounded offline averaging.
use super::checkpoint::{publish, require_free, validate_values, write_header};
use crate::{
    Error, Result,
    lora::io,
    training::{TrainConfig, TrainingMode, prepare},
};
use hrx::artifacts::safetensors::{DType, Entry, FileView};
use rustix::fs::{FlockOperation, flock};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

// 16 MiB input + 16 MiB accumulator + <=16 MiB output. Headers are bounded separately.
const ELEMENTS: usize = 4 * 1024 * 1024;
const HEADER_LIMIT: usize = 1 << 20;
const MANIFEST: &str = "artifact.json";

/// Whole-file identity, including the safetensors header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Digest {
    /// Exact file length.
    pub bytes: u64,
    /// BLAKE3 of all file bytes.
    pub blake3: String,
}

pub(crate) struct DigestWriter {
    file: File,
    hash: blake3::Hasher,
    bytes: u64,
}
impl DigestWriter {
    pub(crate) fn new(file: File) -> Self {
        Self { file, hash: blake3::Hasher::new(), bytes: 0 }
    }
    pub(crate) fn finish(self) -> Result<Digest> {
        self.file.sync_all().map_err(io)?;
        Ok(Digest { bytes: self.bytes, blake3: self.hash.finalize().to_hex().to_string() })
    }
}
impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// Artifact contents; only `Resume` includes optimizer and sampler state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Complete update-boundary training checkpoint.
    Resume,
    /// Unmodified inference model.
    Model,
    /// Inference model averaged from compatible saved snapshots.
    Average,
}

/// Offline averaging over saved snapshots, never implicit per-step EMA.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Averaging {
    /// Equal weight per selected snapshot.
    Uniform,
    /// Initialize from the first snapshot, then decay by actual optimizer-step gaps.
    Exponential {
        /// Half-life measured in optimizer updates, finite and positive.
        half_life_steps: f64,
    },
}

/// One source of an export or average.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    /// Source directory at creation time; the output is self-contained.
    pub path: PathBuf,
    /// Completed optimizer updates in the source.
    pub step: usize,
    /// Source file identity.
    pub model: Digest,
}

/// Provenance sufficient to evaluate a standalone model without the original run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Artifact schema version.
    pub version: u32,
    /// Whether this artifact can resume training.
    pub kind: Kind,
    /// Identity of the originating run, preserved through export and averaging.
    pub run_id: String,
    /// Prepared-data fingerprint binding base weights, encoders and dataset.
    pub training_fingerprint: String,
    /// Training source identity.
    pub software: String,
    /// Training settings, including RAW component paths and evaluation prompts.
    pub config: TrainConfig,
    /// Last completed optimizer update represented by the artifact.
    pub step: usize,
    /// Manual retention protection.
    pub pinned: bool,
    /// Actual stored model file identity.
    pub model: Digest,
    /// Inputs used for this export or average.
    pub sources: Vec<Source>,
    /// Set only for averaged artifacts.
    pub averaging: Option<Averaging>,
}
impl Manifest {
    pub(crate) fn training(
        config: &TrainConfig,
        step: usize,
        fingerprint: &str,
        software: &str,
        model: Digest,
        kind: Kind,
    ) -> Result<Self> {
        let identity = serde_json::to_vec(&(config, fingerprint)).map_err(io)?;
        Ok(Self {
            version: 1,
            kind,
            run_id: blake3::hash(&identity).to_hex().to_string(),
            training_fingerprint: fingerprint.into(),
            software: software.into(),
            config: config.clone(),
            step,
            pinned: false,
            model,
            sources: Vec::new(),
            averaging: None,
        })
    }
    pub(crate) fn write(&self, directory: &Path) -> Result<()> {
        prepare::write_json(&directory.join(MANIFEST), self)
    }
}

/// A readable artifact plus a shared directory lock preventing managed pruning.
pub struct Artifact {
    path: PathBuf,
    manifest: Manifest,
    _lock: File,
}
impl Artifact {
    /// Open a native full checkpoint or a standalone exported model.
    pub fn open(path: &Path) -> Result<Self> {
        let path = std::fs::canonicalize(path).map_err(io)?;
        let lock = File::open(&path).map_err(io)?;
        if !lock.metadata().map_err(io)?.is_dir() {
            return Err(Error::invalid("checkpoint must be a directory"));
        }
        flock(&lock, FlockOperation::LockShared).map_err(io)?;
        let manifest = read_manifest(&path)?;
        let length = std::fs::metadata(path.join("model.safetensors")).map_err(io)?.len();
        if length != manifest.model.bytes {
            return Err(Error::invalid("artifact model length mismatch"));
        }
        if manifest.kind == Kind::Resume
            && (!path.join("state.json").is_file()
                || !path.join("optimizer.safetensors").is_file())
        {
            return Err(Error::invalid("incomplete resumable checkpoint"));
        }
        Ok(Self { path, manifest, _lock: lock })
    }
    /// Validated metadata; payload checksums are verified while exporting/averaging.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    /// Canonical artifact directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
    fn source(&self) -> Source {
        Source {
            path: self.path.clone(),
            step: self.manifest.step,
            model: self.manifest.model.clone(),
        }
    }
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let m: Manifest =
        serde_json::from_slice(&std::fs::read(path.join(MANIFEST)).map_err(io)?).map_err(io)?;
    if m.version != 1
        || m.config.mode != TrainingMode::Full
        || m.run_id.is_empty()
        || m.training_fingerprint.is_empty()
        || m.model.blake3.len() != 64
        || !m.model.blake3.bytes().all(|b| b.is_ascii_hexdigit())
        || (m.kind == Kind::Average) != m.averaging.is_some()
    {
        return Err(Error::invalid("unsupported or invalid full-model artifact manifest"));
    }
    Ok(m)
}

/// A cheap listing entry; listing never hashes model payloads.
#[derive(Debug, Serialize)]
pub struct Summary {
    /// Artifact directory.
    pub path: PathBuf,
    /// Model, averaged model, or resumable checkpoint.
    pub kind: Kind,
    /// Completed optimizer updates.
    pub step: usize,
    /// Protected from automatic retention.
    pub pinned: bool,
    /// Model payload bytes; shared files can occupy less physical disk collectively.
    pub model_bytes: u64,
}

/// List completed native full artifacts under a run's checkpoints, snapshots and exports.
pub fn list(run: &Path) -> Result<Vec<Summary>> {
    let mut result = Vec::new();
    for name in ["checkpoints", "snapshots", "exports"] {
        for path in directories(&run.join(name))? {
            if !path.join(MANIFEST).is_file() {
                continue;
            }
            let a = Artifact::open(&path)?;
            result.push(Summary {
                path: a.path.clone(),
                kind: a.manifest.kind,
                step: a.manifest.step,
                pinned: a.manifest.pinned,
                model_bytes: a.manifest.model.bytes,
            });
        }
    }
    result.sort_by(|a, b| a.step.cmp(&b.step).then(a.path.cmp(&b.path)));
    Ok(result)
}
fn directories(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for item in std::fs::read_dir(root).map_err(io)? {
        let item = item.map_err(io)?;
        if item.file_type().map_err(io)?.is_dir()
            && !item.file_name().to_string_lossy().starts_with('.')
        {
            result.push(item.path());
        }
    }
    Ok(result)
}

/// Pin or unpin an artifact atomically; pinning never changes its model bytes.
pub fn pin(path: &Path, pinned: bool) -> Result<()> {
    let lock = File::open(path).map_err(io)?;
    flock(&lock, FlockOperation::LockExclusive).map_err(io)?;
    let mut m = read_manifest(path)?;
    m.pinned = pinned;
    m.write(path)?;
    lock.sync_all().map_err(io)
}

/// Keep the newest unpinned artifacts, skipping artifacts currently in use.
pub(crate) fn retain(root: &Path, keep: usize, run_id: &str) -> Result<()> {
    let mut candidates = Vec::new();
    for path in directories(root)? {
        if path.join(MANIFEST).is_file() {
            let m = read_manifest(&path)?;
            if !m.pinned && m.run_id == run_id {
                candidates.push((m.step, path));
            }
        }
    }
    candidates.sort();
    let remove = candidates.len().saturating_sub(keep);
    for (_, path) in candidates.into_iter().take(remove) {
        let lock = File::open(&path).map_err(io)?;
        match flock(&lock, FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => {}
            Err(e) if e == rustix::io::Errno::WOULDBLOCK => continue,
            Err(e) => return Err(io(e)),
        }
        let m = read_manifest(&path)?;
        if !m.pinned && m.run_id == run_id {
            std::fs::remove_dir_all(&path).map_err(io)?;
        }
    }
    if root.is_dir() {
        File::open(root).map_err(io)?.sync_all().map_err(io)?;
    }
    Ok(())
}

// HRX validates safetensors offsets/shapes. Drop its mmap immediately after
// copying the small index; payloads use streaming reads with bounded RSS.
struct ModelReader {
    file: File,
    entries: BTreeMap<String, Entry>,
    hash: blake3::Hasher,
    bytes: u64,
}
impl ModelReader {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(io)?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(io)?;
        let n = usize::try_from(u64::from_le_bytes(prefix)).map_err(io)?;
        if n > HEADER_LIMIT {
            return Err(Error::invalid("model header exceeds 1 MiB"));
        }
        let mut header = vec![0; n];
        file.read_exact(&mut header).map_err(io)?;
        // SAFETY: artifact directory locks prevent this application's writers/pruners from modifying the immutable model.
        let mapped = unsafe { FileView::map(path) }?;
        let entries = mapped.entries().clone();
        drop(mapped);
        let mut offset = 0;
        for e in entries.values() {
            if !matches!(e.dtype, DType::BF16 | DType::F32)
                || e.offset != offset
                || e.elements()? == 0
            {
                return Err(Error::invalid("expected canonical BF16/F32 model tensors"));
            }
            offset += e.bytes;
        }
        if entries.is_empty()
            || file.metadata().map_err(io)?.len() != 8 + n as u64 + offset as u64
        {
            return Err(Error::invalid("model length/schema mismatch"));
        }
        let mut hash = blake3::Hasher::new();
        hash.update(&prefix);
        hash.update(&header);
        Ok(Self { file, entries, hash, bytes: 8 + n as u64 })
    }
    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.file.read_exact(bytes).map_err(io)?;
        self.hash.update(bytes);
        self.bytes += bytes.len() as u64;
        Ok(())
    }
    fn verify(&self, expected: &Digest) -> Result<()> {
        if self.bytes != expected.bytes
            || self.hash.finalize().to_hex().as_str() != expected.blake3
        {
            return Err(Error::invalid("artifact model checksum mismatch"));
        }
        Ok(())
    }
    fn check_all(mut self, expected: &Digest) -> Result<()> {
        let entries = self.entries.clone();
        let mut buf = vec![0u8; ELEMENTS * 4];
        for e in entries.values() {
            for start in (0..e.bytes).step_by(buf.len()) {
                let n = (e.bytes - start).min(buf.len());
                self.read(&mut buf[..n])?;
                validate_values(dtype(e.dtype), &buf[..n], false)?;
            }
        }
        self.verify(expected)
    }
}
fn dtype(d: DType) -> &'static str {
    if d == DType::F32 { "F32" } else { "BF16" }
}

/// Export a self-contained model without modifying its resumable source.
pub fn export(source: &Path, output: &Path) -> Result<Manifest> {
    export_impl(source, output, false)
}
/// The trainer just serialized and validated this file; avoid reading it again.
pub(crate) fn export_saved(source: &Path, output: &Path) -> Result<Manifest> {
    export_impl(source, output, true)
}
fn export_impl(source: &Path, output: &Path, trusted: bool) -> Result<Manifest> {
    let a = Artifact::open(source)?;
    if trusted && output.exists() {
        let existing = Artifact::open(output)?;
        let e = existing.manifest();
        let s = a.manifest();
        if e.kind == Kind::Model
            && e.run_id == s.run_id
            && e.software == s.software
            && e.step == s.step
            && e.model == s.model
        {
            return Ok(e.clone());
        }
        return Err(Error::invalid("existing snapshot differs from saved model"));
    }
    if !trusted {
        ModelReader::open(&a.path.join("model.safetensors"))?.check_all(&a.manifest.model)?;
    }
    let mut m = a.manifest.clone();
    m.kind = if m.averaging.is_some() { Kind::Average } else { Kind::Model };
    m.pinned = false;
    if m.averaging.is_none() {
        m.sources = vec![a.source()];
    }
    let parent =
        output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(io)?;
    publish(output, |tmp| {
        let src = a.path.join("model.safetensors");
        let dst = tmp.join("model.safetensors");
        if std::fs::hard_link(&src, &dst).is_err() {
            require_free(tmp, usize::try_from(m.model.bytes).map_err(io)?)?;
            let mut input = File::open(&src).map_err(io)?;
            let mut out = DigestWriter::new(File::create_new(&dst).map_err(io)?);
            let mut buf = vec![0u8; ELEMENTS * 4];
            loop {
                let n = input.read(&mut buf).map_err(io)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n]).map_err(io)?;
            }
            if out.finish()? != m.model {
                return Err(Error::invalid("export copy checksum mismatch"));
            }
        }
        m.write(tmp)
    })?;
    Ok(m)
}

/// Blend one FP32 chunk with finite FP32/BF16 values; used by streaming averaging.
/// `alpha` is the incoming snapshot weight, in `0..=1`; alpha=1 initializes the chunk.
pub fn blend_chunk(acc: &mut [f32], bytes: &[u8], bf16: bool, alpha: f32) -> Result<()> {
    let width = if bf16 { 2 } else { 4 };
    if acc.len().checked_mul(width) != Some(bytes.len()) || !(0.0..=1.0).contains(&alpha) {
        return Err(Error::invalid("averaging chunk dimensions/weight"));
    }
    for (dst, b) in acc.iter_mut().zip(bytes.chunks_exact(width)) {
        let x = if bf16 {
            crate::numerics::to_f32(u16::from_le_bytes(b.try_into().unwrap()))
        } else {
            f32::from_le_bytes(b.try_into().unwrap())
        };
        if !x.is_finite() {
            return Err(Error::invalid("nonfinite averaging input"));
        }
        *dst = if alpha == 1.0 { x } else { *dst * (1.0 - alpha) + x * alpha };
        if !dst.is_finite() {
            return Err(Error::invalid("nonfinite averaging result"));
        }
    }
    Ok(())
}
fn weights(steps: &[usize], method: Averaging) -> Result<Vec<f32>> {
    if steps.len() < 2 || steps.windows(2).any(|p| p[0] >= p[1]) {
        return Err(Error::invalid("averaging needs at least two distinct increasing steps"));
    }
    if let Averaging::Exponential { half_life_steps: h } = method
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(Error::invalid("EMA half-life must be finite and positive"));
    }
    let mut result = vec![1.0];
    for i in 1..steps.len() {
        let alpha = match method {
            Averaging::Uniform => 1.0 / (i + 1) as f64,
            Averaging::Exponential { half_life_steps: h } => {
                -(-std::f64::consts::LN_2 * (steps[i] - steps[i - 1]) as f64 / h).exp_m1()
            }
        };
        if alpha as f32 == 0.0 {
            return Err(Error::invalid("EMA increment is below FP32 precision"));
        }
        result.push(alpha as f32);
    }
    Ok(result)
}

/// Average compatible snapshots offline with <=48 MiB tensor working buffers.
/// Input steps are sorted; outputs preserve each tensor's dtype and shape.
pub fn average(inputs: &[PathBuf], output: &Path, method: Averaging) -> Result<Manifest> {
    if !(2..=16).contains(&inputs.len()) {
        return Err(Error::invalid("choose between 2 and 16 snapshots per average"));
    }
    let mut sources = inputs.iter().map(|p| Artifact::open(p)).collect::<Result<Vec<_>>>()?;
    sources.sort_by_key(|a| a.manifest.step);
    let first = &sources[0].manifest;
    for a in &sources {
        if a.manifest.run_id != first.run_id
            || a.manifest.training_fingerprint != first.training_fingerprint
            || a.manifest.software != first.software
            || a.manifest.averaging.is_some()
        {
            return Err(Error::invalid(
                "average requires raw snapshots from the same training run/software",
            ));
        }
    }
    let alphas = weights(&sources.iter().map(|a| a.manifest.step).collect::<Vec<_>>(), method)?;
    let mut readers = sources
        .iter()
        .map(|a| ModelReader::open(&a.path.join("model.safetensors")))
        .collect::<Result<Vec<_>>>()?;
    let schema = readers[0].entries.clone();
    for r in &readers {
        if r.entries.len() != schema.len()
            || schema.iter().any(|(name, e)| {
                r.entries.get(name).is_none_or(|o| o.shape != e.shape || o.dtype != e.dtype)
            })
        {
            return Err(Error::invalid("averaging tensor schema mismatch"));
        }
    }
    let mut m = sources.last().unwrap().manifest.clone();
    m.kind = Kind::Average;
    m.pinned = false;
    m.averaging = Some(method);
    m.sources = sources.iter().map(Artifact::source).collect();
    let parent =
        output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(io)?;
    require_free(parent, usize::try_from(m.model.bytes).map_err(io)?)?;
    publish(output, |tmp| {
        let mut file =
            DigestWriter::new(File::create_new(tmp.join("model.safetensors")).map_err(io)?);
        let specs: Vec<_> = schema
            .iter()
            .map(|(name, e)| (name.as_str(), dtype(e.dtype), e.shape.as_slice(), e.bytes))
            .collect();
        write_header(&mut file, &specs)?;
        let maximum = schema
            .values()
            .map(|e| e.elements())
            .collect::<hrx::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap()
            .min(ELEMENTS);
        let mut acc = vec![0f32; maximum];
        let mut input = vec![0u8; maximum * 4];
        let mut encoded = vec![0u8; maximum * 4];
        for e in schema.values() {
            let width = e.dtype.bytes().unwrap();
            let count = e.elements()?;
            for start in (0..count).step_by(maximum) {
                let n = (count - start).min(maximum);
                for (r, &alpha) in readers.iter_mut().zip(&alphas) {
                    r.read(&mut input[..n * width])?;
                    blend_chunk(&mut acc[..n], &input[..n * width], width == 2, alpha)?;
                }
                if width == 4 {
                    for (&x, b) in acc[..n].iter().zip(encoded[..n * 4].as_chunks_mut::<4>().0)
                    {
                        b.copy_from_slice(&x.to_le_bytes());
                    }
                } else {
                    for (&x, b) in acc[..n].iter().zip(encoded[..n * 2].as_chunks_mut::<2>().0)
                    {
                        let bits = crate::numerics::from_f32_carrying(x);
                        if !crate::numerics::to_f32(bits).is_finite() {
                            return Err(Error::invalid("BF16 averaging overflow"));
                        }
                        b.copy_from_slice(&bits.to_le_bytes());
                    }
                }
                file.write_all(&encoded[..n * width]).map_err(io)?;
            }
        }
        for (r, a) in readers.iter().zip(&sources) {
            r.verify(&a.manifest.model)?;
        }
        m.model = file.finish()?;
        m.write(tmp)
    })?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;

    fn fixture(root: &Path, step: usize, a: &[f32], b: &[f32]) -> PathBuf {
        let path = root.join(format!("step-{step:06}"));
        std::fs::create_dir_all(&path).unwrap();
        let mut file =
            DigestWriter::new(File::create_new(path.join("model.safetensors")).unwrap());
        write_header(
            &mut file,
            &[("a", "BF16", &[a.len()], a.len() * 2), ("b", "F32", &[b.len()], b.len() * 4)],
        )
        .unwrap();
        for &v in a {
            file.write_all(&half::bf16::from_f32(v).to_bits().to_le_bytes()).unwrap();
        }
        for &v in b {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        Manifest::training(
            &TrainConfig::full_preset(),
            step,
            "data",
            "software",
            file.finish().unwrap(),
            Kind::Model,
        )
        .unwrap()
        .write(&path)
        .unwrap();
        path
    }
    fn values(path: &Path, name: &str) -> Vec<f32> {
        let c = Checkpoint::open(&path.join("model.safetensors")).unwrap();
        let t = c.get(name).unwrap();
        if t.dtype == crate::checkpoint::DType::BF16 {
            t.bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect()
        } else {
            t.bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
        }
    }
    fn change_manifest(path: &Path, f: impl FnOnce(&mut Manifest)) {
        let mut m = read_manifest(path).unwrap();
        f(&mut m);
        m.write(path).unwrap();
    }
    #[test]
    fn uniform_and_gap_aware_ema_preserve_mixed_dtypes_and_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let a = fixture(root, 10, &[0.0, 2.0], &[0.0, 0.125]);
        let b = fixture(root, 20, &[4.0, 6.0], &[4.0, 0.625]);
        let c = fixture(root, 40, &[8.0, 10.0], &[8.0, 1.125]);
        let sources = [c.clone(), a.clone(), b.clone()];
        let out = root.join("uniform");
        let m = average(&sources, &out, Averaging::Uniform).unwrap();
        assert_eq!(values(&out, "a"), vec![4.0, 6.0]);
        assert_eq!(values(&out, "b"), vec![4.0, 0.625]);
        assert_eq!(m.sources.iter().map(|s| s.step).collect::<Vec<_>>(), [10, 20, 40]);
        assert_eq!(m.kind, Kind::Average);
        assert!(!out.join("optimizer.safetensors").exists());
        assert!(
            crate::training::Trainer::resume(&out)
                .err()
                .unwrap()
                .to_string()
                .contains("cannot resume")
        );
        ModelReader::open(&out.join("model.safetensors")).unwrap().check_all(&m.model).unwrap();
        let ema = root.join("ema");
        average(&sources, &ema, Averaging::Exponential { half_life_steps: 10.0 }).unwrap();
        // One half-life at step 20, then two half-lives at step 40.
        assert_eq!(values(&ema, "a"), vec![6.5, 8.5]);
        assert_eq!(values(&ema, "b"), vec![6.5, 0.9375]);
        let exported = root.join("ema-export");
        let e = export(&ema, &exported).unwrap();
        assert_eq!(e.sources.len(), 3);
        assert_eq!(values(&exported, "a"), values(&ema, "a"));
        assert!(average(&[out, b], &root.join("nested"), Averaging::Uniform).is_err());
    }
    #[test]
    fn streaming_crosses_chunk_boundaries_and_handles_tail() {
        let dir = tempfile::tempdir().unwrap();
        let n = ELEMENTS + 3;
        let a = fixture(dir.path(), 1, &vec![2.0; n], &[11.0]);
        let b = fixture(dir.path(), 2, &vec![6.0; n], &[13.0]);
        let out = dir.path().join("average");
        average(&[a, b], &out, Averaging::Uniform).unwrap();
        assert!(values(&out, "a").iter().all(|&v| v == 4.0));
        assert_eq!(values(&out, "b"), [12.0]);
    }
    #[test]
    fn export_is_immutable_independent_and_idempotent_only_for_trainer() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let source = fixture(dir.path(), 1, &[2.0], &[3.0]);
        let out = dir.path().join("export");
        let m = export(&source, &out).unwrap();
        assert_eq!(
            std::fs::metadata(source.join("model.safetensors")).unwrap().ino(),
            std::fs::metadata(out.join("model.safetensors")).unwrap().ino()
        );
        assert_eq!(m.sources[0].step, 1);
        assert!(export(&source, &out).is_err());
        pin(&out, true).unwrap();
        assert!(export_saved(&source, &out).unwrap().pinned);
        change_manifest(&out, |m| m.step = 2);
        assert!(export_saved(&source, &out).is_err());
        std::fs::remove_dir_all(source).unwrap();
        assert_eq!(values(&out, "a"), [2.0]);
        ModelReader::open(&out.join("model.safetensors")).unwrap().check_all(&m.model).unwrap();
    }
    #[test]
    fn retention_respects_pins_readers_run_identity_and_incomplete_writes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("snapshots");
        let pinned = fixture(&root, 1, &[1.0], &[1.0]);
        pin(&pinned, true).unwrap();
        let active = fixture(&root, 2, &[2.0], &[2.0]);
        let lock = Artifact::open(&active).unwrap();
        let stale = fixture(&root, 3, &[3.0], &[3.0]);
        let newest = fixture(&root, 4, &[4.0], &[4.0]);
        let foreign = fixture(&root, 5, &[5.0], &[5.0]);
        change_manifest(&foreign, |m| m.run_id = "another run".into());
        let partial = root.join(".step-000006.tmp");
        std::fs::create_dir(&partial).unwrap();
        retain(&root, 1, &lock.manifest.run_id).unwrap();
        assert!(
            pinned.exists()
                && active.exists()
                && newest.exists()
                && foreign.exists()
                && partial.exists()
        );
        assert!(!stale.exists());
        let run_id = lock.manifest.run_id.clone();
        drop(lock);
        pin(&pinned, false).unwrap();
        retain(&root, 1, &run_id).unwrap();
        assert!(!active.exists() && !pinned.exists());
        assert_eq!(list(dir.path()).unwrap().len(), 2);
    }
    #[test]
    fn rejects_incompatible_schema_run_software_and_duplicate_steps() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let a = fixture(root, 1, &[1.0], &[1.0]);
        let b = fixture(root, 2, &[2.0], &[2.0]);
        let out = root.join("bad");
        assert!(average(&[a.clone(), a.clone()], &out, Averaging::Uniform).is_err());
        for field in ["run_id", "training_fingerprint", "software"] {
            let original = read_manifest(&b).unwrap();
            change_manifest(&b, |m| match field {
                "run_id" => m.run_id.push('x'),
                "software" => m.software.push('x'),
                _ => m.training_fingerprint.push('x'),
            });
            assert!(average(&[a.clone(), b.clone()], &out, Averaging::Uniform).is_err());
            assert!(!out.exists());
            original.write(&b).unwrap();
        }
        let wrong = fixture(root, 3, &[1.0, 2.0], &[3.0]);
        assert!(average(&[a, wrong], &out, Averaging::Uniform).is_err());
    }
    #[test]
    fn corrupt_or_nonfinite_input_cannot_publish_or_damage_existing_outputs() {
        use std::io::{Seek, SeekFrom};
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let a = fixture(root, 1, &[1.0], &[1.0]);
        let b = fixture(root, 2, &[2.0], &[2.0]);
        let previous = root.join("previous");
        export(&a, &previous).unwrap();
        let mut file =
            std::fs::OpenOptions::new().write(true).open(b.join("model.safetensors")).unwrap();
        file.seek(SeekFrom::End(-4)).unwrap();
        file.write_all(&3f32.to_le_bytes()).unwrap();
        file.sync_all().unwrap();
        let out = root.join("bad");
        assert!(
            average(&[a.clone(), b.clone()], &out, Averaging::Uniform)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        assert!(export(&b, &out).is_err());
        assert!(!out.exists());
        let nan = fixture(root, 3, &[1.0], &[f32::NAN]);
        assert!(
            average(&[a, nan.clone()], &out, Averaging::Uniform)
                .unwrap_err()
                .to_string()
                .contains("nonfinite")
        );
        assert!(export(&nan, &out).is_err());
        assert_eq!(values(&previous, "b"), [1.0]);
        assert!(
            directories(root).unwrap().iter().all(|p| !p
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with('.'))
        );
        assert_eq!(std::fs::read_dir(root).unwrap().count(), 4);
    }
    #[test]
    fn numerical_and_manifest_validation() {
        for h in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(weights(&[1, 2], Averaging::Exponential { half_life_steps: h }).is_err());
        }
        assert!(weights(&[2, 1], Averaging::Uniform).is_err());
        assert!(blend_chunk(&mut [0.0], &[], false, 0.5).is_err());
        assert!(blend_chunk(&mut [0.0], &1f32.to_le_bytes(), false, f32::NAN).is_err());
        let mut value = [f32::NAN];
        blend_chunk(&mut value, &2f32.to_le_bytes(), false, 1.0).unwrap();
        assert_eq!(value, [2.0]);
        let dir = tempfile::tempdir().unwrap();
        let a = fixture(dir.path(), 1, &[1.0], &[1.0]);
        change_manifest(&a, |m| m.kind = Kind::Resume);
        assert!(Artifact::open(&a).is_err());
        change_manifest(&a, |m| m.version = 99);
        assert!(Artifact::open(&a).is_err());
    }
}
