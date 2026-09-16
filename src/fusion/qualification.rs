//! Local, content-bound evidence for automatic fusion backend selection.
use hrx::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const PREFIX: &str = "txtfusion.layerwise_blocks.0.mlp.up";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub bias: bool,
}
impl Shape {
    pub fn name(&self) -> String {
        format!("{}-{}-{}-{}", self.m, self.k, self.n, self.bias)
    }
    pub fn padded_m(&self) -> usize {
        self.m.div_ceil(512) * 512
    }
    pub fn storage_bytes(&self) -> Option<usize> {
        if self.m == 0
            || self.k == 0
            || self.n == 0
            || !self.k.is_multiple_of(64)
            || !self.n.is_multiple_of(128)
        {
            return None;
        }
        let m = self.m.checked_add(511)?.checked_div(512)?.checked_mul(512)?;
        m.checked_mul(self.k)?
            .checked_mul(2)?
            .checked_add(self.n.checked_mul(self.k)?.checked_mul(2)?)?
            .checked_add(m.checked_mul(self.n)?.checked_mul(4)?)?
            .checked_add(self.m.checked_mul(self.n)?.checked_mul(2)?)?
            .checked_add(self.n.checked_mul(2)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub shape: Shape,
    pub checkpoint: String,
    pub weights: String,
    pub bias: String,
    pub input: String,
}
impl Case {
    pub fn read(directory: &Path) -> Result<Self> {
        let case: Self = serde_json::from_slice(&fs::read(directory.join("case.json"))?)?;
        for (name, digest, bytes) in [
            (
                "weights.bin",
                &case.weights,
                case.shape.n.checked_mul(case.shape.k).and_then(|v| v.checked_mul(2)),
            ),
            (
                "input.bin",
                &case.input,
                case.shape.m.checked_mul(case.shape.k).and_then(|v| v.checked_mul(2)),
            ),
            (
                "bias.bin",
                &case.bias,
                Some(if case.shape.bias {
                    case.shape
                        .n
                        .checked_mul(2)
                        .ok_or_else(|| Error::Message("shape overflow".into()))?
                } else {
                    0
                }),
            ),
        ] {
            let path = directory.join(name);
            if bytes != Some(fs::metadata(&path)?.len() as usize)
                || hrx::bundle::file_digest(&path)? != *digest
            {
                return Err(Error::Message(format!("captured {name} changed")));
            }
        }
        if case.shape.storage_bytes().is_none() {
            return Err(Error::Unsupported("unsupported fusion shape".into()));
        }
        Ok(case)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Samples {
    pub gpu: Vec<f64>,
    pub npu: Vec<f64>,
    pub correct: bool,
}
fn percentile(values: &[f64], p: usize) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(hrx::benchmark::percentile(&sorted, p))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationEvidence {
    pub gpu_seconds: Vec<f64>,
    pub npu_seconds: Vec<f64>,
    /// Hashes of the immutable reference fixture files used by the existing gate.
    pub reference_files: BTreeMap<String, String>,
    pub relative_rms_loss_db: f64,
    pub image_psnr_loss_db: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub case: Case,
    pub implementation: String,
    /// Content identity of the compiler used to produce these artifacts.
    pub toolchain: String,
    pub machine: BTreeMap<String, String>,
    pub artifact: PathBuf,
    pub image: String,
    pub instructions: String,
    pub epilogue: PathBuf,
    pub epilogue_digest: String,
    pub processes: Vec<Samples>,
    pub generation: Option<GenerationEvidence>,
}
impl Record {
    pub fn qualified(&self) -> bool {
        if self.schema != 1 || self.processes.len() != 5 || !valid_digest(&self.toolchain) {
            return false;
        }
        let mut gpu50 = Vec::new();
        let mut npu50 = Vec::new();
        let mut gpu95 = Vec::new();
        let mut npu95 = Vec::new();
        for run in &self.processes {
            if !run.correct || run.gpu.len() != 100 || run.npu.len() != 100 {
                return false;
            }
            let (Some(g), Some(n), Some(gt), Some(nt)) = (
                percentile(&run.gpu, 50),
                percentile(&run.npu, 50),
                percentile(&run.gpu, 95),
                percentile(&run.npu, 95),
            ) else {
                return false;
            };
            gpu50.push(g);
            npu50.push(n);
            gpu95.push(gt);
            npu95.push(nt);
        }
        if percentile(&npu50, 50).unwrap() > 0.95 * percentile(&gpu50, 50).unwrap()
            || percentile(&npu95, 50).unwrap() > percentile(&gpu95, 50).unwrap()
        {
            return false;
        }
        let Some(generation) = &self.generation else {
            return false;
        };
        let required = [
            "job.json",
            "noise.npy",
            "text.npy",
            "bf16.npy",
            "bf16.png",
            "w8a8.npy",
            "w8a8.png",
        ];
        if generation.reference_files.len() != required.len()
            || required.iter().any(|name| {
                generation.reference_files.get(*name).is_none_or(|hash| {
                    hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit())
                })
            })
            || generation.gpu_seconds.len() != 5
            || generation.npu_seconds.len() != 5
            || !generation.relative_rms_loss_db.is_finite()
            || generation.relative_rms_loss_db > 0.1
            || !generation.image_psnr_loss_db.is_finite()
            || generation.image_psnr_loss_db > 0.1
        {
            return false;
        }
        matches!((percentile(&generation.gpu_seconds,50), percentile(&generation.npu_seconds,50)),
            (Some(g),Some(n)) if n <= 1.05*g)
    }
    pub fn verify(
        &self,
        shape: &Shape,
        checkpoint: &str,
        weights: &str,
        bias: &str,
    ) -> Result<()> {
        self.verify_identity(
            shape,
            checkpoint,
            weights,
            bias,
            (&implementation(), &machine()?),
        )?;
        self.verify_artifacts()
    }
    fn verify_identity(
        &self,
        shape: &Shape,
        checkpoint: &str,
        weights: &str,
        bias: &str,
        current: (&str, &BTreeMap<String, String>),
    ) -> Result<()> {
        let (source, machine) = current;
        if self.schema != 1
            || &self.case.shape != shape
            || self.case.checkpoint != checkpoint
            || self.case.weights != weights
            || self.case.bias != bias
            || self.implementation != source
            || &self.machine != machine
            || !valid_digest(&self.toolchain)
        {
            return Err(Error::Message("NPU qualification identity changed".into()));
        }
        Ok(())
    }
    fn verify_artifacts(&self) -> Result<()> {
        for (path, expected) in [
            (self.artifact.join("x.xclbin"), &self.image),
            (self.artifact.join("x.bin"), &self.instructions),
            (self.epilogue.clone(), &self.epilogue_digest),
        ] {
            if hrx::bundle::file_digest(&path)? != *expected {
                return Err(Error::Message(format!(
                    "NPU artifact changed: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

fn valid_digest(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn implementation() -> String {
    hrx::bundle::digest(
        concat!(
            include_str!("../../native/npu/gemm.py"),
            include_str!("../../native/npu/mm.cc"),
            include_str!("../../native/npu/zero.cc"),
            include_str!("npu.rs"),
            include_str!("mod.rs"),
            include_str!("qualification.rs"),
            include_str!("../models/graph.rs"),
            include_str!("../ops/mod.rs"),
            include_str!("../pipeline/mod.rs"),
            include_str!("../../Cargo.lock"),
            include_str!("../../examples/fusion_worker.rs"),
            include_str!("../../scripts/qualify-fusion.py"),
            include_str!("../../tests/unquantized_parity.rs"),
            include_str!("../../kernels/native/fusion_epilogue.loom")
        )
        .as_bytes(),
    )
}

/// Read stable capability, driver, firmware and power identities without opening devices.
/// Missing information prevents automatic qualification rather than weakening its key.
pub fn machine() -> Result<BTreeMap<String, String>> {
    for key in [
        "HRX_RUNTIME_DIR",
        "HRX_NPU_RUNTIME_DIR",
        "HRX_LOOM_LIBRARY",
        "HRX_BUNDLE_MANIFEST",
        "HRX_NPU_BUNDLE_MANIFEST",
    ] {
        if std::env::var_os(key).is_some() {
            return Err(Error::Unsupported(format!(
                "qualification requires pinned runtimes, unset {key}"
            )));
        }
    }
    let mut values = BTreeMap::new();
    for path in [
        "/proc/sys/kernel/osrelease",
        "/sys/class/accel/accel0/device/vendor",
        "/sys/class/accel/accel0/device/device",
        "/sys/class/accel/accel0/device/fw_version",
        "/sys/class/accel/accel0/device/vbnv",
        "/sys/module/amdgpu/srcversion",
        "/sys/module/amdxdna/srcversion",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
    ] {
        values.insert(path.into(), fs::read_to_string(path)?.trim().to_owned());
    }
    let mut gpus = 0;
    for entry in fs::read_dir("/sys/class/drm")? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name
            .strip_prefix("card")
            .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let path = entry.path().join("device");
        if fs::read_to_string(path.join("vendor")).ok().is_none_or(|s| s.trim() != "0x1002") {
            continue;
        }
        for field in ["vendor", "device", "revision", "power_dpm_force_performance_level"] {
            values.insert(
                format!("{name}/{field}"),
                fs::read_to_string(path.join(field))?.trim().into(),
            );
        }
        gpus += 1;
    }
    if gpus == 0 {
        return Err(Error::Unsupported("missing AMD GPU identity".into()));
    }
    values.insert("gpu_bundle".into(), hrx::bundle::default_manifest()?.archive_sha256);
    values.insert("npu_bundle".into(), hrx::npu::provision::default_manifest()?.archive_sha256);
    Ok(values)
}

pub fn directory() -> Result<PathBuf> {
    Ok(std::env::var_os("KREA2_FUSION_PROFILES")
        .map(PathBuf::from)
        .unwrap_or(hrx::bundle::cache_root()?.join("krea2-fusion")))
}
pub fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().ok_or_else(|| Error::Message("missing record parent".into()))?;
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bad_shapes_and_invalid_measurements_cannot_qualify() {
        for m in [0, usize::MAX] {
            assert!(Shape { m, k: 2560, n: 10240, bias: false }.storage_bytes().is_none());
        }
        assert!(percentile(&[f64::NAN], 50).is_none());
        assert!(percentile(&[0.0], 50).is_none());
        assert_eq!(percentile(&[1., 2., 3., 4., 5.], 95), Some(5.));
    }
    fn passing() -> Record {
        Record {
            schema: 1,
            case: Case {
                shape: Shape { m: 17, k: 64, n: 128, bias: true },
                checkpoint: "c".repeat(64),
                weights: "a".repeat(64),
                bias: "b".repeat(64),
                input: "d".repeat(64),
            },
            implementation: "source".into(),
            toolchain: "e".repeat(64),
            machine: BTreeMap::from([("firmware".into(), "1".into())]),
            artifact: PathBuf::new(),
            image: String::new(),
            instructions: String::new(),
            epilogue: PathBuf::new(),
            epilogue_digest: String::new(),
            processes: vec![
                Samples { gpu: vec![1.0; 100], npu: vec![0.9; 100], correct: true };
                5
            ],
            generation: Some(GenerationEvidence {
                gpu_seconds: vec![10.0; 5],
                npu_seconds: vec![10.4; 5],
                reference_files: [
                    "job.json",
                    "noise.npy",
                    "text.npy",
                    "bf16.npy",
                    "bf16.png",
                    "w8a8.npy",
                    "w8a8.png",
                ]
                .into_iter()
                .map(|name| (name.into(), "f".repeat(64)))
                .collect(),
                relative_rms_loss_db: 0.05,
                image_psnr_loss_db: 0.05,
            }),
        }
    }
    #[test]
    fn automatic_selection_requires_complete_latency_and_quality_evidence() {
        let valid = passing();
        assert!(valid.qualified());
        let mut bad = valid.clone();
        bad.processes.pop();
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.processes[0].gpu.pop();
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.processes[0].correct = false;
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.processes[0].npu[0] = f64::INFINITY;
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        for run in &mut bad.processes {
            run.npu.fill(0.951);
        }
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        for run in &mut bad.processes {
            run.npu[90..].fill(1.01);
        }
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.generation = None;
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.generation.as_mut().unwrap().npu_seconds.fill(10.51);
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.generation.as_mut().unwrap().relative_rms_loss_db = 0.101;
        assert!(!bad.qualified());
        let mut bad = valid.clone();
        bad.generation.as_mut().unwrap().image_psnr_loss_db = f64::NAN;
        assert!(!bad.qualified());
        let mut bad = valid;
        bad.generation.as_mut().unwrap().reference_files.remove("text.npy");
        assert!(!bad.qualified());
    }
    #[test]
    fn stale_identities_and_modified_artifacts_are_rejected_without_device_access() {
        let valid = passing();
        let check = |record: &Record| {
            record.verify_identity(
                &valid.case.shape,
                &valid.case.checkpoint,
                &valid.case.weights,
                &valid.case.bias,
                (&valid.implementation, &valid.machine),
            )
        };
        assert!(check(&valid).is_ok());
        let mut bad = valid.clone();
        bad.case.shape.m += 1;
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.case.checkpoint.push('0');
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.case.weights.push('0');
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.case.bias.push('0');
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.implementation.push('0');
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.machine.insert("firmware".into(), "2".into());
        assert!(check(&bad).is_err());
        let mut bad = valid.clone();
        bad.machine.insert("power".into(), "balanced".into());
        assert!(check(&bad).is_err());
        let directory = tempfile::tempdir().unwrap();
        let mut record = valid;
        record.artifact = directory.path().into();
        record.epilogue = directory.path().join("epilogue.hsaco");
        record.image = hrx::bundle::digest(b"image");
        record.instructions = hrx::bundle::digest(b"instructions");
        record.epilogue_digest = hrx::bundle::digest(b"epilogue");
        fs::write(record.artifact.join("x.xclbin"), b"image").unwrap();
        fs::write(record.artifact.join("x.bin"), b"instructions").unwrap();
        fs::write(&record.epilogue, b"epilogue").unwrap();
        assert!(record.verify_artifacts().is_ok());
        for path in [
            record.artifact.join("x.xclbin"),
            record.artifact.join("x.bin"),
            record.epilogue.clone(),
        ] {
            let original = fs::read(&path).unwrap();
            fs::write(&path, b"changed").unwrap();
            assert!(record.verify_artifacts().is_err());
            fs::write(path, original).unwrap();
        }
    }
}
