//! Compiling one block bundle: the nine or ten kernels a sequence length needs.
//!
//! A bundle is a directory of HSACOs plus `launch.txt` (the shape the session
//! checks against) and `manifest.json` (each artifact's hash). Its name is a
//! digest over every source, symbol and configuration that went into it, so a
//! bundle is immutable: if the directory exists, it is the right one, and it is
//! usable on a machine with no compiler at all.
//!
//! `scripts/build_kernels.py` writes the same directories from the same rules;
//! either can populate a cache the other reads.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::compile::{digest, Compilation, Lock};
use crate::{compiler, shape, sources, Error, Result, Settings};

/// The metadata version and the shape knobs that are not in `launch.txt`.
/// Changing any of it must change every bundle's name.
const SIGNATURE_PREFIX: &str = "native-kernels-v5:gfx1151:sage-prep-v3:64:vt\n";

/// One kernel to compile: a source, and the file name it takes in the bundle.
struct Job {
    source: String,
    stem: String,
    config: Settings,
}

/// The bundle's shape, which `launch.txt` records and the session re-derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub tokens: i32,
    pub gemm_bits: i32,
    pub attention_bits: i32,
    pub attention_waves: i32,
    pub query_tiles: i32,
    pub capacity: i32,
    pub rows: i32,
    pub m_group: i32,
}

impl Shape {
    /// `attention_bits` is 16 (fp16 QK and PV, ComfyUI's SDPA class) or 4 / 8
    /// (the smoothed SageAttention-style kernels).
    pub fn new(tokens: i32, gemm_bits: i32, attention_bits: i32) -> Result<Shape> {
        if !(16..=16896).contains(&tokens) {
            return Err(Error("tokens must be 16..16896".into()));
        }
        if gemm_bits != 4 && gemm_bits != 8 {
            return Err(Error("GEMM operand width must be 4 or 8".into()));
        }
        if ![4, 8, 16].contains(&attention_bits) {
            return Err(Error("KREA2_ATTN_QK must be 4, 8 or 16".into()));
        }
        let query_tiles =
            if attention_bits == 16 { shape::fp16_query_tiles(tokens) } else { 1 };
        let capacity = if query_tiles == 2 {
            (tokens + 79) / 64 * 64
        } else {
            std::cmp::max((tokens + 47) / 32 * 32, (tokens + 63) / 64 * 64)
        };
        let rows = shape::gemm_rows(tokens, gemm_bits);
        Ok(Shape {
            tokens,
            gemm_bits,
            attention_bits,
            // Long sequences leave less room per wave for the score tile.
            attention_waves: if tokens < 8192 { 8 } else { 4 },
            query_tiles,
            capacity,
            rows,
            m_group: shape::gemm_m_group(tokens, rows),
        })
    }

    /// `KREA2_ATTN_QK` from the environment, 16 when it is unset or empty.
    pub fn from_environment(tokens: i32, gemm_bits: i32) -> Result<Shape> {
        let bits = match std::env::var("KREA2_ATTN_QK") {
            Ok(value) if !value.is_empty() => {
                value.parse().map_err(|_| Error("KREA2_ATTN_QK must be 4, 8 or 16".into()))?
            }
            _ => 16,
        };
        Shape::new(tokens, gemm_bits, bits)
    }

    /// The eleven fields of `launch.txt`, version 5.
    pub fn launch_text(&self) -> String {
        format!(
            "5 {} {} {} {} {} {} {} {} {} {}\n",
            self.tokens,
            self.rows,
            self.m_group,
            self.capacity,
            self.attention_waves,
            shape::gemm_pitch(6144, self.gemm_bits),
            shape::gemm_pitch(16384, self.gemm_bits),
            self.attention_bits,
            self.gemm_bits,
            self.query_tiles,
        )
    }

    fn jobs(&self) -> Vec<Job> {
        let width = format!("i{}", self.gemm_bits);
        let pitch_hidden = shape::gemm_pitch(6144, self.gemm_bits).to_string();
        let pitch_inter = shape::gemm_pitch(16384, self.gemm_bits).to_string();
        let group = self.m_group.to_string();
        let tile = if self.rows == 256 { "_256" } else { "" };
        let mut jobs = Vec::new();
        let mut add = |source: String, stem: &str, config: &[(&str, &str)]| {
            jobs.push(Job {
                source,
                stem: stem.to_string(),
                config: config
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                    .collect(),
            });
        };
        add(
            format!("prepare_norm_{width}"),
            &format!("prepare_norm_{width}"),
            &[("width", "6144"), ("out_stride", &pitch_hidden), ("eps", "1e-5")],
        );
        add(
            format!("prepare_gated_{width}"),
            &format!("prepare_gated_{width}"),
            &[("width", "6144"), ("out_stride", &pitch_hidden), ("gate_stride", "15360")],
        );
        add(
            format!("prepare_plain_{width}"),
            &format!("prepare_plain_{width}"),
            &[("width", "16384"), ("out_stride", &pitch_inter)],
        );
        add(
            format!("gemm_{width}{tile}"),
            "gemm_qkvg",
            &[
                ("k_size", "6144"),
                ("k_stride", &pitch_hidden),
                ("n_size", "15360"),
                ("m_group", &group),
            ],
        );
        add(
            format!("gemm_{width}_swiglu{tile}"),
            "gemm_gu",
            &[
                ("k_size", "6144"),
                ("k_stride", &pitch_hidden),
                ("n_size", "32768"),
                ("m_group", &group),
            ],
        );
        add(
            format!("gemm_{width}_resid{tile}"),
            "gemm_wo",
            &[
                ("k_size", "6144"),
                ("k_stride", &pitch_hidden),
                ("n_size", "6144"),
                ("m_group", &group),
            ],
        );
        add(
            format!("gemm_{width}_resid{tile}"),
            "gemm_down",
            &[
                ("k_size", "16384"),
                ("k_stride", &pitch_inter),
                ("n_size", "6144"),
                ("m_group", &group),
            ],
        );
        add(
            "rope_qknorm_f16".to_string(),
            "rope_qknorm",
            &[
                ("row_stride", "15360"),
                ("q_heads", "48"),
                ("kv_heads", "12"),
                ("k_offset", "6144"),
                ("eps", "1e-5"),
            ],
        );
        add(
            self.attention_source(),
            "attention",
            &[
                ("q_stride", "6144"),
                ("kv_stride", "1536"),
                ("tokens", &self.tokens.to_string()),
                ("token_capacity", &self.capacity.to_string()),
                ("scale", "0.08838834764831845"),
                ("out_stride", "6144"),
            ],
        );
        if self.query_tiles == 2 {
            add(
                "sage_transpose".to_string(),
                "attention_transpose",
                &[("width", "1536"), ("row_capacity", &self.capacity.to_string())],
            );
        }
        jobs
    }

    fn attention_source(&self) -> String {
        match self.attention_bits {
            16 if self.query_tiles == 2 => "attention_query32".to_string(),
            16 => "attention_gqa_lds_f16_wmma".to_string(),
            bits => {
                let name =
                    if bits == 4 { "attention_sage_i4_fast" } else { "attention_sage_i8_fast" };
                // Eight waves have the registers to hold the next tile; four
                // do not, and prefetch instead.
                match self.attention_waves {
                    8 => name.to_string(),
                    _ => format!("{name}_prefetch"),
                }
            }
        }
    }
}

/// The bundle for `shape`, compiled into `parent` if it is not already there.
///
/// Several processes may call this at once: the winner publishes by renaming a
/// staging directory into place, and the losers find it and verify it.
pub fn prepare(parent: &Path, compiler_path: Option<&str>, shape: &Shape) -> Result<PathBuf> {
    let jobs = shape.jobs();
    let launch = shape.launch_text();
    let mut signature = format!("{SIGNATURE_PREFIX}{launch}");
    for job in &jobs {
        let source = sources::block(&job.source)
            .ok_or_else(|| Error(format!("no embedded kernel source: {}", job.source)))?;
        signature.push_str(source);
        signature.push_str(&job.source);
        signature.push_str(&format!("krea2_{}", job.source));
        signature.push_str(&job.stem);
        for (key, value) in &job.config {
            signature.push_str(&format!("{key}={value}\n"));
        }
    }
    let out = parent.join(format!("T{}-{}", shape.tokens, digest(signature.as_bytes())));
    if out.exists() {
        verify(&out, &jobs, &launch)?;
        return Ok(out);
    }
    std::fs::create_dir_all(parent)
        .map_err(|e| Error(format!("cannot create {}: {e}", parent.display())))?;
    let _lock = Lock::acquire(parent, "kernel cache")?;
    if out.exists() {
        verify(&out, &jobs, &launch)?;
        return Ok(out);
    }

    let staging = parent.join(format!(".prepare-{}", std::process::id()));
    // A dead process may have left a staging directory with this recycled id.
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir(&staging)
        .map_err(|e| Error(format!("cannot create {}: {e}", staging.display())))?;
    let _cleanup = Staging(staging.clone());
    let compiler = compiler(compiler_path);
    let mut hashes = BTreeMap::new();
    for job in &jobs {
        let source = sources::block(&job.source).expect("checked above");
        let artifact = staging.join(format!("{}.hsaco", job.stem));
        Compilation { compiler: &compiler, name: &job.source, source, config: &job.config }
            .run(&staging, &artifact)?;
        let _ = std::fs::remove_file(staging.join(format!("{}.log", job.source)));
        let _ = std::fs::remove_file(staging.join(format!("{}.loom", job.source)));
        hashes.insert(format!("{}.hsaco", job.stem), digest(&read(&artifact)?));
    }
    write(&staging.join("launch.txt"), launch.as_bytes())?;
    write(&staging.join("signature"), signature.as_bytes())?;
    write(&staging.join("compiler.txt"), format!("{compiler}\n").as_bytes())?;
    write(&staging.join("manifest.json"), manifest(&hashes).as_bytes())?;
    std::fs::rename(&staging, &out)
        .map_err(|e| Error(format!("cannot publish {}: {e}", out.display())))?;
    Ok(out)
}

/// The launch metadata and every artifact's hash, before anything is loaded.
fn verify(out: &Path, jobs: &[Job], launch: &str) -> Result<()> {
    let recorded = read(&out.join("manifest.json"))?;
    let recorded: BTreeMap<String, String> = serde_json::from_slice(&recorded)
        .map_err(|e| Error(format!("corrupt kernel manifest in {}: {e}", out.display())))?;
    if read(&out.join("launch.txt"))? != launch.as_bytes() {
        return Err(Error("invalid native launch metadata".into()));
    }
    for job in jobs {
        let name = format!("{}.hsaco", job.stem);
        let want = recorded
            .get(&name)
            .ok_or_else(|| Error(format!("corrupt native kernel: {name}")))?;
        if &digest(&read(&out.join(&name))?) != want {
            return Err(Error(format!("corrupt native kernel: {name}")));
        }
    }
    Ok(())
}

/// The manifest as `nlohmann::json::dump` writes it, so a bundle prepared by
/// either implementation reads the same.
fn manifest(hashes: &BTreeMap<String, String>) -> String {
    serde_json::to_string(hashes).expect("a map of strings is always valid JSON")
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error(format!("cannot read {}: {e}", path.display())))
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes)
        .map_err(|e| Error(format!("cannot write {}: {e}", path.display())))
}

/// The staging directory, removed however the preparation ends. A successful
/// rename leaves nothing to remove.
struct Staging(PathBuf);

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_launch_text_is_the_line_the_session_pins() {
        // tests/test_runtime.py's fixture: 4115 tokens of int8 weights.
        let shape = Shape::new(4115, 8, 16).expect("a shape");
        assert_eq!(shape.launch_text(), "5 4115 256 4 4160 8 6144 16448 16 8 1\n");
    }

    #[test]
    fn the_smoothed_kernels_take_the_prefetch_variant_only_past_eight_thousand() {
        let short = Shape::new(4115, 8, 4).expect("a shape");
        assert_eq!(short.attention_source(), "attention_sage_i4_fast");
        assert_eq!(short.query_tiles, 1, "the smoothed kernels are one query tile");
        let long = Shape::new(9000, 8, 8).expect("a shape");
        assert_eq!(long.attention_source(), "attention_sage_i8_fast_prefetch");
        assert_eq!(
            Shape::new(4115, 8, 16).expect("a shape").attention_source(),
            "attention_gqa_lds_f16_wmma"
        );
    }

    #[test]
    fn every_kernel_a_bundle_names_has_an_embedded_source() {
        for tokens in [16, 4115, 9000, 16896] {
            for gemm_bits in [4, 8] {
                for attention_bits in [4, 8, 16] {
                    let shape = Shape::new(tokens, gemm_bits, attention_bits).expect("a shape");
                    for job in shape.jobs() {
                        assert!(
                            sources::block(&job.source).is_some(),
                            "no source for {} ({tokens}, i{gemm_bits}, qk{attention_bits})",
                            job.source
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_shapes_that_cannot_be_built_are_named() {
        assert!(Shape::new(15, 8, 16).unwrap_err().0.contains("tokens must be"));
        assert!(Shape::new(4115, 6, 16).unwrap_err().0.contains("GEMM operand width"));
        assert!(Shape::new(4115, 8, 12).unwrap_err().0.contains("KREA2_ATTN_QK"));
    }
}
