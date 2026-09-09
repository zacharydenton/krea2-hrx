//! Typed transformer artifacts. HRX owns compilation, caching and integrity.
use std::collections::BTreeMap;

use crate::{shape, sources, Error, Result, Settings};

/// One kernel to compile: a source, and the file name it takes in the bundle.
struct Job {
    source: String,
    stem: String,
    config: Settings,
}

/// Transformer dimensions and launch rules.
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

/// Verified compiler artifacts and the shape they were specialized for.
#[derive(Debug)]
pub struct PreparedBundle {
    shape: Shape,
    artifacts: BTreeMap<String, hrx::loom::Artifact>,
}

impl PreparedBundle {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn artifact(&self, stem: &str) -> Option<&hrx::loom::Artifact> {
        self.artifacts.get(stem)
    }
}

/// Prepare all block exports, reusing HRX's indexed modules and verified cache.
pub fn prepare(compiler_path: Option<&str>, shape: &Shape) -> Result<PreparedBundle> {
    prepare_for_target(compiler_path, shape, &hrx::Target::default())
}

/// Prepare block exports for the device that will execute them.
pub fn prepare_for_target(
    compiler_path: Option<&str>,
    shape: &Shape,
    target: &hrx::Target,
) -> Result<PreparedBundle> {
    let shared = crate::cache::compiler_for_target(compiler_path, target)?;
    let cache = crate::cache_root()?;
    let jobs = shape.jobs();

    // A cold shape compiles every block kernel, and the compiler is built for
    // concurrency: distinct specializations take distinct cache locks and the
    // per-module index lock is only for the one-time index build. Compiling one
    // at a time left that idle and made a new resolution wait seconds it did
    // not need to. Workers pull from a shared queue so a slow kernel does not
    // strand a thread with an empty share.
    let queue = std::sync::Mutex::new(jobs.into_iter());
    let width = match std::env::var("KREA2_COMPILE_WORKERS").ok().and_then(|v| v.parse().ok()) {
        Some(n) if n > 0 => n,
        _ => std::thread::available_parallelism().map_or(1, |n| n.get()).min(8),
    };
    let compiled = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..width)
            .map(|_| {
                let (queue, shared, cache) = (&queue, &shared, &cache);
                scope.spawn(move || -> Result<Vec<(String, hrx::loom::Artifact)>> {
                    let mut done = Vec::new();
                    loop {
                        let job = {
                            let mut queue =
                                queue.lock().map_err(|_| Error("job queue poisoned".into()))?;
                            match queue.next() {
                                Some(job) => job,
                                None => break,
                            }
                        };
                        done.push((job.stem.clone(), compile_one(shared, cache, &job)?));
                    }
                    Ok(done)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| {
                worker.join().map_err(|_| Error("a compile worker panicked".into()))?
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let artifacts: BTreeMap<String, hrx::loom::Artifact> =
        compiled.into_iter().flatten().collect();
    // Keep HRX's bounded module cache: another guidance shape uses these same sources.
    Ok(PreparedBundle { shape: shape.clone(), artifacts })
}

/// One kernel, with whatever the compiler had to say about it.
fn compile_one(
    shared: &hrx::loom::Compiler,
    cache: &std::path::Path,
    job: &Job,
) -> Result<hrx::loom::Artifact> {
    let source = sources::block(&job.source)
        .ok_or_else(|| Error(format!("no embedded kernel source: {}", job.source)))?;
    let mut request = hrx::loom::Specialization::new(format!("krea2_{}", job.source));
    request.config = job
        .config
        .iter()
        .map(|(k, v)| (format!("krea2.{}.{k}", job.source), v.clone()))
        .collect();
    request.report = crate::kernel_reports();
    let artifact = shared.module(source).compile(&request, cache)?;
    report(&job.stem, &artifact);
    Ok(artifact)
}

/// Loom's own diagnostics and, when asked for, its compilation report. Both
/// were discarded before: only the message of a *failed* compile survived, and
/// a warning on a kernel that still built was never seen at all.
fn report(stem: &str, artifact: &hrx::loom::Artifact) {
    for diagnostic in artifact.diagnostics() {
        eprintln!(
            "krea2 kernel {stem}: {} {} at {}:{}: {}",
            diagnostic.severity,
            diagnostic.code,
            diagnostic.line,
            diagnostic.column,
            diagnostic.message
        );
    }
    if let Some(report) = artifact.report() {
        eprintln!("krea2 kernel {stem} report: {report}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
