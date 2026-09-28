//! Typed transformer artifacts. HRX owns compilation, caching and integrity.
use std::collections::BTreeMap;

use super::{Error, Result, Settings, shape, sources};

/// One kernel to compile: a source, and the name it takes in the bundle.
struct Job {
    source: &'static str,
    stem: &'static str,
    config: Settings,
}

/// The attention kernel: fp16 QK and PV with an fp32 online softmax.
pub const ATTENTION_SOURCE: &str = "attention_gqa_lds_f16_wmma";

/// Transformer dimensions and launch rules for one sequence length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Text and image tokens together.
    pub tokens: usize,
    /// Rows of every sequence-sized buffer, from [`shape::capacity`].
    pub capacity: usize,
}

impl Shape {
    /// The rules for `tokens`, refusing a length outside [`shape::TOKENS`].
    pub fn new(tokens: usize) -> Result<Shape> {
        if !shape::TOKENS.contains(&tokens) {
            return Err(Error::invalid(format!(
                "tokens must be {}..{}",
                shape::TOKENS.start(),
                shape::TOKENS.end()
            )));
        }
        Ok(Shape { tokens, capacity: shape::capacity(tokens) })
    }

    fn jobs(&self) -> Vec<Job> {
        let pitch_hidden = shape::gemm_pitch(6144).to_string();
        let pitch_inter = shape::gemm_pitch(16384).to_string();
        let group = shape::GEMM_M_GROUP.to_string();
        let tokens = self.tokens.to_string();
        let capacity = self.capacity.to_string();
        let job = |source, stem, config: &[(&str, &str)]| Job {
            source,
            stem,
            config: config.iter().map(|&(key, value)| (key.into(), value.into())).collect(),
        };
        let gemm = |family, stem, k: &str, stride: &str, n: &str| {
            job(
                shape::gemm_source(family, shape::gemm_tile(stem)),
                stem,
                &[("k_size", k), ("k_stride", stride), ("n_size", n), ("m_group", &group)],
            )
        };
        vec![
            job(
                "prepare_norm_i8",
                "prepare_norm_i8",
                &[("width", "6144"), ("out_stride", &pitch_hidden), ("eps", "1e-5")],
            ),
            job(
                "prepare_gated_i8",
                "prepare_gated_i8",
                &[("width", "6144"), ("out_stride", &pitch_hidden), ("gate_stride", "15360")],
            ),
            job(
                "prepare_plain_i8",
                "prepare_plain_i8",
                &[("width", "16384"), ("out_stride", &pitch_inter)],
            ),
            gemm("gemm_i8_256", "gemm_qkvg", "6144", &pitch_hidden, "15360"),
            gemm("gemm_i8_swiglu_256", "gemm_gu", "6144", &pitch_hidden, "32768"),
            gemm("gemm_i8_resid_256", "gemm_wo", "6144", &pitch_hidden, "6144"),
            gemm("gemm_i8_resid_256", "gemm_down", "16384", &pitch_inter, "6144"),
            job(
                "rope_qknorm_f16",
                "rope_qknorm",
                &[
                    ("row_stride", "15360"),
                    ("q_heads", "48"),
                    ("kv_heads", "12"),
                    ("k_offset", "6144"),
                    ("eps", "1e-5"),
                ],
            ),
            job(
                shape::attention_source(self.tokens),
                "attention",
                &[
                    ("q_stride", "6144"),
                    ("kv_stride", "1536"),
                    ("tokens", &tokens),
                    ("token_capacity", &capacity),
                    ("scale", "0.08838834764831845"),
                    ("out_stride", "6144"),
                ],
            ),
        ]
    }
}

/// Verified compiler artifacts and the shape they were specialized for.
#[derive(Debug)]
pub struct PreparedBundle {
    shape: Shape,
    artifacts: BTreeMap<&'static str, hrx::loom::Artifact>,
}

impl PreparedBundle {
    /// The shape every artifact was specialized for.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    /// The artifact for one bundle stem, such as `gemm_qkvg`.
    pub fn artifact(&self, stem: &str) -> Option<&hrx::loom::Artifact> {
        self.artifacts.get(stem)
    }
}

/// Prepare block exports for the device that will execute them.
pub fn prepare_for_target(
    compiler_path: Option<&str>,
    shape: &Shape,
    target: &hrx::Target,
) -> Result<PreparedBundle> {
    let shared = super::cache::compiler_for_target(compiler_path, target)?;
    let jobs = shape.jobs();

    // A cold shape compiles every block kernel, and the compiler is built for
    // concurrency: distinct specializations take distinct cache locks and the
    // per-module index lock is only for the one-time index build. compile_all
    // runs the batch across the compiler's own workspace pool, which is the one
    // place that knows how many workspaces it can afford, and returns results in
    // request order.
    let specialized: Vec<(&'static str, hrx::loom::Specialization)> =
        jobs.iter().map(specialization).collect::<Result<_>>()?;
    let modules: Vec<hrx::loom::Module> =
        specialized.iter().map(|(source, _)| shared.module(source)).collect();
    let requests: Vec<(&hrx::loom::Module, &hrx::loom::Specialization)> =
        modules.iter().zip(specialized.iter().map(|(_, request)| request)).collect();

    // compile_all does not short-circuit, so every kernel is attempted and the
    // first failure in job order is the one reported.
    let mut artifacts = BTreeMap::new();
    for (job, outcome) in jobs.iter().zip(shared.compile_all(&requests)) {
        let artifact = outcome?;
        super::report(job.stem, &artifact);
        artifacts.insert(job.stem, artifact);
    }
    // Keep HRX's bounded module cache: another guidance shape uses these same sources.
    Ok(PreparedBundle { shape: shape.clone(), artifacts })
}

/// One kernel's embedded source and the specialization that selects it.
fn specialization(job: &Job) -> Result<(&'static str, hrx::loom::Specialization)> {
    let source = sources::block(job.source)
        .ok_or_else(|| Error::internal(format!("no embedded kernel source: {}", job.source)))?;
    let mut request = hrx::loom::Specialization::new(format!("krea2_{}", job.source));
    request.replace_config(
        job.config
            .iter()
            .map(|(k, v)| (format!("krea2.{}.{k}", job.source), v.clone()))
            .collect(),
    );
    request.set_report(if super::kernel_reports() {
        hrx::loom::ReportMode::Summary
    } else {
        hrx::loom::ReportMode::None
    });
    Ok((source, request))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_a_bundle_names_has_an_embedded_source() {
        for tokens in [16, 4115, 9000, 16896] {
            let shape = Shape::new(tokens).expect("a shape");
            for job in shape.jobs() {
                assert!(
                    sources::block(job.source).is_some(),
                    "no source for {} ({tokens})",
                    job.source
                );
            }
        }
    }

    #[test]
    fn a_sequence_the_kernels_cannot_serve_is_named() {
        assert!(Shape::new(15).unwrap_err().to_string().contains("tokens must be 16..16896"));
        assert!(Shape::new(16897).is_err());
    }
}
