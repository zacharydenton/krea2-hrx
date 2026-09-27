//! Typed transformer artifacts. HRX owns compilation, caching and integrity.
use std::collections::BTreeMap;

use super::{shape, sources, Error, Result, Settings};

/// One kernel to compile: a source, and the name it takes in the bundle.
struct Job {
    source: &'static str,
    stem: &'static str,
    config: Settings,
}

/// Transformer dimensions and launch rules for one sequence length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub tokens: usize,
    /// 16 for fp16 QK and PV, or 4 / 8 for the smoothed Sage kernels.
    pub attention_bits: u32,
    /// Waves per attention workgroup.
    pub attention_waves: u32,
    /// Rows of every sequence-sized buffer, from [`shape::capacity`].
    pub capacity: usize,
}

impl Shape {
    /// `attention_bits` is 16 (fp16 QK and PV, ComfyUI's SDPA class) or 4 / 8
    /// (the smoothed SageAttention-style kernels).
    pub fn new(tokens: usize, attention_bits: u32) -> Result<Shape> {
        if !shape::TOKENS.contains(&tokens) {
            return Err(Error(format!(
                "tokens must be {}..{}",
                shape::TOKENS.start(),
                shape::TOKENS.end()
            )));
        }
        if ![4, 8, 16].contains(&attention_bits) {
            return Err(Error("attention QK width must be 4, 8 or 16".into()));
        }
        Ok(Shape {
            tokens,
            attention_bits,
            // Long sequences leave less room per wave for the score tile.
            attention_waves: if tokens < 8192 { 8 } else { 4 },
            capacity: shape::capacity(tokens),
        })
    }

    /// `KREA2_ATTN_QK` from the environment, 16 when it is unset or empty.
    pub fn from_environment(tokens: usize) -> Result<Shape> {
        let bits = match std::env::var("KREA2_ATTN_QK") {
            Ok(value) if !value.is_empty() => {
                value.parse().map_err(|_| Error("KREA2_ATTN_QK must be 4, 8 or 16".into()))?
            }
            _ => 16,
        };
        Shape::new(tokens, bits)
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
        let gemm = |source, stem, k: &str, stride: &str, n: &str| {
            job(
                source,
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
                self.attention_source(),
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

    /// The attention kernel for this width. Eight waves have the registers to
    /// hold the next key tile; four do not, and prefetch instead.
    pub fn attention_source(&self) -> &'static str {
        match (self.attention_bits, self.attention_waves) {
            (16, _) => "attention_gqa_lds_f16_wmma",
            (4, 8) => "attention_sage_i4_fast",
            (4, _) => "attention_sage_i4_fast_prefetch",
            (_, 8) => "attention_sage_i8_fast",
            (_, _) => "attention_sage_i8_fast_prefetch",
        }
    }
}

/// Verified compiler artifacts and the shape they were specialized for.
#[derive(Debug)]
pub struct PreparedBundle {
    shape: Shape,
    artifacts: BTreeMap<&'static str, hrx::loom::Artifact>,
}

impl PreparedBundle {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
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
        .ok_or_else(|| Error(format!("no embedded kernel source: {}", job.source)))?;
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
    fn the_smoothed_kernels_take_the_prefetch_variant_only_past_eight_thousand() {
        let short = Shape::new(4115, 4).expect("a shape");
        assert_eq!(short.attention_source(), "attention_sage_i4_fast");
        let long = Shape::new(9000, 8).expect("a shape");
        assert_eq!(long.attention_source(), "attention_sage_i8_fast_prefetch");
        assert_eq!(
            Shape::new(4115, 16).expect("a shape").attention_source(),
            "attention_gqa_lds_f16_wmma"
        );
    }

    #[test]
    fn every_kernel_a_bundle_names_has_an_embedded_source() {
        for tokens in [16, 4115, 9000, 16896] {
            for attention_bits in [4, 8, 16] {
                let shape = Shape::new(tokens, attention_bits).expect("a shape");
                for job in shape.jobs() {
                    assert!(
                        sources::block(job.source).is_some(),
                        "no source for {} ({tokens}, qk{attention_bits})",
                        job.source
                    );
                }
            }
        }
    }

    #[test]
    fn the_shapes_that_cannot_be_built_are_named() {
        assert!(Shape::new(15, 16).unwrap_err().0.contains("tokens must be 16..16896"));
        assert!(Shape::new(4115, 12).unwrap_err().0.contains("QK width"));
    }
}
