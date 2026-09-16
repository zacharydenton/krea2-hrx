//! Warm source-based HRX lookups versus Krea's caller-key index into HRX.
//! cargo run --release --example kernel_lookup
use hrx::loom::{Kernels, Specialization};
use krea2::kernels::{cache::compiler_for_target, config, sources, PreparedKernels};
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stream = hrx::Stream::open()?;
    let source_cache = Kernels::new(compiler_for_target(None, stream.target())?);
    let keyed_cache = PreparedKernels::default();
    for (name, config) in [
        ("unary_one", config([("count_b", 256)])),
        (
            "gemm_bf16_bf16_nt_bias_wide",
            config([
                ("m", 128),
                ("n", 64),
                ("k", 256),
                ("asize", 128 * 256),
                ("bsize", 64 * 256),
                ("csize", 128 * 64),
                ("astride", 128 * 256),
                ("bstride", 64 * 256),
            ]),
        ),
    ] {
        let source = sources::auxiliary(name).unwrap();
        let mut spec = Specialization::new(format!("krea2_{name}"));
        for (key, value) in &config {
            spec.set_config(format!("krea2.{name}.{key}"), value.to_string());
        }
        for axis in ["x", "y"] {
            spec.set_config(format!("krea2.{name}.grid_{axis}"), "1");
        }
        spec.set_report(krea2::kernels::kernel_reports());
        // Both caches are warm before timing. All source is embedded and trusted.
        unsafe { source_cache.get(&stream, source, &spec)? };
        keyed_cache.get(&stream, name, config.clone(), (1, 1))?;
        let mut samples = [Vec::new(), Vec::new()];
        for round in 0..12 {
            for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let began = Instant::now();
                for _ in 0..2048 {
                    let kernel = if arm == 0 {
                        // Conservative baseline: even specialization construction
                        // is outside the timer for the source-based API.
                        unsafe { source_cache.get(&stream, source, &spec)? }
                    } else {
                        keyed_cache.get(&stream, name, config.clone(), (1, 1))?
                    };
                    black_box(kernel);
                }
                if round >= 3 {
                    samples[arm].push(began.elapsed().as_secs_f64() * 1e9 / 2048.0);
                }
            }
        }
        for arm in &mut samples {
            arm.sort_by(f64::total_cmp);
        }
        println!(
            "{name}: source {:.0} ns, keyed {:.0} ns per lookup",
            samples[0][4], samples[1][4]
        );
    }
    Ok(())
}
