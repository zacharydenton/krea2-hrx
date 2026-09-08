//! Runtime search paths and generated C headers for the shared library.
use std::path::{Path, PathBuf};

fn main() {
    krea2_build_support::emit_rpath();
    headers();
}

/// Generates C headers in OUT_DIR and, in a checkout, `build/include`.
fn headers() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    for (source, header, guard) in [
        ("src/lib.rs", "krea2.h", "KREA2_H"),
        ("src/pipeline.rs", "krea2_pipeline.h", "KREA2_PIPELINE_H"),
    ] {
        println!("cargo:rerun-if-changed={source}");
        let generated = cbindgen::Builder::new()
            .with_src(Path::new(env!("CARGO_MANIFEST_DIR")).join(source))
            .with_language(cbindgen::Language::C)
            // Preserve C linkage for C++ callers.
            .with_cpp_compat(true)
            .with_documentation(true)
            .with_include_guard(guard)
            .with_no_includes()
            .with_sys_include("stddef.h")
            .with_sys_include("stdint.h")
            .with_header(BANNER)
            // Keep C handle names distinct from Rust crate names.
            .rename_item("SessionHandle", "krea2_session")
            .rename_item("WeightsHandle", "krea2_weights")
            .rename_item("PipelineHandle", "krea2_pipeline")
            .rename_item("ProgressFn", "krea2_progress")
            .generate();
        match generated {
            // Header generation failures are warnings; the Rust build can continue.
            Err(error) => println!("cargo:warning=cannot generate {header}: {error}"),
            Ok(bindings) => {
                bindings.write_to_file(out.join(header));
                if let Some(include) = repository_include() {
                    let _ = std::fs::create_dir_all(&include);
                    bindings.write_to_file(include.join(header));
                }
            }
        }
    }
}

const BANNER: &str = "// Generated from the Rust sources by crates/abi/build.rs. Do not edit.";

/// `<repo>/build/include`, when this is a build inside the checkout.
fn repository_include() -> Option<PathBuf> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")?;
    let root = Path::new(&manifest).ancestors().find(|path| path.join("kernels").is_dir())?;
    Some(root.join("build/include"))
}
