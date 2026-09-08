//! Two things the cdylib needs that cargo will not do on its own: the runtime
//! rpath, and the C headers for the ABI it exports.
use std::path::{Path, PathBuf};

fn main() {
    rpath();
    headers();
}

/// The cdylib links libhrx, so it needs the same runtime rpath the hrx crate
/// sets for itself; cargo does not propagate a dependency's link arguments.
fn rpath() {
    let runtime = std::env::var("KREA2_RUNTIME")
        .unwrap_or_else(|_| root().join("build/runtime").to_string_lossy().into_owned());
    println!("cargo:rerun-if-env-changed=KREA2_RUNTIME");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{runtime}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");
}

/// `build/include/krea2.h` and `krea2_pipeline.h`, from the sources that
/// export them, so the declarations cannot drift from the definitions. They
/// are build artifacts for third parties: nothing in this repository compiles
/// C, and no header is committed.
fn headers() {
    let include = root().join("build/include");
    for (source, header, name) in [
        ("src/lib.rs", "krea2.h", "KREA2_H"),
        ("src/pipeline.rs", "krea2_pipeline.h", "KREA2_PIPELINE_H"),
    ] {
        println!("cargo:rerun-if-changed={source}");
        let generated = cbindgen::Builder::new()
            .with_src(Path::new(env!("CARGO_MANIFEST_DIR")).join(source))
            .with_language(cbindgen::Language::C)
            // extern "C" guards, so a C++ translation unit including this
            // header links against the library rather than mangled names.
            .with_cpp_compat(true)
            .with_documentation(true)
            .with_include_guard(name)
            .with_no_includes()
            .with_sys_include("stddef.h")
            .with_sys_include("stdint.h")
            .with_header(BANNER)
            // The names C has always seen. The Rust types cannot take them:
            // `krea2_session` would shadow the crate it wraps.
            .rename_item("SessionHandle", "krea2_session")
            .rename_item("WeightsHandle", "krea2_weights")
            .rename_item("PipelineHandle", "krea2_pipeline")
            .rename_item("ProgressFn", "krea2_progress")
            .generate();
        match generated {
            // A missing header is not worth failing a build over: it is
            // documentation for callers in other languages, not an input.
            Err(error) => println!("cargo:warning=cannot generate {header}: {error}"),
            Ok(bindings) => {
                let _ = std::fs::create_dir_all(&include);
                bindings.write_to_file(include.join(header));
            }
        }
    }
}

const BANNER: &str = "// Generated from the Rust sources by crates/abi/build.rs. Do not edit.";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits in the repository")
        .to_path_buf()
}
