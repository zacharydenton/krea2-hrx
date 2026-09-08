//! Shared runtime-library discovery and linker search paths for build scripts.
use std::path::{Path, PathBuf};

/// The environment variable that names the runtime directory outright.
pub const RUNTIME: &str = "KREA2_RUNTIME";

/// Environment inputs that invalidate cached linker search paths.
const WATCHED: [&str; 3] = [RUNTIME, "XDG_CACHE_HOME", "HOME"];

/// The directory holding `libhrx.so`, in the order a build should prefer:
///
/// 1. `KREA2_RUNTIME`, which is the answer for a packager or a developer with
///    their own HRX build;
/// 2. the repository's `build/runtime`, when the crate is being built in one;
/// 3. `$XDG_CACHE_HOME/krea2-loom/runtime`, for an optional installed library.
///
/// Selects the first directory containing `libhrx.so`, falling back to the
/// checkout's runtime directory when no library is found.
pub fn runtime_directory() -> Option<PathBuf> {
    for name in WATCHED {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for candidate in candidates() {
        if candidate.join("libhrx.so").exists() {
            return Some(candidate);
        }
    }
    // Preserve the checkout path in linker diagnostics before runtime staging.
    in_repository()
}

fn candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(named) = std::env::var_os(RUNTIME) {
        candidates.push(PathBuf::from(named));
    }
    candidates.extend(in_repository());
    candidates.extend(cache_directory());
    candidates
}

/// `<repo>/build/runtime`, when this crate is being built inside the checkout.
/// Returns None when no ancestor contains the kernel sources.
fn in_repository() -> Option<PathBuf> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")?;
    let root = Path::new(&manifest).ancestors().find(|path| path.join("kernels").is_dir())?;
    Some(root.join("build/runtime"))
}

/// Optional runtime installation in the user's cache.
pub fn cache_directory() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("krea2-loom/runtime"))
}

/// The rpath entries a binary or cdylib needs to find `libhrx.so` at run time:
/// the selected library directory, the user's runtime cache, and
/// a `runtime/` directory beside the artifact for a self-contained deployment.
pub fn emit_rpath() {
    let mut emitted = Vec::new();
    for directory in runtime_directory().into_iter().chain(cache_directory()) {
        if !emitted.contains(&directory) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
            emitted.push(directory);
        }
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
