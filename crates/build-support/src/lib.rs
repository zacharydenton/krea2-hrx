//! Where `libhrx.so` lives, decided once for every build script that needs it.
//!
//! Six of them used to assume the crate sat in a checkout and reach for
//! `<repo>/build/runtime`. That is true for a `cargo build` here and false for
//! a `cargo install`, which unpacks the crate under `~/.cargo`, so the binary
//! failed to link. The directory is looked for now rather than assumed.
use std::path::{Path, PathBuf};

/// The environment variable that names the runtime directory outright.
pub const RUNTIME: &str = "KREA2_RUNTIME";

/// The directory holding `libhrx.so`, in the order a build should prefer:
///
/// 1. `KREA2_RUNTIME`, which is the answer for a packager or a developer with
///    their own HRX build;
/// 2. the repository's `build/runtime`, when the crate is being built in one;
/// 3. `$XDG_CACHE_HOME/krea2-loom/runtime`, where an installed build keeps the
///    library it fetched.
///
/// The first that exists wins, and the choice is printed so a build that later
/// fails to link says where it looked. `None` means none of them is there.
pub fn runtime_directory() -> Option<PathBuf> {
    println!("cargo:rerun-if-env-changed={RUNTIME}");
    for candidate in candidates() {
        if candidate.join("libhrx.so").exists() {
            return Some(candidate);
        }
    }
    // Nothing has it. Fall back to the repository's directory so an in-tree
    // build that has not run scripts/runtime.sh yet gets the familiar path in
    // its linker error rather than a bare "cannot find -lhrx".
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
/// A crate unpacked by `cargo install` has no such ancestor.
fn in_repository() -> Option<PathBuf> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")?;
    let root = Path::new(&manifest).ancestors().find(|path| path.join("kernels").is_dir())?;
    Some(root.join("build/runtime"))
}

/// Where an installed build keeps the runtime it fetched.
pub fn cache_directory() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("krea2-loom/runtime"))
}

/// The rpath entries a binary or cdylib needs to find `libhrx.so` at run time:
/// wherever it was linked from, the cache an installed build fetches into, and
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
