// Links libhrx from wherever it is: KREA2_RUNTIME, the repository's
// build/runtime, or the cache an installed build fetches into. libhrx dlopens
// the HSA runtime itself and finds the system one, so nothing else is needed
// here.
fn main() {
    let Some(runtime) = krea2_build_support::runtime_directory() else {
        println!(
            "cargo:warning=no libhrx.so found: set {} to the directory holding it",
            krea2_build_support::RUNTIME
        );
        return;
    };
    println!("cargo:rustc-link-search=native={}", runtime.display());
    println!("cargo:rustc-link-lib=dylib=hrx");
    krea2_build_support::emit_rpath();
}
