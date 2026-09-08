// Link libhrx from the configured directory, checkout, or installed cache.
// HRX loads the HSA runtime dynamically.
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
