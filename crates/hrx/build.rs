// Links libhrx from the runtime directory the build scripts publish
// (build/runtime), which is also where the HSA provider lives.
fn main() {
    let runtime = std::env::var("KREA2_RUNTIME").unwrap_or_else(|_| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("the crate sits in the repository")
            .join("build/runtime");
        root.to_string_lossy().into_owned()
    });
    println!("cargo:rerun-if-env-changed=KREA2_RUNTIME");
    println!("cargo:rustc-link-search=native={runtime}");
    println!("cargo:rustc-link-lib=dylib=hrx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{runtime}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");
}
