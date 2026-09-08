// Tests in this crate dispatch on the GPU, so they need the runtime rpath the
// hrx crate sets for itself; cargo does not propagate a dependency's link args.
fn main() {
    let runtime = std::env::var("KREA2_RUNTIME").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("the crate sits in the repository")
            .join("build/runtime")
            .to_string_lossy()
            .into_owned()
    });
    println!("cargo:rerun-if-env-changed=KREA2_RUNTIME");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{runtime}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");
}
