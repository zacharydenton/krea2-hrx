// The binary links libhrx through the krea2 crates, so it needs the same
// runtime rpath the hrx crate sets for itself; cargo does not propagate a
// dependency's link arguments. KREA2_RUNTIME names the directory holding
// libhrx.so and the HSA provider; the default is the repository's
// build/runtime, which scripts/runtime.sh fills.
fn main() {
    let runtime = std::env::var("KREA2_RUNTIME").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crate sits in the repository")
            .join("build/runtime")
            .to_string_lossy()
            .into_owned()
    });
    println!("cargo:rerun-if-env-changed=KREA2_RUNTIME");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{runtime}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/runtime");
}
