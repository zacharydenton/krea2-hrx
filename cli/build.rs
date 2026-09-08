// Links against the runtime built by scripts/build_native.sh. KREA2_BUILD names
// the directory holding libkrea2_pipeline.so and libkrea2.so; the default is the
// repository's build/. The rpath is absolute so `cargo run` works from anywhere,
// and scripts/build_cli.sh copies the binary next to the libraries afterwards.
fn main() {
    let build = std::env::var("KREA2_BUILD").unwrap_or_else(|_| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crate sits in the repository")
            .join("build");
        root.to_string_lossy().into_owned()
    });
    println!("cargo:rerun-if-env-changed=KREA2_BUILD");
    println!("cargo:rerun-if-changed=../host/krea2_pipeline.h");
    println!("cargo:rustc-link-search=native={build}");
    println!("cargo:rustc-link-lib=dylib=krea2_pipeline");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{build}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
