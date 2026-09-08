// The CLI is a consumer of the runtime's C ABI like any other, so it links
// build/libkrea2_pipeline.so rather than depending on the crate behind it.
// KREA2_BUILD names the directory holding that library; the default is the
// repository's build/, which scripts/build.sh fills before this pass runs. The
// rpath is absolute so `cargo run` works from anywhere.
fn main() {
    let build = std::env::var("KREA2_BUILD").unwrap_or_else(|_| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crate sits in the repository")
            .join("build");
        root.to_string_lossy().into_owned()
    });
    println!("cargo:rerun-if-env-changed=KREA2_BUILD");
    println!("cargo:rerun-if-changed=../crates/abi/src/pipeline.rs");
    println!("cargo:rustc-link-search=native={build}");
    println!("cargo:rustc-link-lib=dylib=krea2_pipeline");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{build}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
