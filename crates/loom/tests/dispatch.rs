//! The gate for the Rust device layer: compile an embedded kernel with
//! `loom-compile`, dispatch it on the gfx1151, and read the result back.
//!
//! This is what `tests/test_hrx_runtime.cpp` proves for the C++ host, kept in
//! the same shape: `unary_one` writes `1 + x` for every element, so a zeroed
//! buffer must come back as bf16 1.0 (`0x3f80`) in every lane, and a span that
//! runs past its allocation must be rejected rather than truncated.
//!
//! Requires a GPU and `loom-compile` on PATH (or `LOOM_COMPILE`); it is skipped
//! when neither is present, so `cargo test` still works on a build machine.
use hrx::{device, Args};
use loom::{auxiliary_kernel, config};

fn usable() -> bool {
    let compiler = loom::compiler(None);
    let found = std::path::Path::new(&compiler).exists()
        || std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path).any(|entry| entry.join(&compiler).exists())
        });
    if !found {
        eprintln!("skipping: no loom-compile ({compiler})");
        return false;
    }
    if let Err(error) = hrx::try_device() {
        eprintln!("skipping: no GPU ({error})");
        return false;
    }
    true
}

#[test]
fn unary_one_writes_one_into_every_element() {
    if !usable() {
        return;
    }
    const COUNT: usize = 1009; // not a multiple of the workgroup size
    let device = device();
    let x = device.allocate(COUNT * 2).expect("input allocation");
    let y = device.allocate(COUNT * 2).expect("output allocation");
    device.zero(x.ptr(), COUNT * 2).expect("zeroing the input");
    device.zero(y.ptr(), COUNT * 2).expect("zeroing the output");

    let kernel = auxiliary_kernel(
        "unary_one",
        &config([("count_b", COUNT as u64)]),
        (COUNT.div_ceil(256) as u32, 1),
    )
    .expect("compiling unary_one");
    let mut args = Args::new();
    args.i32(COUNT as i32).ptr(x.ptr()).ptr(y.ptr());
    kernel.launch_2d(COUNT.div_ceil(256) as u32, 1, 256, &args).expect("dispatching unary_one");
    device.synchronize().expect("draining the stream");

    let mut bytes = vec![0u8; COUNT * 2];
    device.copy_to_host(&mut bytes, y.ptr()).expect("reading the output back");
    for (index, half) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([half[0], half[1]]);
        assert_eq!(bits, 0x3f80, "element {index} is {bits:#06x}, not bf16 1.0");
    }
}

#[test]
fn a_span_past_the_end_of_an_allocation_is_rejected() {
    if !usable() {
        return;
    }
    let device = device();
    let buffer = device.allocate(1024).expect("allocation");
    let mut host = vec![0u8; 512];
    // Reading 512 bytes starting 768 bytes in runs 256 bytes past the end.
    let error = device
        .copy_to_host(&mut host, buffer.ptr().offset(768))
        .expect_err("an over-long span must be rejected");
    assert_eq!(error.0, "GPU buffer span");
    // The same span inside the allocation is fine.
    device.copy_to_host(&mut host, buffer.ptr().offset(512)).expect("an in-range span");
}

#[test]
fn the_process_maps_no_hip_torch_or_system_crypto() {
    if !usable() {
        return;
    }
    // Touch the device so the provider is loaded before the maps are read.
    device().synchronize().expect("draining the stream");
    let maps = std::fs::read_to_string("/proc/self/maps").expect("reading /proc/self/maps");
    for required in ["libhrx.so", "libhsa-runtime64"] {
        assert!(maps.contains(required), "{required} is not mapped");
    }
    for forbidden in [
        "libamdhip64",
        "libhipblas",
        "librocblas",
        "libtorch",
        "libpython",
        "libpcre2",
        "libicu",
        "libcrypto",
    ] {
        assert!(!maps.contains(forbidden), "{forbidden} must not be mapped");
    }
}

/// The Rust cache must key exactly as the C++ host did, or a port silently
/// recompiles everything and, worse, could diverge on what it loads.
///
/// The C++ `Ops::euler_step` launched `euler` with no configuration of its own
/// and a 4x1 grid at 1000 elements, which made the signature `"euler\ngrid_x=4\n
/// grid_y=1\n"`. Compiling it here and finding the artifact under the key that
/// signature derives checks the whole chain -- signature, digest, cache path --
/// without needing an entry some earlier test happened to leave behind.
#[test]
fn the_cache_key_matches_the_cpp_host() {
    if !usable() {
        return;
    }
    // The digest is over source plus signature, and is pure: check it before
    // anything touches the disk.
    let source = loom::sources::auxiliary("euler").expect("the euler kernel");
    let signature = "euler\ngrid_x=4\ngrid_y=1\n";
    let key = loom::compile::digest(format!("{source}{signature}").as_bytes());

    // Compiling through the normal path must land on exactly that key.
    auxiliary_kernel("euler", &loom::Config::new(), (4, 1)).expect("compiling euler");
    let root = loom::cache_root().expect("a cache directory");
    let path = root.join(format!("{key}.hsaco"));
    assert!(
        path.exists(),
        "{} is missing: auxiliary_kernel compiled euler for a 4x1 grid, so the C++ \
         host's key must name the artifact it produced",
        path.display()
    );
    assert!(root.join(format!("{key}.sha256")).exists(), "the hash beside it");
}
