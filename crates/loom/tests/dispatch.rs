//! GPU dispatch, allocation-span and runtime-dependency checks.
//! Requires HRX, a gfx1151 GPU and a compiler; run with --ignored.
use hrx::{device, Args};
use loom::{auxiliary_kernel, config};

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn prepared_operations_reuse_both_shapes_and_remain_ordered() {
    let stream = hrx::Device::open().unwrap();
    let _scope = stream.enter();
    let prepared = loom::cache::PreparedKernels::default();
    for count in [257usize, 1009, 257, 1009] {
        let grid = (count.div_ceil(256) as u32, 1);
        let kernel = prepared.get("unary_one", loom::Config::new(), grid).unwrap();
        let buffer = stream.allocate(count * 2).unwrap();
        stream.zero(buffer.ptr(), buffer.len()).unwrap();
        let mut args = Args::new();
        args.i32(count as i32).ptr(buffer.ptr()).ptr(buffer.ptr());
        // Safety: unary_one operates independently on count bf16 elements.
        // Input/output aliasing is intentional; both launches use this stream.
        unsafe {
            kernel.launch_2d(grid.0, grid.1, 256, &args).unwrap();
            kernel.launch_2d(grid.0, grid.1, 256, &args).unwrap();
        }
        let mut output = vec![0u16; count];
        stream.read(&mut output, buffer.ptr()).unwrap();
        assert!(output.iter().all(|value| *value == 0x4000)); // bf16 2.0
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn unary_one_writes_one_into_every_element() {
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
        None,
    )
    .expect("compiling unary_one");
    let mut args = Args::new();
    args.i32(COUNT as i32).ptr(x.ptr()).ptr(y.ptr());
    unsafe { kernel.launch_2d(COUNT.div_ceil(256) as u32, 1, 256, &args) }
        .expect("dispatching unary_one");
    device.synchronize().expect("draining the stream");

    let mut bytes = vec![0u8; COUNT * 2];
    device.copy_to_host(&mut bytes, y.ptr()).expect("reading the output back");
    for (index, half) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([half[0], half[1]]);
        assert_eq!(bits, 0x3f80, "element {index} is {bits:#06x}, not bf16 1.0");
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn a_span_past_the_end_of_an_allocation_is_rejected() {
    let device = device();
    let buffer = device.allocate(1024).expect("allocation");
    let mut host = vec![0u8; 512];
    // Reading 512 bytes starting 768 bytes in runs 256 bytes past the end.
    let error = device
        .copy_to_host(&mut host, buffer.ptr().offset(768))
        .expect_err("an over-long span must be rejected");
    assert_eq!(error.to_string(), "span 768+512 exceeds 1024 bytes");
    // The same span inside the allocation is fine.
    device.copy_to_host(&mut host, buffer.ptr().offset(512)).expect("an in-range span");
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_process_maps_no_hip_torch_or_system_crypto() {
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

/// The model must populate the shared cache, without the former C++ cache layer.
#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn auxiliary_compilation_uses_the_shared_hrx_artifact() {
    let source = loom::sources::auxiliary("euler").unwrap();
    let compiler = loom::compiler(None).unwrap();
    let mut request = hrx::loom::Request::new(source, "krea2_euler");
    request.config.insert("krea2.euler.grid_x".into(), "4".into());
    request.config.insert("krea2.euler.grid_y".into(), "1".into());
    auxiliary_kernel("euler", &loom::Config::new(), (4, 1), None).unwrap();
    let directory = loom::cache_root().unwrap().join(compiler.key(&request).unwrap());
    let artifact = directory.join("kernel.hsaco");
    assert!(artifact.is_file());
    assert_eq!(compiler.compile(&request, &loom::cache_root().unwrap()).unwrap(), artifact);
}
