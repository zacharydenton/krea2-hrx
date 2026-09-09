//! GPU dispatch, allocation-span and runtime-dependency checks.
//! Requires HRX, a gfx1151 GPU and a compiler; run with --ignored.
use hrx::Stream;
use kernels::{auxiliary_kernel, config, Scalars};

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn prepared_operations_reuse_both_shapes_and_remain_ordered() {
    let mut stream = Stream::open().unwrap();
    let prepared = kernels::cache::PreparedKernels::default();
    for count in [257usize, 1009, 257, 1009] {
        let grid = (count.div_ceil(256) as u32, 1);
        let kernel = prepared.get(&stream, "unary_one", kernels::Config::new(), grid).unwrap();
        let buffer = stream.allocate(count * 2).unwrap();
        stream.fill(buffer.binding(), 0).unwrap();
        let constants = Scalars::new().index(count).pack("unary_one", &kernel).unwrap();
        // Safety: unary_one operates independently on count bf16 elements.
        // Input/output aliasing is intentional; both launches use this stream.
        let bindings = [buffer.binding(), buffer.binding()];
        unsafe {
            stream
                .dispatch(&kernel, [grid.0, grid.1, 1], [256, 1, 1], &constants, &bindings)
                .unwrap();
            stream
                .dispatch(&kernel, [grid.0, grid.1, 1], [256, 1, 1], &constants, &bindings)
                .unwrap();
        }
        let mut output = vec![0u16; count];
        stream.read_blocking(buffer.binding(), bytemuck::cast_slice_mut(&mut output)).unwrap();
        assert!(output.iter().all(|value| *value == 0x4000)); // bf16 2.0
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn unary_one_writes_one_into_every_element() {
    const COUNT: usize = 1009; // not a multiple of the workgroup size
    let mut stream = Stream::open().expect("a stream");
    let x = stream.allocate(COUNT * 2).expect("input allocation");
    let y = stream.allocate(COUNT * 2).expect("output allocation");
    stream.fill(x.binding(), 0).expect("zeroing the input");
    stream.fill(y.binding(), 0).expect("zeroing the output");

    let grid = (COUNT.div_ceil(256) as u32, 1);
    let kernel = auxiliary_kernel(
        &stream,
        "unary_one",
        &config([("count_b", COUNT as u64)]),
        grid,
        None,
    )
    .expect("compiling unary_one");
    let constants = Scalars::new().index(COUNT).pack("unary_one", &kernel).unwrap();
    let bindings = [x.binding(), y.binding()];
    // Safety: the kernel writes COUNT independent bf16 elements of `y`.
    unsafe {
        stream.dispatch(&kernel, [grid.0, grid.1, 1], [256, 1, 1], &constants, &bindings)
    }
    .expect("dispatching unary_one");

    let mut bytes = vec![0u8; COUNT * 2];
    stream.read_blocking(y.binding(), &mut bytes).expect("reading the output back");
    for (index, half) in bytes.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([half[0], half[1]]);
        assert_eq!(bits, 0x3f80, "element {index} is {bits:#06x}, not bf16 1.0");
    }
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn a_span_past_the_end_of_an_allocation_is_rejected() {
    let mut stream = Stream::open().expect("a stream");
    let buffer = stream.allocate(1024).expect("allocation");
    let mut host = vec![0u8; 512];
    // Reading 512 bytes starting 768 bytes in runs 256 bytes past the end.
    let error = buffer.try_slice(768, 512).expect_err("an over-long span must be rejected");
    assert_eq!(error.to_string(), "span 768+512 exceeds 1024 bytes");
    // The same span inside the allocation is fine.
    let inside = buffer.try_slice(512, 512).expect("an in-range span");
    stream.read_blocking(inside, &mut host).expect("an in-range read");
}

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_process_maps_no_hip_torch_or_system_crypto() {
    // Touch the device so the provider is loaded before the maps are read.
    let mut stream = Stream::open().expect("a stream");
    stream.synchronize().expect("draining the stream");
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
    let stream = Stream::open().expect("a stream");
    let source = kernels::sources::auxiliary("euler").unwrap();
    let compiler = kernels::compiler(None).unwrap();
    let mut request = hrx::loom::Specialization::new("krea2_euler");
    request.config.insert("krea2.euler.grid_x".into(), "4".into());
    request.config.insert("krea2.euler.grid_y".into(), "1".into());
    auxiliary_kernel(&stream, "euler", &kernels::Config::new(), (4, 1), None).unwrap();
    let directory =
        kernels::cache_root().unwrap().join(compiler.module(source).key(&request).unwrap());
    let artifact = directory.join("kernel.hsaco");
    assert!(artifact.is_file());
    assert_eq!(compiler.module(source).compile(&request).unwrap().path(), artifact);
}

/// A `Kernel` is a value now, not an `Arc`, so nothing outside HRX can observe
/// how many clones name one executable. What is observable is the consequence:
/// a clone must keep the executable alive after the cache that made it is gone.
/// Dispatching through one proves the share is real rather than a borrow.
#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn a_cached_kernel_outlives_the_cache_that_made_it() {
    const COUNT: usize = 256;
    let mut stream = Stream::open().unwrap();
    let kernel = {
        let prepared = kernels::cache::PreparedKernels::default();
        let config = kernels::config([("count_b", COUNT as u64)]);
        let first = prepared.get(&stream, "unary_one", config.clone(), (1, 1)).unwrap();
        let second = prepared.get(&stream, "unary_one", config, (1, 1)).unwrap();
        assert_eq!(first.symbol(), second.symbol());
        assert_eq!(first.info().workgroup_size, second.info().workgroup_size);
        first // `prepared` drops here, with its own clone of the executable
    };

    let x = stream.allocate(COUNT * 2).unwrap();
    let y = stream.allocate(COUNT * 2).unwrap();
    stream.fill(x.binding(), 0).unwrap();
    stream.fill(y.binding(), 0).unwrap();
    let constants = Scalars::new().index(COUNT).pack("unary_one", &kernel).unwrap();
    let bindings = [x.binding(), y.binding()];
    // Safety: the kernel writes COUNT independent bf16 elements of `y`.
    unsafe { stream.dispatch(&kernel, [1, 1, 1], [256, 1, 1], &constants, &bindings) }.unwrap();
    let mut output = vec![0u16; COUNT];
    stream.read_blocking(y.binding(), bytemuck::cast_slice_mut(&mut output)).unwrap();
    assert!(output.iter().all(|bits| *bits == 0x3f80), "bf16 1.0 in every element");
}

#[test]
#[ignore = "requires the provisioned Loom compiler"]
fn compiler_cache_separates_device_targets() {
    let first = hrx::Target::new("gfx1151").unwrap();
    let second = hrx::Target::new("gfx1100").unwrap();
    let a = kernels::cache::compiler_for_target(None, &first).unwrap();
    let b = kernels::cache::compiler_for_target(None, &second).unwrap();
    assert_eq!(a.target(), &first);
    assert_eq!(b.target(), &second);
    assert_eq!(kernels::cache::compiler_for_target(None, &first).unwrap().target(), &first);
}
