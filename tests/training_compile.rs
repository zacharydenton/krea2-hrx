//! Offline compiler checks: no GPU device or stream is opened.
use krea2::kernels::{compiler, sources};

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn training_kernels_compile_without_a_device() {
    let compiler = compiler(None).unwrap();
    for &(name, source) in sources::AUXILIARY
        .iter()
        .filter(|(name, _)| name.starts_with("train_") || *name == "lora_transport")
    {
        let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
        for (key, value) in [
            ("grid_x", 5),
            ("grid_y", 1),
            ("rows", 7),
            ("cols", 128),
            ("size", 896),
            ("tokens", 7),
            ("heads", 8),
            ("kv", 2),
            ("qsize", 7168),
            ("ksize", 1792),
            ("stats", 56),
            ("tables", 896),
            ("width", 8),
            ("parts", 1),
            ("reverse", 1),
        ] {
            request.set_config(format!("krea2.{name}.{key}"), value.to_string());
        }
        compiler.module(source).compile(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
        if name == "lora_transport" {
            request.set_config("krea2.lora_transport.reverse", "2");
            compiler.module(source).compile(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn real_training_attention_shapes_compile_without_spills() {
    let compiler = compiler(None).unwrap();
    for tokens in [1024usize, 1043, 4096, 4115] {
        for name in [
            "train_attention",
            "train_attention_delta",
            "train_attention_dq",
            "train_attention_dkv",
        ] {
            let source = sources::auxiliary(name).unwrap();
            let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
            for (key, value) in [
                ("grid_x", tokens * if name == "train_attention_dkv" { 12 } else { 48 }),
                ("grid_y", 1),
                ("tokens", tokens),
                ("heads", 48),
                ("kv", 12),
                ("qsize", tokens * 6144),
                ("ksize", tokens * 1536),
                ("stats", tokens * 48),
            ] {
                request.set_config(format!("krea2.{name}.{key}"), value.to_string());
            }
            let artifact = compiler
                .module(source)
                .compile(&request)
                .unwrap_or_else(|e| panic!("{name}/{tokens}: {e}"));
            let spills: Vec<_> =
                artifact.diagnostics().iter().filter(|d| d.code == "BACKEND/009").collect();
            assert!(spills.is_empty(), "{name}/{tokens}: {spills:?}");
        }
    }
}

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn adapter_gradient_gemms_compile_for_ragged_and_real_shapes() {
    let compiler = compiler(None).unwrap();
    let name = "gemm_bf16_f32_nt";
    let source = sources::auxiliary(name).unwrap();
    for (m, n, k) in
        [(3usize, 19usize, 7usize), (11, 3, 7), (32, 6144, 1043), (16384, 32, 4115)]
    {
        let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
        for (key, value) in [
            ("m", m),
            ("n", n),
            ("k", k),
            ("asize", m * k),
            ("bsize", n * k),
            ("csize", m * n),
            ("astride", m * k),
            ("bstride", n * k),
            ("grid_x", n.div_ceil(64)),
            ("grid_y", m.div_ceil(64)),
        ] {
            request.set_config(format!("krea2.{name}.{key}"), value.to_string());
        }
        compiler
            .module(source)
            .compile(&request)
            .unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
    }
}
