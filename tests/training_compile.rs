//! Offline compiler checks: no GPU device or stream is opened.
use krea2::kernels::{compiler, sources};

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn training_kernels_compile_without_a_device() {
    let compiler = compiler(None).unwrap();
    for &(name, source) in sources::AUXILIARY.iter().filter(|(name, _)| {
        (name.starts_with("train_")
            && !name.starts_with("train_gemm")
            && !name.starts_with("train_attention_flash"))
            || *name == "lora_transport"
    }) {
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
    // Include actual 512-area buckets with text tokens, not only round lengths.
    for tokens in [1015usize, 1023, 1024, 1025, 1043, 1067, 1070, 1074, 4096, 4115] {
        for name in [
            "train_attention",
            "train_attention_flash",
            "train_attention_flash_dq",
            "train_attention_flash_dkv",
            "train_attention_flash_delta",
            "train_attention_delta",
            "train_attention_dq",
            "train_attention_dkv",
        ] {
            let source = sources::auxiliary(name).unwrap();
            let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
            for (key, value) in [
                (
                    "grid_x",
                    if name.starts_with("train_attention_flash") {
                        tokens.div_ceil(16)
                    } else {
                        tokens * if name.starts_with("train_attention_dkv") { 12 } else { 48 }
                    },
                ),
                (
                    "grid_y",
                    match name {
                        "train_attention_flash" | "train_attention_flash_dkv" => 12,
                        "train_attention_flash_dq" | "train_attention_flash_delta" => 48,
                        _ => 1,
                    },
                ),
                ("q_stride", 6144),
                ("kv_stride", 1536),
                ("out_stride", 6144),
                (
                    "token_capacity",
                    tokens.div_ceil(16) * 16
                        + if name == "train_attention_flash" { 16 } else { 0 },
                ),
                ("tokens", tokens),
                ("heads", 48),
                ("kv", 12),
                ("qsize", tokens * 6144),
                ("ksize", tokens * 1536),
                ("stats", tokens * 48),
            ] {
                request.set_config(format!("krea2.{name}.{key}"), value.to_string());
            }
            request.set_report(hrx::loom::ReportMode::Details);
            let artifact = compiler
                .module(source)
                .compile(&request)
                .unwrap_or_else(|e| panic!("{name}/{tokens}: {e}"));
            let spills: Vec<_> =
                artifact.diagnostics().iter().filter(|d| d.code == "BACKEND/009").collect();
            assert!(spills.is_empty(), "{name}/{tokens}: {spills:?}");
            if let Some(directory) = std::env::var_os("KREA2_BENCH_REPORT_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    directory.join(format!("{name}-{tokens}-compiler.json")),
                    artifact.report().unwrap().json().to_string(),
                )
                .unwrap();
            }
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

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn training_dense_tiles_compile_without_spills() {
    let compiler = compiler(None).unwrap();
    for name in ["train_gemm", "train_gemm_nn"] {
        for (m, n, k) in [
            (513usize, 64usize, 192usize),
            (513, 192, 64),
            (1070, 64, 6144),
            (1070, 6144, 1536),
            (1043, 6144, 6144),
            (1043, 16384, 6144),
            (1043, 6144, 16384),
            (4115, 16384, 6144),
        ] {
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
                ("grid_x", n / 64),
                ("grid_y", m.div_ceil(128)),
            ] {
                request.set_config(format!("krea2.{name}.{key}"), value.to_string());
            }
            request.set_report(hrx::loom::ReportMode::Details);
            let artifact =
                compiler.module(sources::auxiliary(name).unwrap()).compile(&request).unwrap();
            assert!(
                artifact.diagnostics().iter().all(|d| d.code != "BACKEND/009"),
                "{name}/{m}x{n}x{k}: unexpected spill"
            );
            if let Some(directory) = std::env::var_os("KREA2_BENCH_REPORT_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    directory.join(format!("{name}-{m}x{n}x{k}-compiler.json")),
                    artifact.report().unwrap().json().to_string(),
                )
                .unwrap();
            }
        }
    }
}

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn adapter_direct_gradients_compile_without_spills() {
    let compiler = compiler(None).unwrap();
    let name = "train_gemm_tn_accumulate";
    for (m, n, k) in [
        (3usize, 19usize, 7usize),
        (65, 33, 31),
        (32, 6144, 1070),
        (16384, 32, 4115),
        (64, 16384, 1074),
    ] {
        let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
        for (key, value) in [
            ("m", m),
            ("n", n),
            ("k", k),
            ("grid_x", n.div_ceil(32)),
            ("grid_y", m.div_ceil(32)),
        ] {
            request.set_config(format!("krea2.{name}.{key}"), value.to_string());
        }
        request.set_report(hrx::loom::ReportMode::Details);
        let artifact =
            compiler.module(sources::auxiliary(name).unwrap()).compile(&request).unwrap();
        assert!(
            artifact.diagnostics().iter().all(|d| d.code != "BACKEND/009"),
            "{m}x{n}x{k}: unexpected spill"
        );
        if let Some(directory) = std::env::var_os("KREA2_BENCH_REPORT_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join(format!("{name}-{m}x{n}x{k}-compiler.json")),
                artifact.report().unwrap().json().to_string(),
            )
            .unwrap();
        }
    }
}

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn optimizer_kernels_compile_without_spills() {
    let compiler = compiler(None).unwrap();
    for name in ["train_grad_norm", "train_adamw", "train_adamw_graph"] {
        for count in [57usize, 32 * 6144, 64 * 16384] {
            let parts = count.div_ceil(1024);
            let grid = if name == "train_grad_norm" { parts } else { count.div_ceil(256) };
            let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
            for (key, value) in [("parts", parts), ("grid_x", grid), ("grid_y", 1)] {
                request.set_config(format!("krea2.{name}.{key}"), value.to_string());
            }
            request.set_report(hrx::loom::ReportMode::Details);
            let artifact =
                compiler.module(sources::auxiliary(name).unwrap()).compile(&request).unwrap();
            assert!(
                artifact.diagnostics().iter().all(|d| d.code != "BACKEND/009"),
                "{name}/{count}: unexpected spill"
            );
            if let Some(directory) = std::env::var_os("KREA2_BENCH_REPORT_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    directory.join(format!("{name}-{count}-compiler.json")),
                    artifact.report().unwrap().json().to_string(),
                )
                .unwrap();
            }
        }
    }
}
