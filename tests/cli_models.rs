//! Exercise CLI model discovery with isolated HF caches and no network or GPU.
use std::path::Path;
use std::process::Command;

fn cli(home: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_krea2"));
    command
        .current_dir(home)
        .env("HOME", home)
        .env("HF_HUB_OFFLINE", "1")
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("HUGGINGFACE_HUB_CACHE")
        .env_remove("XDG_CACHE_HOME")
        .args(["-p", "a red fox in the snow"]);
    command
}

fn error(command: &mut Command) -> String {
    let output = command.output().expect("run the CLI");
    assert!(!output.status.success());
    String::from_utf8(output.stderr).expect("UTF-8 diagnostic")
}

#[test]
fn default_and_raw_checkpoints_report_hf_cache_misses_offline() {
    let home = tempfile::tempdir().unwrap();
    for (args, checkpoint) in [
        (Vec::new(), "krea2_turbo_int8_convrot.safetensors"),
        (vec!["--checkpoint", "raw"], "krea2_raw_int8_convrot.safetensors"),
    ] {
        let message = error(cli(home.path()).args(args));
        assert!(message.contains(checkpoint), "{message}");
        assert!(message.contains("is not in the Hugging Face cache"), "{message}");
    }
}

#[test]
fn default_model_reuses_the_standard_cache_and_environment_overrides() {
    for setting in [None, Some("HF_HOME"), Some("HF_HUB_CACHE"), Some("XDG_CACHE_HOME")] {
        let home = tempfile::tempdir().unwrap();
        let custom = home.path().join("custom-cache");
        let cache = match setting {
            None => home.path().join(".cache/huggingface/hub"),
            Some("HF_HOME") => custom.join("hub"),
            Some("HF_HUB_CACHE") => custom.clone(),
            Some("XDG_CACHE_HOME") => custom.join("huggingface/hub"),
            _ => unreachable!(),
        };
        let repository = cache.join("models--Comfy-Org--Krea-2");
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = repository.join("snapshots").join(revision);
        std::fs::create_dir_all(repository.join("refs")).unwrap();
        std::fs::create_dir_all(snapshot.join("diffusion_models")).unwrap();
        std::fs::write(repository.join("refs/main"), revision).unwrap();
        std::fs::write(
            snapshot.join("diffusion_models/krea2_turbo_int8_convrot.safetensors"),
            b"",
        )
        .unwrap();
        let mut command = cli(home.path());
        if let Some(setting) = setting {
            command.env(setting, custom);
        }
        // Finding the checkpoint advances discovery to the missing encoder.
        // It never reaches model loading, GPU initialization, or a download.
        let message = error(&mut command);
        assert!(
            message.contains("text_encoders/qwen3vl_4b_bf16.safetensors")
                && message.contains("is not in the Hugging Face cache"),
            "{setting:?}: {message}"
        );
    }
}

#[test]
fn explicit_checkpoints_do_not_search_sibling_model_directories() {
    let home = tempfile::tempdir().unwrap();
    let checkpoint = home.path().join("diffusion_models/custom.safetensors");
    std::fs::create_dir_all(checkpoint.parent().unwrap()).unwrap();
    std::fs::create_dir_all(home.path().join("text_encoders")).unwrap();
    std::fs::write(&checkpoint, b"").unwrap();
    std::fs::write(home.path().join("text_encoders/qwen3vl_4b_bf16.safetensors"), b"").unwrap();
    let message = error(cli(home.path()).arg("--model").arg(&checkpoint));
    assert!(message.contains("text_encoders/qwen3vl_4b_bf16.safetensors"), "{message}");
    assert!(message.contains("is not in the Hugging Face cache"), "{message}");
}

#[test]
fn all_components_resolve_from_hf_snapshots_with_cached_encoder_fallback() {
    const CHILD: &str = "KREA2_TEST_CACHE_ROOT";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = std::path::PathBuf::from(root);
        let snapshot = root.join("models--Comfy-Org--Krea-2/snapshots/test-revision");
        let encoder = snapshot.join("text_encoders/qwen3vl_4b_bf16.safetensors");
        let fp8 = snapshot.join("text_encoders/qwen3vl_4b_fp8_scaled.safetensors");
        let vae = snapshot.join("vae/qwen_image_vae.safetensors");
        let request =
            krea2::models::Files::of(Path::new("krea2_turbo_int8_convrot")).offline(true);
        let files = request.clone().resolve().unwrap();
        assert_eq!(
            files.checkpoint,
            snapshot.join("diffusion_models/krea2_turbo_int8_convrot.safetensors")
        );
        assert_eq!(files.text_encoder, encoder);
        assert_eq!(files.vae, vae);
        assert!(files.distilled);
        std::fs::remove_file(encoder).unwrap();
        assert_eq!(request.clone().resolve().unwrap().text_encoder, fp8);
        std::fs::remove_file(vae).unwrap();
        let error = request.resolve().unwrap_err();
        assert!(error.0.contains("vae/qwen_image_vae.safetensors"), "{error}");
        assert!(error.0.contains("is not in the Hugging Face cache"), "{error}");
        return;
    }
    // A subprocess isolates HF's cached client and environment from other tests.
    let home = tempfile::tempdir().unwrap();
    let cache = home.path().join("hub");
    let repository = cache.join("models--Comfy-Org--Krea-2");
    std::fs::create_dir_all(repository.join("refs")).unwrap();
    std::fs::write(repository.join("refs/main"), "test-revision").unwrap();
    for name in [
        "diffusion_models/krea2_turbo_int8_convrot.safetensors",
        "text_encoders/qwen3vl_4b_bf16.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "vae/qwen_image_vae.safetensors",
    ] {
        let path = repository.join("snapshots/test-revision").join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "all_components_resolve_from_hf_snapshots_with_cached_encoder_fallback",
        ])
        .env(CHILD, &cache)
        .env("HF_HUB_CACHE", &cache)
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
