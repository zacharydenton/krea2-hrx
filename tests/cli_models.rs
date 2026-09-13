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
fn an_explicit_models_directory_still_supplies_the_default_checkpoint() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    std::fs::create_dir_all(models.join("diffusion_models")).unwrap();
    let checkpoint = models.join("diffusion_models/krea2_turbo_int8_convrot.safetensors");
    std::fs::write(&checkpoint, b"").unwrap();
    let message = error(cli(home.path()).arg("--models").arg(&models));
    assert!(message.contains("the text encoder was not found"), "{message}");
}
