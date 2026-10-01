//! Full-model training validation with an external prepared dataset.
//! No checkpoint, dataset, or result fixture is stored in this repository.
use krea2::training::{TrainConfig, Trainer, dataset::file_hash};

#[test]
#[ignore = "requires a GPU, RAW weights and KREA2_TRAIN_TEST_CONFIG with prepared caches"]
fn checkpoint_resume_matches_two_uninterrupted_updates() {
    let path = std::env::var_os("KREA2_TRAIN_TEST_CONFIG")
        .expect("set KREA2_TRAIN_TEST_CONFIG to an external prepared configuration");
    let mut config = TrainConfig::read(std::path::Path::new(&path)).unwrap();
    let model_file = if config.mode == krea2::training::TrainingMode::Full {
        "model.safetensors"
    } else {
        "adapter.safetensors"
    };
    // Checkpoints can exceed a GiB; keep them beside the external run rather
    // than on a potentially RAM-backed /tmp filesystem.
    let temporary = tempfile::tempdir_in(config.output.parent().unwrap()).unwrap();
    let output = temporary.path().join("run");
    std::fs::create_dir(&output).unwrap();
    std::fs::copy(config.output.join("prepared.json"), output.join("prepared.json")).unwrap();
    std::os::unix::fs::symlink(config.output.join("cache"), output.join("cache")).unwrap();
    config.output = output.clone();
    config.steps = 2;
    config.keep_checkpoints = 4;
    if config.mode == krea2::training::TrainingMode::Full {
        config.snapshot_every = Some(1);
        config.keep_snapshots = 2;
    }

    let allowance = config.memory_gib * (1usize << 30);
    let manager = hrx::residency::ResidencyManager::new(allowance + 4096).unwrap();
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let sibling = manager.budget().reserve(4096).unwrap();
    let mut uninterrupted = Trainer::open_in(config, &context).unwrap();
    assert_eq!(manager.budget().reserved_bytes(), allowance + 4096);
    eprintln!("validating uninterrupted update 1");
    let first = uninterrupted.train_step().unwrap().unwrap();
    assert_eq!(first.step, 1);
    assert!(first.loss.is_finite() && first.loss > 0.0);
    assert!(first.gradient_norm.is_finite() && first.gradient_norm > 0.0);
    let checkpoint = uninterrupted.save().unwrap();
    eprintln!("update 1 loss {}; validating uninterrupted update 2", first.loss);
    let second = uninterrupted.train_step().unwrap().unwrap();
    assert_eq!(second.step, 2);
    assert!(second.loss.is_finite() && second.loss > 0.0);
    assert!(second.gradient_norm.is_finite() && second.gradient_norm > 0.0);
    assert!(uninterrupted.train_step().unwrap().is_none());
    let complete = uninterrupted.save().unwrap();
    if model_file == "model.safetensors" {
        let snapshot = output.join("snapshots/step-000002");
        assert_eq!(
            file_hash(&snapshot.join(model_file)).unwrap(),
            file_hash(&complete.join(model_file)).unwrap()
        );
        assert!(!snapshot.join("optimizer.safetensors").exists());
    }
    let reference = [model_file, "optimizer.safetensors", "state.json"]
        .map(|name| (name, file_hash(&complete.join(name)).unwrap()));
    // A full checkpoint is about 48 GiB; compare hashes without retaining a third copy.
    std::fs::remove_dir_all(&complete).unwrap();
    drop(uninterrupted);
    assert_eq!(manager.budget().reserved_bytes(), 4096);

    eprintln!("update 2 loss {}; validating resumed update 2", second.loss);
    let mut resumed = Trainer::resume_in(&checkpoint, &context).unwrap();
    assert_eq!(manager.budget().reserved_bytes(), allowance + 4096);
    assert_eq!(resumed.step(), 1);
    let resumed_second = resumed.train_step().unwrap().unwrap();
    assert_eq!(second.loss, resumed_second.loss);
    assert_eq!(second.gradient_norm, resumed_second.gradient_norm);
    let resumed_path = resumed.save().unwrap();
    drop(resumed);
    assert_eq!(manager.budget().reserved_bytes(), 4096);
    drop(sibling);
    assert_eq!(manager.budget().reserved_bytes(), 0);
    for (file, hash) in reference {
        assert_eq!(
            hash,
            file_hash(&resumed_path.join(file)).unwrap(),
            "resume differs in {file}"
        );
    }
}
