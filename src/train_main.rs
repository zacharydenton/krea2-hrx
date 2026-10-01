//! Native character-training command-line workflow.
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use krea2::training::{PreparedDataset, TrainConfig, Trainer, TrainingMode};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "krea2-train",
    about = "Krea 2 RAW LoRA and full training in Rust, Loom and HRX",
    version
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the BF16 model, captions, images and buckets without opening the GPU
    Inspect {
        #[arg(long)]
        config: PathBuf,
    },
    /// Cache VAE posterior moments and frozen RAW text conditioning
    Prepare {
        #[arg(long)]
        config: PathBuf,
    },
    /// Train a prepared dataset or resume an update-boundary checkpoint
    Run {
        #[arg(long, required_unless_present = "resume", conflicts_with = "resume")]
        config: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Save and stop after this many additional updates, preserving the full run target
        #[arg(long)]
        stop_after: Option<std::num::NonZeroUsize>,
    },
    /// Manage immutable full-model exports and offline snapshot averages
    Checkpoint {
        #[command(subcommand)]
        command: CheckpointCommand,
    },
    /// Evaluate LoRAs with Turbo or full checkpoints with RAW, using fixed prompts/seeds
    Evaluate {
        #[arg(long, required_unless_present = "checkpoint", conflicts_with = "checkpoint")]
        run: Option<PathBuf>,
        /// A standalone full-model artifact directory
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Evaluation destination for a standalone artifact
        #[arg(long, requires = "checkpoint", conflicts_with = "run")]
        output: Option<PathBuf>,
        /// Turbo BF16 file or supported model name
        #[arg(long, default_value = "krea2_turbo_bf16")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "0.9")]
        strengths: Vec<f32>,
    },
}

#[derive(Subcommand)]
enum CheckpointCommand {
    /// List full-model artifacts and their retention status
    List {
        #[arg(long)]
        run: PathBuf,
    },
    /// Export immutable model weights without optimizer state
    Export {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Protect an artifact from automatic retention, or remove that protection
    Pin {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        unpin: bool,
    },
    /// Average 2-16 raw snapshots offline; no GPU or Python runtime is used
    Average {
        #[arg(long, required = true, num_args=2..=16)]
        checkpoints: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        /// Exponential half-life in optimizer updates; omit for a uniform average
        #[arg(long)]
        half_life_steps: Option<f64>,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Inspect { config } => {
            let config = TrainConfig::read(&config)?;
            if config.mode == TrainingMode::Full {
                let file = krea2::checkpoint::Checkpoint::open(&config.model)?;
                krea2::training::full::spec::validate(&file)?;
            } else {
                krea2::training::model::Transformer::validate_checkpoint(&config.model)?;
            }
            let data = PreparedDataset::scan(&config)?;
            if config.mode == TrainingMode::Full {
                let captioned = data.samples.iter().filter(|s| !s.caption.is_empty()).count();
                println!(
                    "Full RAW tuning: 430 tensors; {} captioned + {} uncaptioned images; {:.2} GiB persistent parameters",
                    captioned,
                    data.samples.len() - captioned,
                    krea2::training::full::spec::persistent_bytes(
                        &krea2::training::full::spec::inventory()
                    ) as f64
                        / (1u64 << 30) as f64
                );
            } else {
                println!(
                    "{} captioned images; trigger {:?}; rank {}/alpha {}",
                    data.samples.len(),
                    config.trigger,
                    config.rank,
                    config.alpha
                );
            }
            for sample in data.samples {
                println!("{}x{}  {}", sample.width, sample.height, sample.image.display());
            }
        }
        Command::Prepare { config } => {
            prefer_oom_victim()?;
            let config = TrainConfig::read(&config)?;
            let data = krea2::training::prepare::prepare(&config)?;
            println!("Prepared {} images in {}", data.samples.len(), config.output.display());
        }
        Command::Run { config, resume, stop_after } => {
            prefer_oom_victim()?;
            let mut trainer = match resume {
                Some(path) => Trainer::resume(&path)?,
                None => Trainer::open(TrainConfig::read(&config.context("missing config")?)?)?,
            };
            let mut log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(trainer.config().output.join("training.jsonl"))?;
            let total = trainer.config().steps;
            let starting_step = trainer.step();
            let mut log_error = None;
            trainer.run(|step, loss, norm, seconds| {
                eprintln!("step {step}/{total}  loss {loss:.6}  grad {norm:.5}  {seconds:.2}s");
                let row = serde_json::json!({
                    "step": step, "loss": loss, "grad_norm": norm, "seconds": seconds
                });
                if let Err(e) = writeln!(log, "{row}").and_then(|_| log.flush()) {
                    log_error = Some(e);
                    return false;
                }
                stop_after.is_none_or(|limit| step - starting_step < limit.get())
            })?;
            if let Some(e) = log_error {
                return Err(e.into());
            }
        }
        Command::Evaluate { run, checkpoint, output, model, strengths } => {
            if let Some(path) = checkpoint {
                evaluate_artifact(path, output)?;
            } else {
                evaluate(run.context("missing run")?, model, strengths)?;
            }
        }
        Command::Checkpoint { command } => {
            use krea2::training::full::artifacts::{self, Averaging};
            match command {
                CheckpointCommand::List { run } => {
                    println!("{}", serde_json::to_string_pretty(&artifacts::list(&run)?)?)
                }
                CheckpointCommand::Export { checkpoint, output } => {
                    let m = artifacts::export(&checkpoint, &output)?;
                    println!(
                        "Exported step {} to {} (model only; cannot resume)",
                        m.step,
                        output.display()
                    );
                }
                CheckpointCommand::Pin { checkpoint, unpin } => {
                    artifacts::pin(&checkpoint, !unpin)?;
                    println!(
                        "{} {}",
                        if unpin { "Unpinned" } else { "Pinned" },
                        checkpoint.display()
                    );
                }
                CheckpointCommand::Average { checkpoints, output, half_life_steps } => {
                    prefer_oom_victim()?;
                    let method = half_life_steps
                        .map_or(Averaging::Uniform, |half_life_steps| Averaging::Exponential {
                            half_life_steps,
                        });
                    let m = artifacts::average(&checkpoints, &output, method)?;
                    println!(
                        "Averaged {} snapshots through step {} into {}",
                        m.sources.len(),
                        m.step,
                        output.display()
                    );
                }
            }
        }
    }
    Ok(())
}

fn prefer_oom_victim() -> Result<()> {
    // GPU-pinned RAM is not fully reflected in process RSS. Prefer terminating
    // this disposable training process over the user's desktop if RAM runs out.
    std::fs::write("/proc/self/oom_score_adj", "1000")
        .context("could not set training process OOM priority")
}

fn evaluate(run: PathBuf, model: PathBuf, strengths: Vec<f32>) -> Result<()> {
    use krea2::pipeline::{Files, Pipeline};
    if strengths.is_empty() || strengths.iter().any(|v| !v.is_finite()) {
        bail!("strengths must be finite");
    }
    let config: TrainConfig = serde_json::from_slice(&std::fs::read(run.join("run.json"))?)?;
    let prompts = evaluation_prompts(&config)?;
    let files = if config.mode == TrainingMode::Lora {
        Some(
            Files::of(&model)
                .distilled(Some(true))
                .text_encoder(config.text_encoder.as_deref())
                .vae(config.vae.as_deref())
                .offline(config.offline)
                .resolve()?,
        )
    } else {
        None
    };
    let mut checkpoints = std::fs::read_dir(run.join("checkpoints"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.join(if config.mode == TrainingMode::Full {
                "model.safetensors"
            } else {
                "adapter.safetensors"
            })
            .is_file()
                && p.join("state.json").is_file()
        })
        .collect::<Vec<_>>();
    checkpoints.sort();
    if checkpoints.is_empty() {
        bail!("no complete checkpoints to evaluate");
    }
    for checkpoint in checkpoints {
        let _artifact = if checkpoint.join("artifact.json").is_file() {
            Some(krea2::training::full::artifacts::Artifact::open(&checkpoint)?)
        } else {
            None
        };
        let adapter = if config.mode == TrainingMode::Lora {
            Some(krea2::lora::Adapter::load(&checkpoint.join("adapter.safetensors"))?)
        } else {
            None
        };
        let label = checkpoint.file_name().context("checkpoint name")?;
        let weights: &[f32] =
            if config.mode == TrainingMode::Full { &[1.0] } else { &strengths };
        for &strength in weights {
            let out =
                run.join("evaluation").join(label).join(if config.mode == TrainingMode::Full {
                    "raw".to_owned()
                } else {
                    format!("strength-{strength}")
                });
            std::fs::create_dir_all(&out)?;
            let pipeline = if let Some(adapter) = &adapter {
                Pipeline::open_with_adapter(
                    files.as_ref().expect("Turbo files").clone(),
                    None,
                    adapter,
                    strength,
                )?
            } else {
                let files = Files::of(&checkpoint.join("model.safetensors"))
                    .distilled(Some(false))
                    .text_encoder(config.text_encoder.as_deref())
                    .vae(config.vae.as_deref())
                    .offline(config.offline)
                    .resolve()?;
                Pipeline::open(files, None)?
            };
            render(&pipeline, &prompts, &out)?;
        }
    }
    Ok(())
}

fn evaluation_prompts(config: &TrainConfig) -> Result<Vec<String>> {
    Ok(if config.validation_prompts.is_empty() && config.mode == TrainingMode::Full {
        bail!(
            "full checkpoint evaluation requires validation_prompts describing the training domain"
        );
    } else if config.validation_prompts.is_empty() {
        [
            "portrait photograph of {trigger} in soft daylight",
            "profile photograph of {trigger} outdoors",
            "{trigger} laughing, candid photograph",
            "{trigger} wearing a blue suit in a library",
            "{trigger} on a snowy mountain",
            "full body photograph of {trigger} walking in a park",
        ]
        .map(|p| p.replace("{trigger}", &config.trigger))
        .to_vec()
    } else {
        config.validation_prompts.clone()
    })
}

fn render(
    pipeline: &krea2::pipeline::Pipeline,
    prompts: &[String],
    out: &std::path::Path,
) -> Result<()> {
    use krea2::pipeline::Request;
    for (index, prompt) in prompts.iter().enumerate() {
        for seed in [37, 38] {
            let mut request = Request::new(prompt);
            request.seed = seed;
            let rgb = pipeline.generate(&request, None)?;
            let image =
                image::RgbImage::from_raw(1024, 1024, rgb).context("RGB output dimensions")?;
            let path = out.join(format!("prompt-{index:02}-seed-{seed}.png"));
            image.save(&path)?;
            println!("{}", path.display());
        }
    }
    std::fs::write(out.join("prompts.json"), serde_json::to_vec_pretty(prompts)?)?;
    Ok(())
}

fn evaluate_artifact(path: PathBuf, output: Option<PathBuf>) -> Result<()> {
    use krea2::pipeline::{Files, Pipeline};
    let artifact = krea2::training::full::artifacts::Artifact::open(&path)?;
    let config = &artifact.manifest().config;
    let prompts = evaluation_prompts(config)?;
    let model = artifact.path().join("model.safetensors");
    krea2::training::full::spec::validate(&krea2::checkpoint::Checkpoint::open(&model)?)?;
    let files = Files::of(&model)
        .distilled(Some(false))
        .text_encoder(config.text_encoder.as_deref())
        .vae(config.vae.as_deref())
        .offline(config.offline)
        .resolve()?;
    let output = output.unwrap_or_else(|| {
        artifact
            .path()
            .parent()
            .unwrap()
            .join("evaluation")
            .join(artifact.path().file_name().unwrap())
    });
    std::fs::create_dir_all(&output)?;
    let pipeline = Pipeline::open(files, None)?;
    render(&pipeline, &prompts, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_cli_and_evaluation_destinations() {
        for args in [
            vec!["krea2-train", "checkpoint", "list", "--run", "run"],
            vec!["krea2-train", "checkpoint", "export", "--checkpoint", "a", "--output", "b"],
            vec!["krea2-train", "checkpoint", "pin", "--checkpoint", "a", "--unpin"],
            vec![
                "krea2-train",
                "checkpoint",
                "average",
                "--checkpoints",
                "a",
                "b",
                "--output",
                "c",
                "--half-life-steps",
                "100",
            ],
            vec!["krea2-train", "evaluate", "--checkpoint", "a", "--output", "images"],
            vec!["krea2-train", "evaluate", "--run", "run"],
        ] {
            assert!(Args::try_parse_from(&args).is_ok(), "{args:?}");
        }
        for args in [
            vec!["krea2-train", "evaluate"],
            vec!["krea2-train", "evaluate", "--run", "run", "--checkpoint", "a"],
            vec!["krea2-train", "evaluate", "--run", "run", "--output", "images"],
            vec!["krea2-train", "checkpoint", "average", "--checkpoints", "a", "--output", "c"],
        ] {
            assert!(Args::try_parse_from(&args).is_err(), "{args:?}");
        }
    }
}
