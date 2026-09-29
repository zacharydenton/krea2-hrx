//! Native character-training command-line workflow.
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use krea2::training::{PreparedDataset, TrainConfig, Trainer};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "krea2-train",
    about = "Krea 2 RAW character LoRA training in Rust, Loom and HRX",
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
    /// Evaluate retained checkpoints with Turbo and fixed prompts/seeds
    Evaluate {
        #[arg(long)]
        run: PathBuf,
        /// Turbo BF16 file or supported model name
        #[arg(long, default_value = "krea2_turbo_bf16")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "0.9")]
        strengths: Vec<f32>,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Inspect { config } => {
            let config = TrainConfig::read(&config)?;
            krea2::training::model::Transformer::validate_checkpoint(&config.model)?;
            let data = PreparedDataset::scan(&config)?;
            println!(
                "{} captioned images; trigger {:?}; rank {}/alpha {}",
                data.samples.len(),
                config.trigger,
                config.rank,
                config.alpha
            );
            for sample in data.samples {
                println!("{}x{}  {}", sample.width, sample.height, sample.image.display());
            }
        }
        Command::Prepare { config } => {
            let config = TrainConfig::read(&config)?;
            let data = krea2::training::prepare::prepare(&config)?;
            println!("Prepared {} images in {}", data.samples.len(), config.output.display());
        }
        Command::Run { config, resume, stop_after } => {
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
        Command::Evaluate { run, model, strengths } => evaluate(run, model, strengths)?,
    }
    Ok(())
}

fn evaluate(run: PathBuf, model: PathBuf, strengths: Vec<f32>) -> Result<()> {
    use krea2::pipeline::{Files, Pipeline, Request};
    if strengths.is_empty() || strengths.iter().any(|v| !v.is_finite()) {
        bail!("strengths must be finite");
    }
    let config: TrainConfig = serde_json::from_slice(&std::fs::read(run.join("run.json"))?)?;
    let prompts = if config.validation_prompts.is_empty() {
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
    };
    let files = Files::of(&model)
        .distilled(Some(true))
        .text_encoder(config.text_encoder.as_deref())
        .vae(config.vae.as_deref())
        .offline(config.offline)
        .resolve()?;
    let mut checkpoints = std::fs::read_dir(run.join("checkpoints"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("adapter.safetensors").is_file() && p.join("state.json").is_file())
        .collect::<Vec<_>>();
    checkpoints.sort();
    if checkpoints.is_empty() {
        bail!("no complete checkpoints to evaluate");
    }
    for checkpoint in checkpoints {
        let adapter = krea2::lora::Adapter::load(&checkpoint.join("adapter.safetensors"))?;
        let label = checkpoint.file_name().context("checkpoint name")?;
        for &strength in &strengths {
            let out = run.join("evaluation").join(label).join(format!("strength-{strength}"));
            std::fs::create_dir_all(&out)?;
            let pipeline =
                Pipeline::open_with_adapter(files.clone(), None, &adapter, strength)?;
            for (index, prompt) in prompts.iter().enumerate() {
                for seed in [37, 38] {
                    let mut request = Request::new(prompt);
                    request.seed = seed;
                    let rgb = pipeline.generate(&request, None)?;
                    let image = image::RgbImage::from_raw(1024, 1024, rgb)
                        .context("RGB output dimensions")?;
                    let path = out.join(format!("prompt-{index:02}-seed-{seed}.png"));
                    image.save(&path)?;
                    println!("{}", path.display());
                }
            }
            std::fs::write(out.join("prompts.json"), serde_json::to_vec_pretty(&prompts)?)?;
        }
    }
    Ok(())
}
