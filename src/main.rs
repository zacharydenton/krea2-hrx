//! Command-line interface, model discovery and image output for `krea2::pipeline`.
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser};
use krea2::pipeline::{Files, Pipeline, Request};

#[derive(Parser)]
#[command(
    name = "krea2",
    about = env!("CARGO_PKG_DESCRIPTION"),
    after_help = "The prompt comes from -p or from stdin. Steps and guidance default to the \
                  checkpoint: Turbo 8 unguided, Raw 52 at 3.5. KREA2_NATIVE_PROFILE=1 prints \
                  per-stage times.",
    version
)]
struct Args {
    /// Backend for the first text-fusion up projection; auto requires saved qualification
    #[arg(long, value_enum, default_value = "auto")]
    fusion_backend: krea2::fusion::FusionBackend,
    /// The prompt; read from stdin when absent
    #[arg(short, long)]
    prompt: Option<String>,
    /// Negative prompt (guided sampling only)
    #[arg(short, long)]
    negative: Option<String>,
    /// Output image (.png or .ppm)
    #[arg(short, long, default_value = "image.png")]
    out: PathBuf,
    /// Images to generate, from consecutive seeds, named out-0, out-1, ...
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=1000))]
    images: u32,
    #[arg(long, default_value_t = 1024)]
    width: i32,
    #[arg(long, default_value_t = 1024)]
    height: i32,
    /// Sampling steps (default: the checkpoint's)
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..=100))]
    steps: Option<i32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Classifier-free guidance, cond + g * (cond - uncond)
    #[arg(long)]
    guidance: Option<f32>,
    /// Turbo or Raw sampler; required with a custom --model file (default model: Turbo)
    #[arg(long, value_parser = ["turbo", "raw"])]
    checkpoint: Option<String>,
    /// Attention kernels, chosen when a sequence length is first compiled
    #[arg(long, value_parser = ["f16", "i8", "i4"])]
    attn: Option<String>,
    /// The int8 ConvRot checkpoint: an explicit file or a name in Comfy-Org/Krea-2
    #[arg(long)]
    model: Option<PathBuf>,
    /// Qwen3-VL-4B text encoder file (default: Hugging Face cache)
    #[arg(long)]
    text_encoder: Option<PathBuf>,
    /// Qwen-Image VAE file (default: Hugging Face cache)
    #[arg(long)]
    vae: Option<PathBuf>,
    /// Loom shared library override (default: HRX_LOOM_LIBRARY or the pinned bundle)
    #[arg(long)]
    compiler_library: Option<PathBuf>,
    /// Only the output lines
    #[arg(short, long)]
    quiet: bool,
}

/// The name for image `index` of `count`: "fox.png" alone, else "fox-3.png".
fn numbered(path: &Path, index: u32, count: u32) -> PathBuf {
    if count == 1 {
        return path.to_path_buf();
    }
    let stem = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let mut name = format!("{stem}-{index}");
    if let Some(extension) = path.extension() {
        name.push('.');
        name.push_str(&extension.to_string_lossy());
    }
    path.with_file_name(name)
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Png,
    Ppm,
}

fn output_format(path: &Path) -> std::result::Result<OutputFormat, &'static str> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("png") => Ok(OutputFormat::Png),
        Some(extension) if extension.eq_ignore_ascii_case("ppm") => Ok(OutputFormat::Ppm),
        _ => Err("--out must have a .png or .ppm extension"),
    }
}

fn write_image(path: &Path, rgb: &[u8], width: i32, height: i32) -> Result<()> {
    // Validate before creating the file so an unsupported extension never
    // truncates an existing output or leaves misleading image bytes behind.
    let format = output_format(path).map_err(anyhow::Error::msg)?;
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot write {}", path.display()))?;
    let mut file = std::io::BufWriter::new(file);
    match format {
        OutputFormat::Png => {
            let mut encoder = png::Encoder::new(&mut file, width as u32, height as u32);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(rgb)?;
            writer.finish()?;
        }
        OutputFormat::Ppm => {
            write!(file, "P6\n{width} {height}\n255\n")?;
            file.write_all(rgb)?;
        }
    }
    file.flush()?;
    Ok(())
}

fn usage(message: impl Into<String>) -> anyhow::Error {
    Args::command().error(clap::error::ErrorKind::ValueValidation, message.into()).into()
}

/// One line of sampling progress, redrawn in place, with the time left.
/// Returning true carries on; the pipeline stops on false.
fn report(step: usize, steps: usize, seconds: f64) -> bool {
    let left = if step > 0 { seconds / step as f64 * (steps - step) as f64 } else { 0.0 };
    eprint!("\r  step {step}/{steps}  {seconds:.1} s  ({left:.0} s left)   ");
    if step == steps {
        eprintln!();
    }
    let _ = std::io::stderr().flush();
    true
}

fn prompt_from_stdin() -> Result<String> {
    if std::io::stdin().is_terminal() {
        return Ok(String::new());
    }
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    Ok(text.trim_end_matches(['\n', '\r']).to_string())
}

fn run(args: Args) -> Result<()> {
    let prompt = match &args.prompt {
        Some(prompt) => prompt.clone(),
        None => prompt_from_stdin()?,
    };
    if prompt.trim().is_empty() {
        return Err(usage("no prompt (give -p \"...\" or pipe it on stdin)"));
    }
    if args.width % 16 != 0 || args.height % 16 != 0 {
        return Err(usage("--width and --height must be multiples of 16"));
    }
    if !(64..=2048).contains(&args.width) || !(64..=2048).contains(&args.height) {
        return Err(usage("--width and --height must be between 64 and 2048"));
    }
    if args.guidance.is_some_and(|g| !(0.0..=100.0).contains(&g)) {
        return Err(usage("--guidance must be between 0 and 100"));
    }
    output_format(&args.out).map_err(usage)?;
    if args.seed.checked_add(u64::from(args.images - 1)).is_none() {
        return Err(usage("--seed plus --images exceeds the maximum seed"));
    }
    let named = args.checkpoint.as_deref().unwrap_or("turbo");
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("krea2_{named}_int8_convrot")));
    let directory = args.out.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(directory) = directory {
        if !directory.is_dir() {
            bail!(
                "cannot write {}: {} is not a directory",
                args.out.display(),
                directory.display()
            );
        }
    }
    if let Some(width) = &args.attn {
        // Read by the kernel builder when a sequence length is first compiled.
        std::env::set_var(
            "KREA2_ATTN_QK",
            match width.as_str() {
                "f16" => "16",
                "i8" => "8",
                _ => "4",
            },
        );
    }

    let loading = Instant::now();
    let request = Files::of(&model)
        .text_encoder(args.text_encoder.as_deref())
        .vae(args.vae.as_deref())
        .distilled(args.checkpoint.as_deref().map(|choice| choice == "turbo"));
    request
        .is_distilled()
        .map_err(|error| usage(format!("{error}; pass --checkpoint turbo or raw")))?;
    let files = request.resolve()?;
    let compiler =
        args.compiler_library.as_ref().map(|path| path.to_string_lossy().into_owned());
    let pipeline = Pipeline::with_options(
        files,
        compiler.as_deref(),
        krea2::pipeline::PipelineOptions { fusion_backend: args.fusion_backend },
    )?;
    let load = loading.elapsed().as_secs_f64();
    let steps = args.steps.unwrap_or(if pipeline.distilled() { 8 } else { 52 });
    let guidance = args.guidance.unwrap_or(if pipeline.distilled() { 0.0 } else { 3.5 });
    if !args.quiet {
        eprintln!(
            "{} {}x{}, {steps} steps, guidance {guidance:.2}, seed {}{}: {} image{}, loaded in {load:.1} s",
            if pipeline.distilled() { "turbo" } else { "raw" },
            args.width,
            args.height,
            args.seed,
            if guidance > 0.0 { " (two forwards per step)" } else { "" },
            args.images,
            if args.images == 1 { "" } else { "s" },
        );
    }

    for index in 0..args.images {
        let began = Instant::now();
        let request = Request {
            prompt: &prompt,
            negative_prompt: args.negative.as_deref().unwrap_or(""),
            width: args.width as usize,
            height: args.height as usize,
            steps: args.steps.map(|steps| steps as usize),
            guidance: args.guidance,
            seed: args.seed + u64::from(index),
            initial_latents: None,
        };
        let rgb = pipeline.generate(
            &request,
            (!args.quiet).then_some(&mut report as krea2::pipeline::Progress),
        )?;
        if !args.quiet {
            eprintln!("  fusion: {}", pipeline.fusion_selection());
        }
        let path = numbered(&args.out, index, args.images);
        write_image(&path, &rgb, args.width, args.height)?;
        println!(
            "{}  seed {}  {:.2} s",
            path.display(),
            args.seed + u64::from(index),
            began.elapsed().as_secs_f64()
        );
        let _ = std::io::stdout().flush();
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(error) = run(args) {
        if let Some(error) = error.downcast_ref::<clap::Error>() {
            error.exit();
        }
        eprintln!("krea2: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_image_keeps_the_name_and_more_are_numbered() {
        let path = Path::new("build/fox.png");
        assert_eq!(numbered(path, 0, 1), PathBuf::from("build/fox.png"));
        assert_eq!(numbered(path, 0, 4), PathBuf::from("build/fox-0.png"));
        assert_eq!(numbered(path, 3, 4), PathBuf::from("build/fox-3.png"));
        // A name without an extension, and a directory with a dot in it.
        assert_eq!(numbered(Path::new("out"), 2, 3), PathBuf::from("out-2"));
        assert_eq!(numbered(Path::new("v1.2/fox"), 1, 2), PathBuf::from("v1.2/fox-1"));
    }

    #[test]
    fn unsupported_formats_do_not_create_or_truncate_outputs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("image.jpg");
        assert!(write_image(&path, &[1, 2, 3], 1, 1).is_err());
        assert!(!path.exists());
        std::fs::write(&path, b"original").unwrap();
        assert!(write_image(&path, &[1, 2, 3], 1, 1).is_err());
        assert_eq!(std::fs::read(path).unwrap(), b"original");
        assert!(matches!(output_format(Path::new("image.PNG")), Ok(OutputFormat::Png)));
        assert!(matches!(output_format(Path::new("image.PPM")), Ok(OutputFormat::Ppm)));
    }

    #[test]
    fn png_round_trips_and_ppm_carries_its_header() {
        let directory = std::env::temp_dir().join(format!("krea2-cli-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let (width, height): (i32, i32) = (7, 5);
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i * 7 % 256) as u8).collect();

        let png_path = directory.join("image.png");
        write_image(&png_path, &rgb, width, height).unwrap();
        let decoder = png::Decoder::new(std::fs::File::open(&png_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut decoded).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(&decoded[..info.buffer_size()], &rgb[..]);

        let ppm_path = directory.join("image.ppm");
        write_image(&ppm_path, &rgb, width, height).unwrap();
        let bytes = std::fs::read(&ppm_path).unwrap();
        let header = format!("P6\n{width} {height}\n255\n");
        assert!(bytes.starts_with(header.as_bytes()));
        assert_eq!(&bytes[header.len()..], &rgb[..]);
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
