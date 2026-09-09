//! Component test harness: `<checkpoint> <directory>`.
//! Reads float32 `text.bin` and `hidden.bin`; writes conditioning, time embedding,
//! modulation and final-layer outputs for comparison with reference implementations.
use std::path::{Path, PathBuf};

use hrx::Stream;
use krea2_models::{Files, Models, MODULATION_ELEMENTS};
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::Tensor;

fn main() -> std::process::ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, checkpoint, directory] = &arguments[..] else {
        eprintln!("usage: krea2-native-components <checkpoint> <directory>");
        return std::process::ExitCode::from(2);
    };
    match run(Path::new(checkpoint), Path::new(directory)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(checkpoint: &Path, directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = Stream::open()?;
    let files = Files::of(checkpoint).resolve()?;
    let models = Models::open(&mut stream, &files, None)?;

    let text = read(&mut stream, &models, &directory.join("text.bin"), 2560)?;
    let condition = models.text_fusion(&mut stream, &text)?;
    write(&mut stream, &directory.join("condition.bin"), &condition)?;

    let (embedding, modulation) = models.time(&mut stream, 0.75)?;
    write(&mut stream, &directory.join("temb.bin"), &embedding)?;
    write(&mut stream, &directory.join("mod.bin"), &modulation)?;

    let tables = models.modulation(&mut stream, &modulation)?;
    let mut values = vec![0f32; MODULATION_ELEMENTS];
    stream.read(tables.binding(), bytemuck::cast_slice_mut(&mut values))?;
    std::fs::write(directory.join("block_mod.bin"), bytemuck::cast_slice(&values))?;

    let hidden = read(&mut stream, &models, &directory.join("hidden.bin"), 6144)?;
    let final_out = models.last(&mut stream, &hidden, &embedding)?;
    write(&mut stream, &directory.join("final.bin"), &final_out)?;
    Ok(())
}

/// A float32 fixture, rounded to bf16 on the way to the device.
fn read(
    stream: &mut Stream,
    models: &Models,
    path: &PathBuf,
    cols: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let values: Vec<u16> =
        bytemuck::cast_slice::<u8, f32>(&bytes).iter().map(|&v| from_f32(v)).collect();
    Ok(Tensor::from_slice(models.ops.pool(), stream, &values, values.len() / cols, cols)?)
}

fn write(
    stream: &mut Stream,
    path: &PathBuf,
    tensor: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    let values: Vec<f32> = tensor.download(stream)?.into_iter().map(to_f32).collect();
    std::fs::write(path, bytemuck::cast_slice(&values))?;
    Ok(())
}
