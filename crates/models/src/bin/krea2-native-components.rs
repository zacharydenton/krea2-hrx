//! The component oracle harness: `<checkpoint> <directory>`.
//!
//! Reads `text.bin` and `hidden.bin` as float32, writes `condition.bin`,
//! `temb.bin`, `mod.bin`, `block_mod.bin` and `final.bin` the same way, so the
//! Python regression tests can compare each stage of the graph against Torch —
//! and so this build can be compared against the C++ one file by file.
//!
//! Not part of the inference API.
use std::path::{Path, PathBuf};

use hrx::device;
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
    let files = Files::of(checkpoint).resolve()?;
    let models = Models::open(&files)?;

    let text = read(&models, &directory.join("text.bin"), 2560)?;
    write(&directory.join("condition.bin"), &models.text_fusion(&text)?)?;

    let (embedding, modulation) = models.time(0.75)?;
    write(&directory.join("temb.bin"), &embedding)?;
    write(&directory.join("mod.bin"), &modulation)?;

    let tables = models.modulation(&modulation)?;
    let mut values = vec![0f32; MODULATION_ELEMENTS];
    device().read(&mut values, tables.ptr())?;
    std::fs::write(directory.join("block_mod.bin"), bytemuck::cast_slice(&values))?;

    let hidden = read(&models, &directory.join("hidden.bin"), 6144)?;
    write(&directory.join("final.bin"), &models.last(&hidden, &embedding)?)?;
    Ok(())
}

/// A float32 fixture, rounded to bf16 on the way to the device.
fn read(
    models: &Models,
    path: &PathBuf,
    cols: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let values: Vec<u16> =
        bytemuck::cast_slice::<u8, f32>(&bytes).iter().map(|&v| from_f32(v)).collect();
    Ok(Tensor::from_slice(models.ops.pool(), &values, values.len() / cols, cols)?)
}

fn write(path: &PathBuf, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let values: Vec<f32> = tensor.download()?.into_iter().map(to_f32).collect();
    std::fs::write(path, bytemuck::cast_slice(&values))?;
    Ok(())
}
