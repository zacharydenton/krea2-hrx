//! Caption validation, deterministic image buckets and preprocessing manifests.
use super::TrainConfig;
use crate::lora::io;
use crate::{Error, Result};
use image::{ImageDecoder, ImageReader, RgbImage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// One selected image/caption and the exact crop/cache identity it uses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    /// Original image path.
    pub image: PathBuf,
    /// Caption after trimming surrounding whitespace.
    pub caption: String,
    /// Bucket width in pixels.
    pub width: usize,
    /// Bucket height in pixels.
    pub height: usize,
    /// Image-content hash; catches duplicates even under different filenames.
    pub image_hash: String,
    /// Fingerprint of the image, caption and bucket.
    pub key: String,
}

/// A stable sample order and model-bound preprocessing identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedDataset {
    /// Cache format version.
    pub version: u32,
    /// Samples in stable path order before epoch shuffling.
    pub samples: Vec<Sample>,
    /// Model/preprocessing fingerprint supplied by the preparation phase.
    pub fingerprint: String,
}

/// BLAKE3 of file contents, using bounded host memory.
pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut hash = blake3::Hasher::new();
    let mut chunk = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut chunk).map_err(io)?;
        if count == 0 {
            break;
        }
        hash.update(&chunk[..count]);
    }
    Ok(hash.finalize().to_hex().to_string())
}

/// Seven aspect buckets, with approximately resolution-squared pixels each.
pub fn buckets(resolution: usize) -> Vec<(usize, usize)> {
    [(1., 1.), (3., 4.), (4., 3.), (2., 3.), (3., 2.), (9., 16.), (16., 9.)]
        .into_iter()
        .map(|(w, h)| {
            let ratio: f64 = w / h;
            let width = ((resolution as f64 * ratio.sqrt()) as usize / 16) * 16;
            let height = ((resolution as f64 / ratio.sqrt()) as usize / 16) * 16;
            (width, height)
        })
        .collect()
}

fn images(directory: &Path, excluded: Option<&Path>, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory).map_err(io)? {
        let entry = entry.map_err(io)?;
        let kind = entry.file_type().map_err(io)?;
        if kind.is_dir() {
            if excluded != Some(entry.path().as_path()) {
                images(&entry.path(), excluded, paths)?;
            }
        } else if kind.is_file()
            && entry.path().extension().and_then(|v| v.to_str()).is_some_and(|v| {
                ["jpg", "jpeg", "png"].contains(&v.to_ascii_lowercase().as_str())
            })
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}

/// Decode with metadata orientation; JPEG/PNG only, entirely in Rust.
pub fn decode(path: &Path) -> Result<RgbImage> {
    let mut decoder = ImageReader::open(path)
        .map_err(io)?
        .with_guessed_format()
        .map_err(io)?
        .into_decoder()
        .map_err(io)?;
    let orientation = decoder.orientation().map_err(io)?;
    let mut image = image::DynamicImage::from_decoder(decoder).map_err(io)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

/// Resize to cover and center-crop, preserving the source's aspect ratio.
pub fn crop(sample: &Sample) -> Result<RgbImage> {
    let source = decode(&sample.image)?;
    let scale = (sample.width as f64 / source.width() as f64)
        .max(sample.height as f64 / source.height() as f64);
    let width = ((source.width() as f64 * scale).ceil() as u32).max(sample.width as u32);
    let height = ((source.height() as f64 * scale).ceil() as u32).max(sample.height as u32);
    let resized =
        image::imageops::resize(&source, width, height, image::imageops::FilterType::Lanczos3);
    Ok(image::imageops::crop_imm(
        &resized,
        (width - sample.width as u32) / 2,
        (height - sample.height as u32) / 2,
        sample.width as u32,
        sample.height as u32,
    )
    .to_image())
}

impl PreparedDataset {
    /// Validate every image and caption on CPU before loading any models.
    pub fn scan(config: &TrainConfig) -> Result<Self> {
        config.validate()?;
        let root = std::fs::canonicalize(&config.dataset).map_err(io)?;
        let excluded = if config.output.exists() {
            Some(std::fs::canonicalize(&config.output).map_err(io)?)
        } else {
            None
        };
        if excluded.as_deref() == Some(root.as_path()) {
            return Err(Error::invalid("dataset and output must be different directories"));
        }
        let mut paths = Vec::new();
        images(&root, excluded.as_deref(), &mut paths)?;
        paths.sort();
        if paths.is_empty() {
            return Err(Error::invalid("dataset contains no JPEG/PNG images"));
        }
        let mut seen = BTreeMap::new();
        let mut samples = Vec::new();
        let tokenizer = crate::tokenizer::Tokenizer::embedded()?;
        let choices = buckets(config.resolution);
        for path in paths {
            let caption_path = path.with_extension("txt");
            let caption = match std::fs::read_to_string(&caption_path) {
                Ok(text) => text,
                Err(e)
                    if config.mode == super::TrainingMode::Full
                        && e.kind() == std::io::ErrorKind::NotFound =>
                {
                    String::new()
                }
                Err(e) => {
                    return Err(Error::invalid(format!("{}: {e}", caption_path.display())));
                }
            }
            .trim()
            .to_owned();
            if config.mode == super::TrainingMode::Lora
                && (caption.is_empty() || !caption.contains(&config.trigger))
            {
                return Err(Error::invalid(format!(
                    "{}: caption must contain trigger {:?}",
                    caption_path.display(),
                    config.trigger
                )));
            }
            tokenizer.training_prompt(&caption).map_err(|error| {
                Error::invalid(format!("{}: {error}", caption_path.display()))
            })?;
            let image_hash = file_hash(&path)?;
            if let Some(other) = seen.insert(image_hash.clone(), path.clone()) {
                return Err(Error::invalid(format!(
                    "duplicate images: {} and {}",
                    other.display(),
                    path.display()
                )));
            }
            let image = decode(&path)?;
            let ratio = image.width() as f64 / image.height() as f64;
            let &(width, height) = choices
                .iter()
                .min_by(|(aw, ah), (bw, bh)| {
                    (ratio / (*aw as f64 / *ah as f64))
                        .ln()
                        .abs()
                        .total_cmp(&(ratio / (*bw as f64 / *bh as f64)).ln().abs())
                })
                .expect("nonempty buckets");
            let key = blake3::hash(
                &serde_json::to_vec(&(&image_hash, &caption, width, height)).map_err(io)?,
            )
            .to_hex()
            .to_string();
            samples.push(Sample { image: path, caption, width, height, image_hash, key });
        }
        Ok(Self { version: 1, samples, fingerprint: String::new() })
    }

    /// Save a small crop contact sheet beside the prepared dataset.
    pub fn contact_sheet(&self, path: &Path) -> Result<()> {
        let columns = 5;
        let mut sheet = RgbImage::new(
            columns * 192,
            self.samples.len().div_ceil(columns as usize) as u32 * 192,
        );
        for (index, sample) in self.samples.iter().enumerate() {
            let image = crop(sample)?;
            let thumb = image::imageops::thumbnail(&image, 192, 192);
            image::imageops::replace(
                &mut sheet,
                &thumb,
                (index as u32 % columns * 192).into(),
                (index as u32 / columns * 192).into(),
            );
        }
        sheet.save(path).map_err(io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_training_accepts_missing_and_empty_captions_but_rejects_invalid_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        RgbImage::new(32, 32).save(&path).unwrap();
        let c = TrainConfig {
            model: "raw.safetensors".into(),
            dataset: dir.path().into(),
            output: dir.path().join("run"),
            ..TrainConfig::full_preset()
        };
        let first = PreparedDataset::scan(&c).unwrap();
        assert!(first.samples[0].caption.is_empty());
        std::fs::write(path.with_extension("txt"), " \n").unwrap();
        assert_eq!(PreparedDataset::scan(&c).unwrap().samples, first.samples);
        std::fs::write(path.with_extension("txt"), "a new domain without a trigger").unwrap();
        assert_ne!(PreparedDataset::scan(&c).unwrap().samples[0].key, first.samples[0].key);
        std::fs::write(path.with_extension("txt"), [255u8]).unwrap();
        assert!(PreparedDataset::scan(&c).is_err());
        std::fs::remove_file(path.with_extension("txt")).unwrap();
        std::fs::create_dir(path.with_extension("txt")).unwrap();
        assert!(PreparedDataset::scan(&c).is_err());
    }
    #[test]
    fn buckets_preserve_square_and_portrait_landscape_pairs() {
        for resolution in [512, 768, 1024] {
            let b = buckets(resolution);
            assert_eq!(b[0], (resolution, resolution));
            for pair in b[1..].as_chunks::<2>().0 {
                assert_eq!(pair[0], (pair[1].1, pair[1].0));
            }
            assert!(
                b.iter().all(|(w, h)| w % 16 == 0
                    && h % 16 == 0
                    && w * h <= resolution * resolution)
            );
        }
    }
    #[test]
    fn dataset_rejects_duplicates_missing_captions_and_changes_cache_keys() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("one.png");
        RgbImage::from_pixel(32, 48, image::Rgb([20, 40, 60])).save(&path).unwrap();
        let c = TrainConfig {
            model: "raw.safetensors".into(),
            dataset: temp.path().into(),
            output: temp.path().join("run"),
            trigger: "bluej".into(),
            ..Default::default()
        };
        assert!(PreparedDataset::scan(&c).is_err());
        std::fs::write(path.with_extension("txt"), "bluej in a red shirt").unwrap();
        let first = PreparedDataset::scan(&c).unwrap();
        std::fs::write(path.with_extension("txt"), "bluej in a blue shirt").unwrap();
        let second = PreparedDataset::scan(&c).unwrap();
        assert_ne!(first.samples[0].key, second.samples[0].key);
        std::fs::create_dir(&c.output).unwrap();
        std::fs::copy(&path, c.output.join("crops.png")).unwrap();
        assert_eq!(PreparedDataset::scan(&c).unwrap().samples, second.samples);
        let cropped = crop(&second.samples[0]).unwrap();
        assert_eq!(
            cropped.dimensions(),
            (second.samples[0].width as u32, second.samples[0].height as u32)
        );
        std::fs::copy(&path, temp.path().join("two.png")).unwrap();
        std::fs::write(temp.path().join("two.txt"), "bluej").unwrap();
        assert!(PreparedDataset::scan(&c).unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn oversized_captions_fail_during_cpu_inspection() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("character.png");
        RgbImage::new(32, 32).save(&path).unwrap();
        std::fs::write(path.with_extension("txt"), format!("bluej {}", "word ".repeat(600)))
            .unwrap();
        let config = TrainConfig {
            model: "missing-model.safetensors".into(),
            dataset: temp.path().into(),
            output: temp.path().join("run"),
            trigger: "bluej".into(),
            ..Default::default()
        };
        let error = PreparedDataset::scan(&config).unwrap_err().to_string();
        assert!(error.contains("character.txt"));
        assert!(error.contains("conditioning tokens"));
        assert!(!config.output.exists());
    }
}
