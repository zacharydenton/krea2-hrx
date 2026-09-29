//! Native preprocessing round-trip validation with external images and model weights.
use krea2::checkpoint::{Checkpoint, DType};
use krea2::models::{Files, Models};
use krea2::numerics::{from_f32, to_f32};
use krea2::ops::Tensor;
use krea2::training::{PreparedDataset, TrainConfig, dataset, sample_latent, vae::Encoder};
use rand::SeedableRng;

#[test]
#[ignore = "requires GPU, cached models and KREA2_TRAIN_TEST_CONFIG with prepared images"]
fn cached_posterior_reencodes_exactly_and_reconstructs_the_image() {
    let path = std::env::var_os("KREA2_TRAIN_TEST_CONFIG")
        .expect("set KREA2_TRAIN_TEST_CONFIG to an external prepared configuration");
    let config = TrainConfig::read(std::path::Path::new(&path)).unwrap();
    let data: PreparedDataset =
        serde_json::from_slice(&std::fs::read(config.output.join("prepared.json")).unwrap())
            .unwrap();
    let files = Files::of(&config.model)
        .distilled(Some(false))
        .text_encoder(config.text_encoder.as_deref())
        .vae(config.vae.as_deref())
        .offline(true)
        .resolve()
        .unwrap();
    let mut stream = hrx::Stream::open().unwrap();
    let models = Models::open(&mut stream, &files, None).unwrap();
    let encoder = Encoder::load(&mut stream, &files.vae).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(37);
    let mut shapes = std::collections::BTreeSet::new();
    let previews = config.output.join("validation/vae");
    std::fs::create_dir_all(&previews).unwrap();
    // Cover each dataset bucket, including portrait/landscape when available.
    for sample in &data.samples {
        if !shapes.insert((sample.width, sample.height)) {
            continue;
        }
        let source = dataset::crop(sample).unwrap();
        let pixels: Vec<u16> =
            source.as_raw().iter().map(|v| from_f32(f32::from(*v) / 127.5 - 1.0)).collect();
        let input = Tensor::from_slice(
            models.ops.pool(),
            &mut stream,
            &pixels,
            sample.width * sample.height,
            3,
        )
        .unwrap();
        let encoded = encoder
            .encode(&models.ops, &mut stream, &input, sample.height, sample.width)
            .unwrap()
            .download(&mut stream)
            .unwrap();
        let file = Checkpoint::open(
            &config.output.join("cache").join(format!("{}.posterior.safetensors", sample.key)),
        )
        .unwrap();
        let cached = file.get("posterior").unwrap();
        assert_eq!(cached.dtype, DType::BF16);
        let bits: Vec<u16> =
            cached.bytes.chunks_exact(2).map(|v| u16::from_le_bytes([v[0], v[1]])).collect();
        assert_eq!(encoded, bits, "cached posterior differs from fresh encoding");
        let moments = encoded.into_iter().map(to_f32).collect::<Vec<_>>();
        let packed =
            sample_latent(&moments, sample.height / 8, sample.width / 8, &mut rng).unwrap();
        let latent = Tensor::from_slice(
            models.ops.pool(),
            &mut stream,
            &packed.into_iter().map(from_f32).collect::<Vec<_>>(),
            sample.height / 16 * (sample.width / 16),
            64,
        )
        .unwrap();
        let rgb = models.decode(&mut stream, &latent, sample.width, sample.height).unwrap();
        assert_eq!(rgb.len(), source.as_raw().len());
        let mse = rgb
            .iter()
            .zip(source.as_raw())
            .map(|(a, b)| ((f64::from(*a) - f64::from(*b)) / 255.0).powi(2))
            .sum::<f64>()
            / rgb.len() as f64;
        let psnr = -10.0 * mse.log10();
        let label = format!("{}x{}", sample.width, sample.height);
        source.save(previews.join(format!("{label}-source.png"))).unwrap();
        image::RgbImage::from_raw(sample.width as u32, sample.height as u32, rgb)
            .unwrap()
            .save(previews.join(format!("{label}-reconstruction.png")))
            .unwrap();
        eprintln!("VAE {label}: PSNR {psnr:.2} dB; previews in {}", previews.display());
        assert!(psnr > 20.0, "VAE round trip lost image structure: PSNR {psnr:.2} dB");
    }
    assert!(!shapes.is_empty());
}
