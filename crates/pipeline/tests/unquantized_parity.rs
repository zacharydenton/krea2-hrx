//! Accuracy against the original unquantized BF16 model at the fixture resolution.
//! Identical noise, conditioning and scheduler; no quantized reference model.
//! Run scripts/parity.sh with local immutable checkpoints and reference fixtures.
use std::path::{Path, PathBuf};

use krea2_models::Files;
use krea2_pipeline::Pipeline;

const REFERENCE: [&str; 5] = ["job.json", "noise.npy", "text.npy", "bf16.npy", "bf16.png"];
const BASELINE: [&str; 2] = ["w8a8.npy", "w8a8.png"];

/// Minting is a separate workflow, against an explicitly pinned reference with no baseline.
/// An ordinary gate always requires all seven hashes and files, before opening the GPU.
fn validate_fixture(
    fixture: &Path,
    manifest: &serde_json::Value,
    minting: bool,
) -> Result<(), String> {
    let files = manifest["files"].as_object().ok_or("manifest must contain file hashes")?;
    let required: Vec<_> = REFERENCE
        .iter()
        .chain(if minting { [].iter() } else { BASELINE.iter() })
        .copied()
        .collect();
    if files.len() != required.len() || required.iter().any(|name| !files.contains_key(*name)) {
        return Err(if minting {
            "minting requires a reference-only manifest; run capture_reference.py manifest --reference-only --write"
        } else {
            "the quality gate requires a complete manifest with both accepted baseline hashes"
        }.into());
    }
    if minting && BASELINE.iter().any(|name| fixture.join(name).exists()) {
        return Err("minting requires both accepted baseline files absent".into());
    }
    for name in required {
        let bytes = std::fs::read(fixture.join(name))
            .map_err(|e| format!("reference fixture {name}: {e}"))?;
        if Some(hrx::bundle::digest(&bytes).as_str()) != files[name].as_str() {
            return Err(format!("reference fixture changed: {name}"));
        }
    }
    Ok(())
}

// The fixture format is deliberately narrow: NumPy v1/v2, little-endian f32,
// C order, with exact expected dimensions. No Python runtime is involved.
fn array(path: &Path, shape: &[usize]) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(bytes.len() >= 12 && &bytes[..6] == b"\x93NUMPY", "{}", path.display());
    let (start, len) = match (bytes[6], bytes[7]) {
        (1, 0) => (10, u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize),
        (2, 0) => (12, u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize),
        version => panic!("unsupported NumPy version {version:?}"),
    };
    let header = std::str::from_utf8(&bytes[start..start + len]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "unsupported dtype: {header}");
    assert!(header.contains("'fortran_order': False"), "unsupported order: {header}");
    let dimensions = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let dimensions: Vec<usize> = dimensions
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(dimensions, shape, "{}", path.display());
    let data = &bytes[start + len..];
    assert_eq!(data.len(), shape.iter().product::<usize>() * 4);
    let result: Vec<f32> =
        data.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
    assert!(result.iter().all(|v| v.is_finite()), "{}", path.display());
    result
}

fn metrics(ours: &[f32], truth: &[f32]) -> (f64, f64) {
    assert_eq!(ours.len(), truth.len());
    let (mut dot, mut aa, mut bb, mut error) = (0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in ours.iter().zip(truth) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        aa += a * a;
        bb += b * b;
        error += (a - b) * (a - b);
    }
    (dot / (aa * bb).sqrt(), (error / bb).sqrt())
}

fn rgb(path: &Path, size: usize) -> Vec<u8> {
    let decoder = png::Decoder::new(std::fs::File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut bytes).unwrap();
    assert_eq!((info.width as usize, info.height as usize), (size, size));
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    bytes.truncate(info.buffer_size());
    bytes
}

fn image_psnr(ours: &[u8], truth: &[u8]) -> f64 {
    assert_eq!(ours.len(), truth.len());
    let mse = ours
        .iter()
        .zip(truth)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / truth.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Writes a NumPy v1 f32 array, the format `array` above reads. Minting the accepted
/// baseline has to produce a fixture file, and the fixture format is deliberately narrow.
fn write_array(path: &Path, values: &[f32], shape: &[usize]) {
    let dimensions: String =
        shape.iter().map(|n| format!("{n}, ")).collect::<Vec<_>>().concat();
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}), }}");
    // The header is padded so that the data starts on a 64-byte boundary.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn write_rgb(path: &Path, pixels: &[u8], size: usize) {
    let file =
        std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut encoder =
        png::Encoder::new(std::io::BufWriter::new(file), size as u32, size as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().unwrap().write_image_data(pixels).unwrap();
}

fn sidecar(path: &Path, extension: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(extension);
    PathBuf::from(name)
}

/// The receipt is published last so interrupted exports cannot be promoted as a pair.
fn export_candidate(
    path: &Path,
    state: &[f32],
    pixels: &[u8],
    size: usize,
    manifest: &serde_json::Value,
) {
    let receipt = sidecar(path, ".json");
    if receipt.exists() {
        std::fs::remove_file(&receipt).unwrap();
    }
    let bytes: Vec<_> = state.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, &bytes).unwrap();
    let image = sidecar(path, ".png");
    write_rgb(&image, pixels, size);
    let reference: serde_json::Map<_, _> = REFERENCE
        .iter()
        .map(|name| ((*name).into(), manifest["files"][name].clone()))
        .collect();
    let record = serde_json::json!({
        "version": 1,
        "size": size,
        "reference": reference,
        "latents_sha256": hrx::bundle::digest(&bytes),
        "image_sha256": hrx::bundle::digest(&std::fs::read(image).unwrap()),
    });
    std::fs::write(receipt, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
}

#[test]
fn fresh_baseline_requires_explicit_minting_and_verified_reference() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = directory.path();
    let mut files = serde_json::Map::new();
    for name in REFERENCE {
        std::fs::write(fixture.join(name), name).unwrap();
        files.insert(name.into(), hrx::bundle::digest(name.as_bytes()).into());
    }
    let manifest = serde_json::json!({ "files": files });
    assert!(validate_fixture(fixture, &manifest, true).is_ok());
    assert!(validate_fixture(fixture, &manifest, false).is_err());
    assert!(validate_fixture(fixture, &serde_json::json!({"files": {}}), true).is_err());
    for name in REFERENCE {
        std::fs::remove_file(fixture.join(name)).unwrap();
        assert!(validate_fixture(fixture, &manifest, true).is_err(), "missing {name}");
        std::fs::write(fixture.join(name), "changed").unwrap();
        assert!(validate_fixture(fixture, &manifest, true).is_err(), "changed {name}");
        std::fs::write(fixture.join(name), name).unwrap();
    }
    // Either leftover baseline file is enough to refuse minting, including an orphan PNG.
    for name in BASELINE {
        std::fs::write(fixture.join(name), name).unwrap();
        assert!(validate_fixture(fixture, &manifest, true).is_err());
        std::fs::remove_file(fixture.join(name)).unwrap();
    }
}

#[test]
fn ordinary_gate_requires_complete_unchanged_baseline() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = directory.path();
    let mut files = serde_json::Map::new();
    for name in REFERENCE.into_iter().chain(BASELINE) {
        std::fs::write(fixture.join(name), name).unwrap();
        files.insert(name.into(), hrx::bundle::digest(name.as_bytes()).into());
    }
    let manifest = serde_json::json!({ "files": files });
    assert!(validate_fixture(fixture, &manifest, false).is_ok());
    assert!(validate_fixture(fixture, &manifest, true).is_err());
    for name in BASELINE {
        std::fs::remove_file(fixture.join(name)).unwrap();
        assert!(validate_fixture(fixture, &manifest, false).is_err());
        std::fs::write(fixture.join(name), "changed").unwrap();
        assert!(validate_fixture(fixture, &manifest, false).is_err());
        std::fs::write(fixture.join(name), name).unwrap();
    }
}

#[test]
fn candidate_export_preserves_native_pixels_and_binds_reference() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("candidate.f32");
    let state: Vec<f32> = (0..64).map(|v| v as f32 / 16.0).collect();
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|v| (v % 256) as u8).collect();
    let mut files = serde_json::Map::new();
    for name in REFERENCE.into_iter().chain(BASELINE) {
        files.insert(name.into(), hrx::bundle::digest(name.as_bytes()).into());
    }
    let manifest = serde_json::json!({ "files": files });
    export_candidate(&path, &state, &pixels, 16, &manifest);
    assert_eq!(rgb(&sidecar(&path, ".png"), 16), pixels);
    let raw = std::fs::read(&path).unwrap();
    let restored: Vec<f32> =
        raw.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
    assert_eq!(restored, state);
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(sidecar(&path, ".json")).unwrap()).unwrap();
    assert_eq!(record["version"], 1);
    assert_eq!(record["size"], 16);
    assert_eq!(record["latents_sha256"], hrx::bundle::digest(&raw));
    assert_eq!(
        record["image_sha256"],
        hrx::bundle::digest(&std::fs::read(sidecar(&path, ".png")).unwrap())
    );
    assert_eq!(record["reference"].as_object().unwrap().len(), REFERENCE.len());
    for name in REFERENCE {
        assert_eq!(record["reference"][name], manifest["files"][name]);
    }
    let npy = directory.path().join("w8a8.npy");
    write_array(&npy, &state, &[1, 1, 64]);
    assert_eq!(array(&npy, &[1, 1, 64]), state);
}

#[test]
#[ignore = "requires gfx1151, local weights and the unquantized BF16 reference fixture"]
fn unquantized_bf16_reference_quality_does_not_regress() {
    use krea2_numerics::{from_f32, to_f32};
    use krea2_ops::{Ops, Pool, Tensor};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = std::env::var_os("KREA2_QUALITY_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("build/quality"));
    let manifest_text = std::env::var_os("KREA2_QUALITY_MANIFEST")
        .map(|path| std::fs::read_to_string(path).unwrap())
        .unwrap_or_else(|| include_str!("fixtures/unquantized.json").into());
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    let minting = match std::env::var("KREA2_QUALITY_MINT") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("KREA2_QUALITY_MINT must be unset or exactly 1"),
    };
    validate_fixture(&fixture, &manifest, minting).unwrap_or_else(|e| panic!("{e}"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("job.json")).unwrap()).unwrap();
    // The reference must be the unquantized model: the official diffusers repository, or a
    // bf16 checkpoint mapped onto it, but never the quantized weights the candidate runs.
    let source = meta["repo"]
        .as_str()
        .or_else(|| meta["checkpoint"].as_str())
        .expect("job.json must record the reference's source");
    assert!(
        !source.contains("int8") && !source.contains("convrot"),
        "reference captured from quantized weights: {source}"
    );
    assert!(
        matches!(meta["dtype"].as_str().unwrap_or("bfloat16"), "bfloat16" | "float32"),
        "reference captured at an unexpected precision"
    );
    let size = meta["size"].as_u64().unwrap() as usize;
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let text_tokens = meta["text_tokens"].as_u64().unwrap() as usize;
    // The schedule is the fixture's, not this test's: a shift the reference was not captured
    // with would silently compare two different trajectories. Fixtures written before
    // scripts/capture_reference.py recorded the field carry the value both arms used.
    let shift = meta["shift"].as_f64().unwrap_or(1.15);
    let tokens = size / 16 * (size / 16);
    let text = array(&fixture.join("text.npy"), &[text_tokens, 12, 2560]);
    let mut state = array(&fixture.join("noise.npy"), &[tokens, 64]);
    let truth = array(&fixture.join("bf16.npy"), &[1, tokens, 64]);
    let truth_rgb = rgb(&fixture.join("bf16.png"), size);
    let accepted = (!minting).then(|| array(&fixture.join("w8a8.npy"), &[1, tokens, 64]));
    let accepted_rgb = (!minting).then(|| rgb(&fixture.join("w8a8.png"), size));
    let checkpoint =
        std::env::var_os("KREA2_CHECKPOINT").map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join("comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors")
        });
    // Run the production GPU Euler kernel, including its BF16 rounding, rather
    // than substituting a host sampler in the parity check.
    let stream = hrx::Device::open().unwrap();
    let _scope = stream.enter();
    let pool = Pool::new();
    let ops = Ops::new(pool.clone());
    let initial: Vec<_> = state.iter().map(|&v| from_f32(v)).collect();
    let resident = Tensor::from_slice(&pool, &initial, tokens, 64).unwrap();
    let pipeline =
        Pipeline::open(Files::of(&checkpoint).offline(true).resolve().unwrap(), None).unwrap();
    for step in 0..steps {
        let sigma = krea2_pipeline::schedule::sigma(step, steps, shift);
        let next = krea2_pipeline::schedule::sigma(step + 1, steps, shift);
        let velocity =
            pipeline.transformer(&text, text_tokens, &state, size, size, sigma).unwrap();
        let velocity: Vec<_> = velocity.into_iter().map(from_f32).collect();
        let velocity = Tensor::from_slice(&pool, &velocity, tokens, 64).unwrap();
        ops.euler_step(&resident, &velocity, next - sigma).unwrap();
        state = resident.download().unwrap().into_iter().map(to_f32).collect();
        eprintln!("BF16 reference: step {}/{}", step + 1, steps);
    }
    let (cosine, rms) = metrics(&state, &truth);
    let ours_rgb = pipeline.decode(&state, size, size).unwrap();
    if let Some(path) = std::env::var_os("KREA2_QUALITY_OUTPUT") {
        export_candidate(Path::new(&path), &state, &ours_rgb, size, &manifest);
    }
    let psnr = image_psnr(&ours_rgb, &truth_rgb);
    eprintln!("unquantized reference: cosine {cosine:.6}, relative RMS {rms:.6}");
    eprintln!("unquantized reference image: PSNR {psnr:.6} dB");
    let (Some(accepted), Some(accepted_rgb)) = (accepted, accepted_rgb) else {
        write_array(&fixture.join("w8a8.npy"), &state, &[1, tokens, 64]);
        write_rgb(&fixture.join("w8a8.png"), &ours_rgb, size);
        eprintln!(
            "minted the accepted baseline in {}; re-pin it with \
             scripts/capture_reference.py manifest --write",
            fixture.display()
        );
        return;
    };
    let accepted_rms = metrics(&accepted, &truth).1;
    let loss_db = 20.0 * (rms / accepted_rms).log10();
    let accepted_psnr = image_psnr(&accepted_rgb, &truth_rgb);
    eprintln!("accepted: relative RMS {accepted_rms:.6}, PSNR {accepted_psnr:.6} dB, loss {loss_db:.6} dB");
    assert!(
        loss_db <= 0.1,
        "quality regressed against the unquantized reference by {loss_db} dB"
    );
    assert!(
        psnr >= accepted_psnr - 0.1,
        "image quality regressed against the unquantized reference"
    );
}
