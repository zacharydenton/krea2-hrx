//! Invalid constructors must fail before initializing the GPU.
//! Tests inspect process mappings to check that the HSA provider was not loaded.
use std::path::{Path, PathBuf};

use krea2_session::{Error, Session};

/// The session a rejection never returns.
fn refused(result: krea2_session::Result<Session>, what: &str) -> Error {
    match result {
        Ok(_) => panic!("{what} was accepted"),
        Err(error) => error,
    }
}

/// A directory with a `launch.txt` and a checkpoint, per test.
fn fixture(name: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("krea2-session-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("the fixture directory");
    root
}

/// A safetensors file from a header and that many zero bytes of payload.
fn checkpoint(path: &Path, header: &str, payload: usize) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(std::iter::repeat_n(0u8, payload));
    std::fs::write(path, bytes).expect("the checkpoint");
}

/// The HSA provider, and therefore the GPU, was never opened.
fn no_device_was_opened() -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    !maps.contains("libhsa-runtime") && !maps.contains("hrx_provider")
}

#[test]
fn an_incomplete_checkpoint_and_bad_metadata_are_refused_before_the_gpu() {
    let root = fixture("reject");
    let valid = "4 16 256 4 64 8 6144 16448 4 8\n";
    std::fs::write(root.join("launch.txt"), valid).expect("launch.txt");

    // One block's wq only: every other tensor is missing.
    let path = root.join("incomplete.safetensors");
    checkpoint(
        &path,
        r#"{"blocks.0.attn.wq.weight":{"dtype":"I8","shape":[16,6144],"data_offsets":[0,98304]}}"#,
        98304,
    );
    for _ in 0..3 {
        let error = refused(Session::open(&path, &root, 16, 1), "an incomplete checkpoint");
        assert!(error.message.contains("missing tensor"), "{error}");
    }

    // A raster group of 0, an old version, the wrong wave count, a dense
    // down-projection pitch, a 128-row tile for int8 (that family has only the
    // 256-row one), a bad attention width, a missing field, and the two query
    // tile fields that no longer describe any bundle this builds.
    for metadata in [
        "5 16 256 4 64 8 6144 16448 16 8\n",
        "5 16 256 4 64 8 6144 16448 16 8 2\n",
        "5 4115 256 4 4160 8 6144 16448 16 8 2\n",
        "3 16 256 4 64 8 6144 16448 4 8\n",
        "4 16 256 0 64 8 6144 16448 4 8\n",
        "2 16 1 64 8\n",
        "4 16 256 4 64 4 6144 16448 4 8\n",
        "4 16 256 4 64 8 6144 16384 4 8\n",
        "4 16 128 1 64 8 6144 16448 4 8\n",
        "4 16 256 4 64 8 6144 16448 6 8\n",
        "4 16 256 4 64 8 6144 16448 4\n",
    ] {
        std::fs::write(root.join("launch.txt"), metadata).expect("launch.txt");
        let tokens = if metadata.starts_with("5 4115 ") { 4115 } else { 16 };
        let error = refused(Session::open(&path, &root, tokens, 1), metadata);
        assert!(error.invalid_argument, "{metadata:?} was not an invalid argument: {error}");
    }

    // A path that is not a checkpoint at all.
    std::fs::write(root.join("launch.txt"), valid).expect("launch.txt");
    let error = refused(Session::open(&root, &root, 16, 1), "a directory");
    assert!(error.message.contains(".safetensors"), "{error}");

    assert!(no_device_was_opened(), "a rejected constructor opened the GPU");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn the_interleaved_gate_and_up_rows_are_validated_before_any_allocation() {
    let root = fixture("interleave");
    std::fs::write(root.join("launch.txt"), "4 16 256 4 64 8 6144 16448 4 8\n")
        .expect("launch.txt");
    let path = root.join("tiny.safetensors");

    // mlp.up carries one scale too few, so the 16-row interleave cannot be
    // laid out; and separately, a weight with no rows at all.
    for zero_rows in [false, true] {
        let mut entries = Vec::new();
        let mut offset = 0usize;
        for name in [
            "attn.wq",
            "attn.wk",
            "attn.wv",
            "attn.gate",
            "attn.wo",
            "mlp.gate",
            "mlp.up",
            "mlp.down",
        ] {
            let rows = if zero_rows { 0 } else { 16 };
            entries.push(format!(
                r#""blocks.0.{name}.weight":{{"dtype":"I8","shape":[{rows},1],"data_offsets":[{offset},{}]}}"#,
                offset + rows
            ));
            offset += rows;
            let scales = if name == "mlp.up" { 15 } else { 16 };
            entries.push(format!(
                r#""blocks.0.{name}.weight_scale":{{"dtype":"F32","shape":[{scales}],"data_offsets":[{offset},{}]}}"#,
                offset + scales * 4
            ));
            offset += scales * 4;
        }
        checkpoint(&path, &format!("{{{}}}", entries.join(",")), offset);
        let error = refused(Session::open(&path, &root, 16, 1), "a malformed operand");
        let wanted =
            if zero_rows { "invalid weight shape" } else { "scale count in blocks.0.mlp.up" };
        assert!(error.message.contains(wanted), "{zero_rows}: {error}");
    }

    assert!(no_device_was_opened(), "a rejected constructor opened the GPU");
    std::fs::remove_dir_all(root).ok();
}
