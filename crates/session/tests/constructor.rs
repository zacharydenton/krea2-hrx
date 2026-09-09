//! Invalid constructors must fail before initializing the GPU.
//! Tests inspect process mappings to check that the HSA provider was not loaded.
//!
//! `Session::open` now takes the caller's stream, so it cannot itself be the
//! subject: holding a stream means the runtime is already mapped. What it does
//! first is [`Session::validate`], which takes no stream — so these run that,
//! and the mapping assertions still hold over the whole validation path.
use std::path::{Path, PathBuf};

use krea2_session::{Error, Session};

/// The checkpoint a rejection never returns.
fn refused<T>(result: krea2_session::Result<T>, what: &str) -> Error {
    match result {
        Ok(_) => panic!("{what} was accepted"),
        Err(error) => error,
    }
}

/// A directory for a checkpoint, per test.
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

/// Neither the native runtime nor the HSA provider, and therefore not the GPU,
/// was ever loaded. `libhrx.so` is checked as well as the provider because the
/// runtime is loaded first: a constructor that reached it has already done more
/// than reject its arguments.
fn no_device_was_opened() -> bool {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    !maps.contains("libhsa-runtime")
        && !maps.contains("hrx_provider")
        && !maps.contains("libhrx.so")
}

#[test]
fn an_incomplete_checkpoint_and_invalid_dimensions_are_refused_before_the_gpu() {
    let root = fixture("reject");

    // One block's wq only: every other tensor is missing.
    let path = root.join("incomplete.safetensors");
    checkpoint(
        &path,
        r#"{"blocks.0.attn.wq.weight":{"dtype":"I8","shape":[16,6144],"data_offsets":[0,98304]}}"#,
        98304,
    );
    for _ in 0..3 {
        let error = refused(Session::validate(&path, 16, 1), "an incomplete checkpoint");
        assert!(error.message.contains("missing tensor"), "{error}");
    }

    for (tokens, layers) in [(0, 1), (15, 1), (16897, 1), (16, 0), (16, 29), (usize::MAX, 1)] {
        let error = refused(Session::validate(&path, tokens, layers), "invalid dimensions");
        assert!(error.invalid_argument, "{tokens}/{layers}: {error}");
    }

    // A path that is not a checkpoint at all.
    let error = refused(Session::validate(&root, 16, 1), "a directory");
    assert!(error.message.contains(".safetensors"), "{error}");

    assert!(no_device_was_opened(), "a rejected constructor loaded the native runtime");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn the_interleaved_gate_and_up_rows_are_validated_before_any_allocation() {
    let root = fixture("interleave");
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
        let error = refused(Session::validate(&path, 16, 1), "a malformed operand");
        let wanted =
            if zero_rows { "invalid weight shape" } else { "scale count in blocks.0.mlp.up" };
        assert!(error.message.contains(wanted), "{zero_rows}: {error}");
    }

    assert!(no_device_was_opened(), "a rejected constructor loaded the native runtime");
    std::fs::remove_dir_all(root).ok();
}
