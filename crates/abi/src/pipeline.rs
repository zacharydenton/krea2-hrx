//! `krea2_pipeline_*`: standalone Krea 2 inference behind its C ABI.
//!
//! The same boundary discipline as the block ABI next door — raw pointers into
//! slices, results into `(code, message)`, a catch so a panic becomes an error
//! — over `krea2-pipeline` rather than `krea2-session`.
use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::path::Path;

use krea2_pipeline::{Files, Pipeline, Request, Result};

use crate::{report, OK};

/// Must match `KREA2_PIPELINE_ABI_VERSION`.
const ABI_VERSION: u32 = 3;

const FAILED: c_int = 1;

/// Called once per sampling step, after the step, with the seconds spent so
/// far. Returning nonzero abandons the image.
pub type ProgressFn = Option<extern "C" fn(*mut c_void, c_int, c_int, f64) -> c_int>;

/// The opaque handle C sees as `krea2_pipeline *`.
pub struct PipelineHandle {
    pipeline: Pipeline,
    progress: std::sync::Mutex<Option<(ProgressFn, usize)>>,
}

#[no_mangle]
pub extern "C" fn krea2_pipeline_abi_version() -> u32 {
    ABI_VERSION
}

/// Runs `body`, turning its failure — or a panic — into a code and a message.
///
/// # Safety
/// `error` must be null or point to `capacity` writable bytes.
unsafe fn guard(
    error: *mut c_char,
    capacity: usize,
    body: impl FnOnce() -> Result<()> + std::panic::UnwindSafe,
) -> c_int {
    if !error.is_null() && capacity > 0 {
        *error = 0;
    }
    match std::panic::catch_unwind(body) {
        Ok(Ok(())) => OK,
        Ok(Err(failure)) => {
            report(error, capacity, &failure.0);
            FAILED
        }
        Err(_) => {
            report(error, capacity, "native inference failed");
            FAILED
        }
    }
}

/// # Safety
/// `text` must be null or a NUL-terminated string.
unsafe fn text_of<'a>(text: *const c_char) -> Option<&'a str> {
    if text.is_null() {
        return None;
    }
    CStr::from_ptr(text).to_str().ok()
}

/// # Safety
/// `handle` must be a live pipeline from `krea2_pipeline_create*`.
unsafe fn pipeline_of<'a>(handle: *const PipelineHandle) -> Result<&'a PipelineHandle> {
    handle.as_ref().ok_or_else(|| krea2_pipeline::Error("null pipeline".into()))
}

/// # Safety
/// `pointer` must be null or point to `len` readable elements.
unsafe fn slice_of<'a, T>(pointer: *const T, len: usize) -> Option<&'a [T]> {
    match pointer.is_null() {
        true => None,
        false => Some(std::slice::from_raw_parts(pointer, len)),
    }
}

/// ComfyUI's files named explicitly. `text_encoder` or `vae` null: found beside
/// the model, or in the Hugging Face cache. `distilled`: 1 for Turbo, 0 for
/// Raw, -1 to read the file name.
///
/// # Safety
/// Every pointer must be null or a NUL-terminated string; `out` must be
/// writable.
#[no_mangle]
pub unsafe extern "C" fn krea2_pipeline_create_files(
    model: *const c_char,
    text_encoder: *const c_char,
    vae: *const c_char,
    distilled: c_int,
    compiler: *const c_char,
    out: *mut *mut PipelineHandle,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    if !error.is_null() && capacity > 0 {
        *error = 0;
    }
    if out.is_null() {
        report(error, capacity, "model and output pointer are required");
        return FAILED;
    }
    *out = std::ptr::null_mut();
    let (model, text_encoder, vae, compiler) =
        (text_of(model), text_of(text_encoder), text_of(vae), text_of(compiler));
    let Some(model) = model else {
        report(error, capacity, "model and output pointer are required");
        return FAILED;
    };
    let built = std::panic::catch_unwind(|| {
        let files = Files::of(Path::new(model))
            .text_encoder(text_encoder.filter(|p| !p.is_empty()).map(Path::new))
            .vae(vae.filter(|p| !p.is_empty()).map(Path::new))
            .distilled(match distilled {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
            .resolve()?;
        Pipeline::open(files, compiler.filter(|c| !c.is_empty()))
    });
    match built {
        Ok(Ok(pipeline)) => {
            *out = Box::into_raw(Box::new(PipelineHandle {
                pipeline,
                progress: std::sync::Mutex::new(None),
            }));
            OK
        }
        Ok(Err(failure)) => {
            report(error, capacity, &failure.0);
            FAILED
        }
        Err(_) => {
            report(error, capacity, "native initialization failed");
            FAILED
        }
    }
}

/// # Safety
/// As [`krea2_pipeline_create_files`].
#[no_mangle]
pub unsafe extern "C" fn krea2_pipeline_create(
    model: *const c_char,
    compiler: *const c_char,
    out: *mut *mut PipelineHandle,
    error: *mut c_char,
    capacity: usize,
) -> c_int {
    krea2_pipeline_create_files(
        model,
        std::ptr::null(),
        std::ptr::null(),
        -1,
        compiler,
        out,
        error,
        capacity,
    )
}

/// # Safety
/// `pipeline` must come from a create call and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn krea2_pipeline_destroy(pipeline: *mut PipelineHandle) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Null clears it. The callback runs on the calling thread, inside the
/// pipeline's lock, so it must not call back into the pipeline.
///
/// # Safety
/// `pipeline` must be live for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn krea2_pipeline_set_progress(
    pipeline: *mut PipelineHandle,
    progress: ProgressFn,
    user: *mut c_void,
) {
    let Some(handle) = pipeline.as_ref() else {
        return;
    };
    let mut slot = handle.progress.lock().unwrap_or_else(|e| e.into_inner());
    *slot = progress.map(|function| (Some(function), user as usize));
}

/// # Safety
/// `pipeline` must be live.
#[no_mangle]
pub unsafe extern "C" fn krea2_pipeline_distilled(pipeline: *const PipelineHandle) -> c_int {
    c_int::from(pipeline.as_ref().is_some_and(|handle| handle.pipeline.distilled()))
}

/// Tokenizes arbitrary UTF-8 with no chat template.
///
/// # Safety
/// `ids` must be null or point to `capacity` writable elements.
#[no_mangle]
pub unsafe extern "C" fn krea2_tokenize(
    pipeline: *mut PipelineHandle,
    text: *const c_char,
    ids: *mut i32,
    capacity: usize,
    written: *mut usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(error, error_capacity, || {
        let handle = pipeline_of(pipeline)?;
        let (Some(text), false) = (text_of(text), written.is_null()) else {
            return Err(krea2_pipeline::Error("text and count are required".into()));
        };
        let encoded = handle.pipeline.tokenize(text)?;
        *written = encoded.len();
        if ids.is_null() {
            return Ok(());
        }
        if capacity < encoded.len() {
            return Err(krea2_pipeline::Error("token buffer too small".into()));
        }
        std::ptr::copy_nonoverlapping(encoded.as_ptr(), ids, encoded.len());
        Ok(())
    })
}

/// Encodes a prompt with Krea's template: float32 `[tokens][12][2560]`. With
/// `output` null, reports the token count without running the encoder.
///
/// # Safety
/// `output` must be null or point to `elements` writable floats.
#[no_mangle]
pub unsafe extern "C" fn krea2_encode(
    pipeline: *mut PipelineHandle,
    prompt: *const c_char,
    output: *mut c_float,
    elements: usize,
    tokens: *mut usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(error, error_capacity, || {
        let handle = pipeline_of(pipeline)?;
        let (Some(prompt), false) = (text_of(prompt), tokens.is_null()) else {
            return Err(krea2_pipeline::Error("prompt and count are required".into()));
        };
        if output.is_null() {
            *tokens = handle.pipeline.prompt_tokens(prompt)?;
            return Ok(());
        }
        let (count, values) = handle.pipeline.encode(prompt)?;
        *tokens = count;
        if elements < values.len() {
            return Err(krea2_pipeline::Error("text output buffer too small".into()));
        }
        std::ptr::copy_nonoverlapping(values.as_ptr(), output, values.len());
        Ok(())
    })
}

/// One transformer forward: packed latents and tapped text states in, float32
/// packed velocity out.
///
/// # Safety
/// Each buffer must point to the number of elements its length argument names.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn krea2_transformer(
    pipeline: *mut PipelineHandle,
    text: *const c_float,
    text_elements: usize,
    text_tokens: c_int,
    latents: *const c_float,
    latent_elements: usize,
    width: c_int,
    height: c_int,
    timestep: c_float,
    velocity: *mut c_float,
    velocity_elements: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(error, error_capacity, || {
        let handle = pipeline_of(pipeline)?;
        let invalid = || krea2_pipeline::Error("invalid transformer arguments".into());
        let (Some(text), Some(latents), false) = (
            slice_of(text, text_elements),
            slice_of(latents, latent_elements),
            velocity.is_null() || text_tokens < 1 || width < 1 || height < 1,
        ) else {
            return Err(invalid());
        };
        let (width, height) = (width as usize, height as usize);
        if velocity_elements != width / 16 * (height / 16) * 64 {
            return Err(invalid());
        }
        let out = handle.pipeline.transformer(
            text,
            text_tokens as usize,
            latents,
            width,
            height,
            timestep,
        )?;
        std::ptr::copy_nonoverlapping(out.as_ptr(), velocity, out.len());
        Ok(())
    })
}

/// Latents to contiguous RGB8 HWC.
///
/// # Safety
/// `rgb` must point to `rgb_bytes` writable bytes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn krea2_decode(
    pipeline: *mut PipelineHandle,
    latents: *const c_float,
    elements: usize,
    width: c_int,
    height: c_int,
    rgb: *mut u8,
    rgb_bytes: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(error, error_capacity, || {
        let handle = pipeline_of(pipeline)?;
        let (Some(latents), false) = (slice_of(latents, elements), rgb.is_null()) else {
            return Err(krea2_pipeline::Error("RGB output buffer too small".into()));
        };
        if width < 1 || height < 1 || rgb_bytes < (width as usize) * (height as usize) * 3 {
            return Err(krea2_pipeline::Error("RGB output buffer too small".into()));
        }
        let pixels = handle.pipeline.decode(latents, width as usize, height as usize)?;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), rgb, pixels.len());
        Ok(())
    })
}

/// One image, unguided: the Turbo form.
///
/// # Safety
/// As [`krea2_generate_guided`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn krea2_generate(
    pipeline: *mut PipelineHandle,
    prompt: *const c_char,
    width: c_int,
    height: c_int,
    steps: c_int,
    seed: u64,
    initial_latents: *const c_float,
    latent_elements: usize,
    rgb: *mut u8,
    rgb_bytes: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    if !(1..=100).contains(&steps) {
        report(error, error_capacity, "invalid generation arguments");
        return FAILED;
    }
    krea2_generate_guided(
        pipeline,
        prompt,
        std::ptr::null(),
        0.0,
        width,
        height,
        steps,
        seed,
        initial_latents,
        latent_elements,
        rgb,
        rgb_bytes,
        error,
        error_capacity,
    )
}

/// One image with classifier-free guidance in Krea's convention. `guidance`
/// below zero or `steps` of zero or less take the checkpoint's defaults.
///
/// # Safety
/// `rgb` must point to `rgb_bytes` writable bytes; `initial_latents` must be
/// null or point to `latent_elements` readable floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn krea2_generate_guided(
    pipeline: *mut PipelineHandle,
    prompt: *const c_char,
    negative_prompt: *const c_char,
    guidance: c_float,
    width: c_int,
    height: c_int,
    steps: c_int,
    seed: u64,
    initial_latents: *const c_float,
    latent_elements: usize,
    rgb: *mut u8,
    rgb_bytes: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(error, error_capacity, || {
        let handle = pipeline_of(pipeline)?;
        let invalid = || krea2_pipeline::Error("invalid generation arguments".into());
        let (Some(prompt), false) = (text_of(prompt), rgb.is_null() || width < 1 || height < 1)
        else {
            return Err(invalid());
        };
        let (width, height) = (width as usize, height as usize);
        if rgb_bytes < width * height * 3 {
            return Err(invalid());
        }
        if initial_latents.is_null() && latent_elements != 0 {
            return Err(krea2_pipeline::Error("latent count without input buffer".into()));
        }
        let request = Request {
            prompt,
            negative_prompt: text_of(negative_prompt).unwrap_or(""),
            width,
            height,
            steps: (steps > 0).then_some(steps as usize),
            guidance: (guidance >= 0.0).then_some(guidance),
            seed,
            initial_latents: slice_of(initial_latents, latent_elements),
        };
        // The callback is read once per image, so a set_progress racing a
        // generate takes effect on the next one rather than mid-image.
        let callback = *handle.progress.lock().unwrap_or_else(|e| e.into_inner());
        let mut relay = |step: usize, steps: usize, seconds: f64| match callback {
            Some((Some(function), user)) => {
                function(user as *mut c_void, step as c_int, steps as c_int, seconds) == 0
            }
            _ => true,
        };
        let pixels = handle.pipeline.generate(
            &request,
            callback.is_some().then_some(&mut relay as krea2_pipeline::Progress),
        )?;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), rgb, pixels.len());
        Ok(())
    })
}
