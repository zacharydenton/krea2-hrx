//! A safe wrapper over the runtime's C ABI (`build/include/krea2_pipeline.h`),
//! and the
//! worked example of using it from another language: open the checkpoint, set a
//! progress callback, generate RGB.
//!
//! The shared library owns the GPU session; every call is serialised inside it,
//! so `Pipeline` is `Send` but deliberately not `Sync`. Errors come back as a
//! nonzero return with a message written into a caller-provided buffer.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

/// Must match `krea2_pipeline_abi_version()` in the library it links.
pub const ABI_VERSION: u32 = 3;

#[repr(C)]
struct Handle {
    _private: [u8; 0],
}

type ProgressFn =
    extern "C" fn(user: *mut c_void, step: c_int, steps: c_int, seconds: f64) -> c_int;

#[link(name = "krea2_pipeline")]
extern "C" {
    fn krea2_pipeline_abi_version() -> u32;
    fn krea2_pipeline_create_files(
        model: *const c_char,
        text_encoder: *const c_char,
        vae: *const c_char,
        distilled: c_int,
        compiler: *const c_char,
        out: *mut *mut Handle,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn krea2_pipeline_destroy(pipeline: *mut Handle);
    fn krea2_pipeline_distilled(pipeline: *const Handle) -> c_int;
    fn krea2_pipeline_set_progress(
        pipeline: *mut Handle,
        progress: Option<ProgressFn>,
        user: *mut c_void,
    );
    fn krea2_generate_guided(
        pipeline: *mut Handle,
        prompt: *const c_char,
        negative_prompt: *const c_char,
        guidance: f32,
        width: c_int,
        height: c_int,
        steps: c_int,
        seed: u64,
        initial_latents: *const f32,
        latent_elements: usize,
        rgb: *mut u8,
        rgb_bytes: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

/// What the library said went wrong, verbatim.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// One image's worth of parameters. `steps` and `guidance` of `None` take the
/// checkpoint's own defaults (Turbo: 8 steps unguided; Raw: 52 at 3.5).
pub struct Request<'a> {
    pub prompt: &'a str,
    pub negative: Option<&'a str>,
    pub guidance: Option<f32>,
    pub width: i32,
    pub height: i32,
    pub steps: Option<i32>,
    pub seed: u64,
}

/// A boxed closure called once per sampling step; returning `false` cancels.
type Callback = Box<dyn FnMut(i32, i32, f64) -> bool>;

pub struct Pipeline {
    handle: *mut Handle,
    // Boxed so its address is stable while the library holds it as `user`.
    progress: Option<Box<Callback>>,
}

// The library serialises calls on a session; the handle must not be shared
// between threads without that discipline, so only Send.
unsafe impl Send for Pipeline {}

fn message(buffer: &[c_char]) -> String {
    // Safety: the library always writes a NUL-terminated string into the buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy().into_owned()
}

fn c_string(text: &str) -> Result<CString, Error> {
    CString::new(text).map_err(|_| Error("text contains a NUL byte".into()))
}

fn c_path(path: &Path) -> Result<CString, Error> {
    c_string(path.to_str().ok_or_else(|| Error(format!("{} is not UTF-8", path.display())))?)
}

impl Pipeline {
    /// Opens ComfyUI's files. `text_encoder` and `vae` of `None` are found
    /// beside the model in ComfyUI's layout; `distilled` of `None` infers Turbo
    /// or Raw from the file name; `compiler` of `None` uses `LOOM_COMPILE`/PATH.
    pub fn open(
        model: &Path,
        text_encoder: Option<&Path>,
        vae: Option<&Path>,
        distilled: Option<bool>,
        compiler: Option<&Path>,
    ) -> Result<Self, Error> {
        // Safety: no arguments; the library is linked at build time.
        let library = unsafe { krea2_pipeline_abi_version() };
        if library != ABI_VERSION {
            return Err(Error(format!(
                "libkrea2_pipeline.so is ABI {library}, this binary expects {ABI_VERSION}"
            )));
        }
        let model = c_path(model)?;
        let text_encoder = text_encoder.map(c_path).transpose()?;
        let vae = vae.map(c_path).transpose()?;
        let compiler = compiler.map(c_path).transpose()?;
        let optional =
            |value: &Option<CString>| value.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut handle: *mut Handle = std::ptr::null_mut();
        let mut error = [0 as c_char; 4096];
        // Safety: every pointer is valid for the call, and the error buffer has
        // the capacity we declare.
        let failed = unsafe {
            krea2_pipeline_create_files(
                model.as_ptr(),
                optional(&text_encoder),
                optional(&vae),
                distilled.map_or(-1, |d| d as c_int),
                optional(&compiler),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if failed != 0 || handle.is_null() {
            return Err(Error(message(&error)));
        }
        Ok(Self { handle, progress: None })
    }

    /// 1 for the distilled Turbo checkpoint, 0 for Raw.
    pub fn distilled(&self) -> bool {
        // Safety: the handle is non-null for the life of the value.
        unsafe { krea2_pipeline_distilled(self.handle) != 0 }
    }

    /// Reports after every sampling step: `(step, steps, seconds so far)`.
    /// Returning `false` from the closure abandons the image.
    pub fn on_progress(&mut self, callback: impl FnMut(i32, i32, f64) -> bool + 'static) {
        extern "C" fn trampoline(
            user: *mut c_void,
            step: c_int,
            steps: c_int,
            seconds: f64,
        ) -> c_int {
            // Safety: `user` is the Box the pipeline keeps alive below, and the
            // library calls this on the thread inside generate().
            let callback = unsafe { &mut *(user as *mut Callback) };
            if callback(step, steps, seconds) {
                0
            } else {
                1
            }
        }
        let mut boxed: Box<Callback> = Box::new(Box::new(callback));
        let user = &mut *boxed as *mut Callback as *mut c_void;
        // Safety: the Box outlives the registration -- it is dropped in Drop,
        // after the callback has been cleared.
        unsafe { krea2_pipeline_set_progress(self.handle, Some(trampoline), user) };
        self.progress = Some(boxed);
    }

    /// Generates one image as contiguous RGB8, `height * width * 3` bytes.
    pub fn generate(&mut self, request: &Request<'_>) -> Result<Vec<u8>, Error> {
        let prompt = c_string(request.prompt)?;
        let negative = request.negative.map(c_string).transpose()?;
        let mut rgb = vec![0u8; request.width as usize * request.height as usize * 3];
        let mut error = [0 as c_char; 4096];
        // Safety: rgb has the length we pass, and the strings outlive the call.
        let failed = unsafe {
            krea2_generate_guided(
                self.handle,
                prompt.as_ptr(),
                negative.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                request.guidance.unwrap_or(-1.0),
                request.width,
                request.height,
                request.steps.unwrap_or(-1),
                request.seed,
                std::ptr::null(),
                0,
                rgb.as_mut_ptr(),
                rgb.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if failed != 0 {
            return Err(Error(message(&error)));
        }
        Ok(rgb)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // Clear the callback before the Box goes away, then release the session.
        // Safety: the handle is still live and nothing else refers to it.
        unsafe {
            krea2_pipeline_set_progress(self.handle, None, std::ptr::null_mut());
            krea2_pipeline_destroy(self.handle);
        }
    }
}
