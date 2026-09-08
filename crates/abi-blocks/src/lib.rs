//! `build/libkrea2.so`: the resident block session behind the C ABI that
//! `krea2_loom.py` and the pipeline call.
//!
//! Everything here is the boundary and nothing else — raw pointers into slices,
//! results into `(code, message)`, and a catch so a panic becomes an error
//! instead of unwinding into C.
use std::ffi::{c_char, c_int, CStr};
use std::path::Path;
use std::sync::Arc;

use krea2_session::{Session, Weights, HEAD_DIM, HIDDEN};

/// Must match `KREA2_ABI_VERSION`; `krea2_loom.py` refuses anything else.
const ABI_VERSION: u32 = 3;

const OK: c_int = 0;
const FAILED: c_int = 1;
const INVALID_ARGUMENT: c_int = 64;

/// The opaque handle C sees as `krea2_session *`.
pub struct SessionHandle {
    session: Session,
    profile: bool,
}

#[no_mangle]
pub extern "C" fn krea2_abi_version() -> u32 {
    ABI_VERSION
}

/// Writes a NUL-terminated message into the caller's buffer, truncating it.
///
/// # Safety
/// `error` must be null or point to `capacity` writable bytes.
unsafe fn report(error: *mut c_char, capacity: usize, message: &str) {
    if error.is_null() || capacity == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let room = std::cmp::min(bytes.len(), capacity - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, error, room);
    *error.add(room) = 0;
}

/// Runs `body`, turning its failure — or a panic — into a code and a message.
///
/// # Safety
/// As [`report`].
unsafe fn guard(
    error: *mut c_char,
    capacity: usize,
    body: impl FnOnce() -> krea2_session::Result<()> + std::panic::UnwindSafe,
) -> c_int {
    match std::panic::catch_unwind(body) {
        Ok(Ok(())) => OK,
        Ok(Err(failure)) => {
            report(error, capacity, &failure.message);
            if failure.invalid_argument {
                INVALID_ARGUMENT
            } else {
                FAILED
            }
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            report(error, capacity, &message);
            FAILED
        }
    }
}

/// # Safety
/// `path` must be a NUL-terminated string.
unsafe fn path_of<'a>(path: *const c_char) -> krea2_session::Result<&'a Path> {
    if path.is_null() {
        return Err(krea2_session::Error::invalid("a null path"));
    }
    CStr::from_ptr(path)
        .to_str()
        .map(Path::new)
        .map_err(|_| krea2_session::Error::invalid("a path that is not UTF-8"))
}

/// Opens a session on a checkpoint and a bundle compiled for `tokens`.
///
/// # Safety
/// The paths must be NUL-terminated; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn krea2_create(
    weights: *const c_char,
    kernels: *const c_char,
    tokens: c_int,
    layers: c_int,
    out: *mut *mut SessionHandle,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    if out.is_null() {
        return INVALID_ARGUMENT;
    }
    *out = std::ptr::null_mut();
    let mut created = None;
    let code = guard(
        error,
        error_capacity,
        std::panic::AssertUnwindSafe(|| {
            if tokens < 0 || layers < 0 {
                return Err(krea2_session::Error::invalid(
                    "tokens and layers must be positive",
                ));
            }
            let session = Session::open(
                path_of(weights)?,
                path_of(kernels)?,
                tokens as usize,
                layers as usize,
            )?;
            created = Some(session);
            Ok(())
        }),
    );
    if let Some(session) = created {
        *out = Box::into_raw(Box::new(SessionHandle { session, profile: false }));
    }
    code
}

/// # Safety
/// `session` must come from [`krea2_create`] and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn krea2_destroy(session: *mut SessionHandle) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

/// Enables the per-stage timings on stderr. Returns the previous setting.
///
/// # Safety
/// `session` must come from [`krea2_create`].
#[no_mangle]
pub unsafe extern "C" fn krea2_profile(session: *mut SessionHandle, enable: c_int) -> c_int {
    match session.as_mut() {
        None => 0,
        Some(session) => {
            let was = session.profile;
            session.profile = enable != 0;
            c_int::from(was)
        }
    }
}

/// # Safety
/// The buffers must have the element counts they declare, and `x` must be
/// writable: the residual stream is read back into it.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn krea2_run(
    session: *mut SessionHandle,
    x: *mut u16,
    x_elements: usize,
    mods: *const f32,
    mods_elements: usize,
    cos: *const f32,
    sin: *const f32,
    rope_elements: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    krea2_run_range(
        session,
        0,
        -1,
        x,
        x_elements,
        mods,
        mods_elements,
        cos,
        sin,
        rope_elements,
        error,
        error_capacity,
    )
}

/// A contiguous subset of the loaded blocks, over the same full-session
/// modulation layout. `block_count` of -1 means all the remaining blocks.
///
/// # Safety
/// As [`krea2_run`].
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn krea2_run_range(
    session: *mut SessionHandle,
    first_block: c_int,
    block_count: c_int,
    x: *mut u16,
    x_elements: usize,
    mods: *const f32,
    mods_elements: usize,
    cos: *const f32,
    sin: *const f32,
    rope_elements: usize,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    guard(
        error,
        error_capacity,
        std::panic::AssertUnwindSafe(|| {
            let Some(session) = session.as_mut() else {
                return Err(krea2_session::Error::invalid("a null session"));
            };
            if x.is_null() || mods.is_null() || cos.is_null() || sin.is_null() {
                return Err(krea2_session::Error::invalid("a null buffer"));
            }
            if first_block < 0 || block_count == 0 || block_count < -1 {
                return Err(krea2_session::Error::invalid(
                    "block range must be within the loaded layers",
                ));
            }
            // The declared counts are checked against the session's shape inside
            // run(); the slices are built from them, so a wrong count is a caller
            // error either way.
            let stream = std::slice::from_raw_parts_mut(x, x_elements);
            let modulation = std::slice::from_raw_parts(mods, mods_elements);
            let cos = std::slice::from_raw_parts(cos, rope_elements);
            let sin = std::slice::from_raw_parts(sin, rope_elements);
            let count = match block_count {
                -1 => None,
                n => Some(n as usize),
            };
            session.session.run(stream, modulation, cos, sin, first_block as usize, count)
        }),
    )
}

/// A checkpoint on the device, shared by sessions of different lengths. The
/// pipeline library holds one of these across resolution changes; C sees it as
/// `krea2_weights *`.
pub struct WeightsHandle(Arc<Weights>);

/// # Safety
/// `path` must be NUL-terminated and `out` writable.
#[no_mangle]
pub unsafe extern "C" fn krea2_weights_load(
    path: *const c_char,
    out: *mut *mut WeightsHandle,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    if out.is_null() {
        return INVALID_ARGUMENT;
    }
    *out = std::ptr::null_mut();
    let mut loaded = None;
    let code = guard(
        error,
        error_capacity,
        std::panic::AssertUnwindSafe(|| {
            loaded = Some(Arc::new(Weights::load(path_of(path)?)?));
            Ok(())
        }),
    );
    if let Some(weights) = loaded {
        *out = Box::into_raw(Box::new(WeightsHandle(weights)));
    }
    code
}

/// # Safety
/// `weights` must come from [`krea2_weights_load`].
#[no_mangle]
pub unsafe extern "C" fn krea2_weights_release(weights: *mut WeightsHandle) {
    if !weights.is_null() {
        drop(Box::from_raw(weights));
    }
}

/// The operand width the checkpoint's rows carry, 4 or 8.
///
/// # Safety
/// `weights` must come from [`krea2_weights_load`].
#[no_mangle]
pub unsafe extern "C" fn krea2_weights_bits(weights: *const WeightsHandle) -> c_int {
    match weights.as_ref() {
        Some(weights) => weights.0.bits() as c_int,
        None => 0,
    }
}

/// A session over already-loaded weights.
///
/// # Safety
/// As [`krea2_create`], with `weights` from [`krea2_weights_load`].
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn krea2_create_shared(
    weights: *const WeightsHandle,
    kernels: *const c_char,
    tokens: c_int,
    layers: c_int,
    out: *mut *mut SessionHandle,
    error: *mut c_char,
    error_capacity: usize,
) -> c_int {
    if out.is_null() {
        return INVALID_ARGUMENT;
    }
    *out = std::ptr::null_mut();
    let mut created = None;
    let code = guard(
        error,
        error_capacity,
        std::panic::AssertUnwindSafe(|| {
            let Some(weights) = weights.as_ref() else {
                return Err(krea2_session::Error::invalid("a null weights handle"));
            };
            created = Some(Session::with_weights(
                weights.0.clone(),
                path_of(kernels)?,
                tokens as usize,
                layers as usize,
            )?);
            Ok(())
        }),
    );
    if let Some(session) = created {
        *out = Box::into_raw(Box::new(SessionHandle { session, profile: false }));
    }
    code
}

/// The residual stream's width and the rotary tables', for callers sizing
/// buffers without hard-coding the model.
#[no_mangle]
pub extern "C" fn krea2_hidden_size() -> c_int {
    HIDDEN
}

#[no_mangle]
pub extern "C" fn krea2_head_dim() -> c_int {
    HEAD_DIM
}
