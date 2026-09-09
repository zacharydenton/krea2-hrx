//! C interfaces for resident block sessions and the standalone pipeline.
//!
//! Both APIs share `libkrea2.so`. This module validates buffers, translates errors
//! into status codes/messages, and catches panics before they cross the C ABI.
use std::ffi::{c_char, c_int, CStr};
use std::path::Path;
use std::sync::Arc;

use krea2_session::{Session, Weights, HEAD_DIM, HIDDEN};

pub mod pipeline;

/// The block ABI's version. Foreign callers check this before using handles. These carry
/// the names the header has always had, because C callers use them.
pub const KREA2_ABI_VERSION: u32 = 3;

pub const KREA2_OK: c_int = 0;
pub const KREA2_ERROR: c_int = 1;
pub const KREA2_INVALID_ARGUMENT: c_int = 64;

/// The opaque handle C sees as `krea2_session *`.
///
/// Every entry point takes it by shared reference: calls may overlap, the
/// session serializes them internally, and handing out `&mut` from two threads
/// at once would be undefined behaviour whatever the lock did afterwards.
pub struct SessionHandle {
    session: Session,
}

#[no_mangle]
pub extern "C" fn krea2_abi_version() -> u32 {
    KREA2_ABI_VERSION
}

/// Writes a NUL-terminated message into the caller's buffer, truncating it.
///
/// # Safety
/// `error` must be null or point to `capacity` writable bytes.
pub(crate) unsafe fn report(error: *mut c_char, capacity: usize, message: &str) {
    hrx::ffi::report(error, capacity, message);
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
    hrx::ffi::boundary(error, capacity, KREA2_ERROR, || {
        body().map_err(|failure| {
            hrx::ffi::Failure::new(
                if failure.invalid_argument { KREA2_INVALID_ARGUMENT } else { KREA2_ERROR },
                failure.message,
            )
        })
    })
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
        return KREA2_INVALID_ARGUMENT;
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
        *out = Box::into_raw(Box::new(SessionHandle { session }));
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
    match session.as_ref() {
        None => 0,
        Some(handle) => c_int::from(handle.session.set_profile(enable != 0)),
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
            let Some(session) = session.as_ref() else {
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
            let stream = hrx::ffi::slice_mut(x, x_elements)
                .map_err(|e| krea2_session::Error::invalid(e.to_string()))?;
            let modulation = hrx::ffi::slice(mods, mods_elements)
                .map_err(|e| krea2_session::Error::invalid(e.to_string()))?;
            let cos = hrx::ffi::slice(cos, rope_elements)
                .map_err(|e| krea2_session::Error::invalid(e.to_string()))?;
            let sin = hrx::ffi::slice(sin, rope_elements)
                .map_err(|e| krea2_session::Error::invalid(e.to_string()))?;
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
        return KREA2_INVALID_ARGUMENT;
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
        return KREA2_INVALID_ARGUMENT;
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
                // The bundle is already compiled by the time a session shares
                // it; only the smoothed attention's preparation kernels could
                // still need a compiler, and they take LOOM_COMPILE or PATH.
                None,
            )?);
            Ok(())
        }),
    );
    if let Some(session) = created {
        *out = Box::into_raw(Box::new(SessionHandle { session }));
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

#[cfg(test)]
mod native_contracts {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn constructor_failures_clear_output_handles_and_terminate_errors() {
        for capacity in [1usize, 8, 128] {
            let mut error = [0x55u8; 128];
            let mut output = std::ptr::dangling_mut::<SessionHandle>();
            let status = unsafe {
                krea2_create(
                    null(),
                    null(),
                    -1,
                    1,
                    &mut output,
                    error.as_mut_ptr().cast(),
                    capacity,
                )
            };
            assert_eq!(status, KREA2_INVALID_ARGUMENT);
            assert!(output.is_null());
            assert!(error[..capacity].contains(&0));
            assert!(error[capacity..].iter().all(|&v| v == 0x55));
        }
    }

    #[test]
    fn null_session_and_shared_weight_handles_fail_without_gpu_initialization() {
        let mut error = [0i8; 128];
        let status = unsafe {
            krea2_run(
                null_mut(),
                null_mut(),
                0,
                null(),
                0,
                null(),
                null(),
                0,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, KREA2_INVALID_ARGUMENT);
        assert_ne!(error[0], 0);
        let mut output = std::ptr::dangling_mut::<SessionHandle>();
        let status = unsafe {
            krea2_create_shared(
                null(),
                null(),
                16,
                1,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(status, KREA2_INVALID_ARGUMENT);
        assert!(output.is_null());
        assert_eq!(unsafe { krea2_weights_bits(null()) }, 0);
        unsafe {
            krea2_destroy(null_mut());
            krea2_weights_release(null_mut());
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(!maps.contains("libhrx.so"));
    }

    #[test]
    fn panic_containment_uses_the_shared_ffi_boundary() {
        let mut error = [0i8; 64];
        let status = unsafe {
            guard(error.as_mut_ptr(), error.len(), || -> krea2_session::Result<()> {
                panic!("test model panic")
            })
        };
        assert_eq!(status, KREA2_ERROR);
        assert_ne!(error[0], 0);
    }
}
