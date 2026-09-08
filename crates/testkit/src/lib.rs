//! Auxiliary-kernel test C API in `libnative_ops_test.so`.
//! Used by `tests/test_native_ops.py` for NumPy and Torch oracle comparisons.
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use hrx::{device, Args, Buffer, DevicePtr};
use loom::{auxiliary_kernel, Config};

/// Retain allocations until the C caller frees their raw addresses.
static LIVE: std::sync::Mutex<Option<std::collections::HashMap<usize, Buffer>>> =
    std::sync::Mutex::new(None);

fn live() -> std::sync::MutexGuard<'static, Option<std::collections::HashMap<usize, Buffer>>> {
    let mut guard = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(Default::default);
    guard
}

#[no_mangle]
pub extern "C" fn test_alloc(size: usize) -> *mut c_void {
    let Ok(buffer) = device().allocate(size) else {
        return std::ptr::null_mut();
    };
    let address = buffer.ptr().address();
    live().as_mut().expect("initialized").insert(address, buffer);
    address as *mut c_void
}

#[no_mangle]
pub extern "C" fn test_free(pointer: *mut c_void) {
    live().as_mut().expect("initialized").remove(&(pointer as usize));
}

/// A host-to-device or device-to-host copy, whichever the addresses are.
///
/// # Safety
/// The host side must point to `size` readable or writable bytes.
#[no_mangle]
pub unsafe extern "C" fn test_copy(
    destination: *mut c_void,
    source: *const c_void,
    size: usize,
) {
    let known = |pointer: usize| live().as_ref().expect("initialized").contains_key(&pointer);
    let (to_device, from_device) = (known(destination as usize), known(source as usize));
    let _ = match (to_device, from_device) {
        (true, true) => device().copy_device_to_device(
            DevicePtr::from_address(destination as usize),
            DevicePtr::from_address(source as usize),
            size,
        ),
        (true, false) => device().copy_from_host(
            DevicePtr::from_address(destination as usize),
            std::slice::from_raw_parts(source as *const u8, size),
        ),
        (false, true) => device().copy_to_host(
            std::slice::from_raw_parts_mut(destination as *mut u8, size),
            DevicePtr::from_address(source as usize),
        ),
        (false, false) => {
            std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, size);
            Ok(())
        }
    };
}

/// Compiles and launches `name` with `json` as its configuration and `args` as
/// the kernarg blob, then waits. Nonzero means the message is on stderr.
///
/// # Safety
/// `name` and `json` must be NUL-terminated; `args` must point to `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn test_run(
    name: *const c_char,
    json: *const c_char,
    args: *const c_void,
    size: usize,
    grid_x: c_uint,
    grid_y: c_uint,
    threads: c_uint,
) -> c_int {
    match run(name, json, args, size, grid_x, grid_y, threads) {
        Ok(()) => 0,
        Err(message) => {
            eprintln!("{message}");
            1
        }
    }
}

/// # Safety
/// As [`test_run`].
unsafe fn run(
    name: *const c_char,
    json: *const c_char,
    args: *const c_void,
    size: usize,
    grid_x: c_uint,
    grid_y: c_uint,
    threads: c_uint,
) -> Result<(), String> {
    let name = CStr::from_ptr(name).to_str().map_err(|e| e.to_string())?;
    let json = CStr::from_ptr(json).to_str().map_err(|e| e.to_string())?;
    let config: Config = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let mut blob = Args::new();
    blob.raw(std::slice::from_raw_parts(args as *const u8, size))
        .map_err(|e: hrx::Error| e.0)?;
    let kernel = auxiliary_kernel(name, &config, (grid_x, grid_y), None).map_err(|e| e.0)?;
    kernel.launch_2d(grid_x, grid_y, threads, &blob).map_err(|e| e.0)?;
    device().synchronize().map_err(|e| e.0)?;
    Ok(())
}
