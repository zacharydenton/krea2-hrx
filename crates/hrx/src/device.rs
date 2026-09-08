//! The process-wide device: one gfx1151, one ordered stream, one allocation map.
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{check, sys, Error, Result};

/// A device address. Never dereferenced on the host: it is offset, compared,
/// and handed to the driver, and [`Device`] resolves it to (buffer, offset).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct DevicePtr(usize);

impl DevicePtr {
    pub const NULL: DevicePtr = DevicePtr(0);

    pub const fn from_address(address: usize) -> Self {
        DevicePtr(address)
    }

    pub const fn address(self) -> usize {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Adds a byte offset without checking allocation bounds.
    pub const fn offset(self, bytes: usize) -> Self {
        DevicePtr(self.0 + bytes)
    }
}

/// One device allocation, released when the value is dropped.
pub struct Buffer {
    pointer: DevicePtr,
    bytes: usize,
}

impl Buffer {
    pub fn ptr(&self) -> DevicePtr {
        self.pointer
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        device().release(self.pointer);
    }
}

struct Allocation {
    buffer: sys::hrx_buffer_t,
    bytes: usize,
}

struct State {
    stream: sys::hrx_stream_t,
    allocations: BTreeMap<usize, Allocation>,
}

// The handles are owned by this crate and only ever used under the mutex.
unsafe impl Send for State {}

pub struct Device {
    owns_gpu: bool,
    device: sys::hrx_device_t,
    state: Mutex<State>,
}

// Same discipline as State: nothing hands a raw handle out.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

static DEVICE: OnceLock<Result<Device>> = OnceLock::new();

/// The process-wide device, initialized on first use.
///
/// # Panics
/// If HRX cannot be initialized or no gfx1151 is present. Callers that want to
/// report the failure should use [`try_device`] instead.
pub fn device() -> &'static Device {
    try_device().unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_device() -> Result<&'static Device> {
    match DEVICE.get_or_init(Device::open) {
        Ok(device) => Ok(device),
        Err(error) => Err(error.clone()),
    }
}

impl Device {
    fn open() -> Result<Device> {
        // HRX loads the GPU driver itself. Prefer the provider packaged beside
        // this library without touching the caller's library search path.
        if std::env::var_os("IREE_HAL_AMDGPU_LIBHSA_PATH").is_none() {
            if let Some(path) = provider_beside_this_library() {
                std::env::set_var("IREE_HAL_AMDGPU_LIBHSA_PATH", path);
            }
        }
        // Safety: every call below follows the C API's contract, and each
        // handle is released on the error paths before returning.
        unsafe {
            let initialization = sys::hrx_gpu_initialize(0);
            let mut owns_gpu = false;
            if sys::hrx_status_code(initialization) == sys::HRX_STATUS_ALREADY_EXISTS {
                sys::hrx_status_ignore(initialization);
            } else {
                check(initialization)?;
                owns_gpu = true;
            }
            let mut count: c_int = 0;
            let device = match check(sys::hrx_gpu_device_count(&mut count))
                .and_then(|()| find_gfx1151(count))
            {
                Ok(device) => device,
                Err(error) => {
                    if owns_gpu {
                        sys::hrx_status_ignore(sys::hrx_gpu_shutdown());
                    }
                    return Err(error);
                }
            };
            let mut stream: sys::hrx_stream_t = std::ptr::null_mut();
            if let Err(error) = check(sys::hrx_stream_create(device, 0, &mut stream)) {
                sys::hrx_device_release(device);
                if owns_gpu {
                    sys::hrx_status_ignore(sys::hrx_gpu_shutdown());
                }
                return Err(error);
            }
            Ok(Device {
                owns_gpu,
                device,
                state: Mutex::new(State { stream, allocations: BTreeMap::new() }),
            })
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn raw(&self) -> sys::hrx_device_t {
        self.device
    }

    /// Waits for every operation submitted so far.
    pub fn synchronize(&self) -> Result<()> {
        let state = self.lock();
        // Safety: the stream is live for the life of the device.
        unsafe { check(sys::hrx_stream_synchronize(state.stream)) }
    }

    /// Allocates device-local, host-visible memory. Zero bytes allocates four,
    /// so every allocation has a distinct address.
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        let mut state = self.lock();
        let bytes = bytes.max(4);
        // Safety: `buffer` and `pointer` are written by HRX on success only,
        // and the buffer is released if the address cannot be read.
        unsafe {
            let mut buffer: sys::hrx_buffer_t = std::ptr::null_mut();
            check(sys::hrx_buffer_allocate(
                state.stream,
                bytes,
                sys::HRX_MEMORY_TYPE_DEVICE_LOCAL | sys::HRX_MEMORY_TYPE_HOST_VISIBLE,
                sys::HRX_BUFFER_USAGE_DEFAULT | sys::HRX_BUFFER_USAGE_MAPPING_SCOPED,
                &mut buffer,
            ))?;
            let mut pointer: *mut c_void = std::ptr::null_mut();
            if let Err(error) = check(sys::hrx_buffer_get_device_ptr(buffer, &mut pointer)) {
                sys::hrx_buffer_release(buffer);
                return Err(error);
            }
            let address = pointer as usize;
            state.allocations.insert(address, Allocation { buffer, bytes });
            Ok(Buffer { pointer: DevicePtr(address), bytes })
        }
    }

    /// Releases an allocation by base address; unknown addresses are ignored.
    fn release(&self, pointer: DevicePtr) {
        if pointer.is_null() {
            return;
        }
        let mut state = self.lock();
        let Some(allocation) = state.allocations.remove(&pointer.address()) else {
            return;
        };
        // Safety: the buffer came from this map and is released exactly once,
        // after the stream has drained the work that may still reference it.
        unsafe {
            sys::hrx_status_ignore(sys::hrx_stream_synchronize(state.stream));
            sys::hrx_buffer_release(allocation.buffer);
        }
    }

    /// Resolves any address inside a live allocation to (buffer, offset, len).
    fn find(state: &State, pointer: DevicePtr, bytes: usize) -> Result<sys::hrx_buffer_ref_t> {
        let empty =
            sys::hrx_buffer_ref_t { buffer: std::ptr::null_mut(), offset: 0, length: 0 };
        let Some((base, allocation)) =
            state.allocations.range(..=pointer.address()).next_back()
        else {
            return Ok(empty);
        };
        let offset = pointer.address() - base;
        if offset >= allocation.bytes {
            return Ok(empty);
        }
        if bytes > allocation.bytes - offset {
            return Err(Error("GPU buffer span".into()));
        }
        Ok(sys::hrx_buffer_ref_t { buffer: allocation.buffer, offset, length: bytes })
    }

    /// Device to device, or between the host and the device, by which side of
    /// the copy resolves to an allocation.
    pub fn copy_device_to_device(
        &self,
        destination: DevicePtr,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        let state = self.lock();
        let destination = Self::find(&state, destination, bytes)?;
        let source = Self::find(&state, source, bytes)?;
        if destination.buffer.is_null() || source.buffer.is_null() {
            return Err(Error("copy requires a GPU allocation".into()));
        }
        // Safety: both references were resolved from live allocations above.
        unsafe {
            check(sys::hrx_stream_copy_buffer(
                state.stream,
                source.buffer,
                source.offset,
                destination.buffer,
                destination.offset,
                bytes,
            ))?;
            check(sys::hrx_stream_execution_barrier(state.stream))
        }
    }

    /// Uploads a slice of plain data (`u8`, `u16`, `f32`, ...). The element
    /// type only has to be one the GPU and the host agree on byte for byte,
    /// which is what `Pod` means.
    pub fn write<T: bytemuck::Pod>(&self, destination: DevicePtr, source: &[T]) -> Result<()> {
        self.copy_from_host(destination, bytemuck::cast_slice(source))
    }

    /// Reads back into a slice of plain data.
    pub fn read<T: bytemuck::Pod>(
        &self,
        destination: &mut [T],
        source: DevicePtr,
    ) -> Result<()> {
        self.copy_to_host(bytemuck::cast_slice_mut(destination), source)
    }

    pub fn copy_from_host(&self, destination: DevicePtr, source: &[u8]) -> Result<()> {
        let state = self.lock();
        let reference = Self::find(&state, destination, source.len())?;
        if reference.buffer.is_null() {
            return Err(Error("copy requires a GPU allocation".into()));
        }
        // Safety: `source` is a live slice of the length passed, and the
        // destination span was bounds-checked by `find`.
        unsafe {
            check(sys::hrx_stream_synchronize(state.stream))?;
            check(sys::hrx_synchronous_h2d(
                self.device,
                source.as_ptr() as *const c_void,
                reference.buffer,
                reference.offset,
                source.len(),
            ))
        }
    }

    pub fn copy_to_host(&self, destination: &mut [u8], source: DevicePtr) -> Result<()> {
        let state = self.lock();
        let reference = Self::find(&state, source, destination.len())?;
        if reference.buffer.is_null() {
            return Err(Error("copy requires a GPU allocation".into()));
        }
        // Safety: as above, with the roles reversed.
        unsafe {
            check(sys::hrx_stream_synchronize(state.stream))?;
            check(sys::hrx_synchronous_d2h(
                self.device,
                reference.buffer,
                reference.offset,
                destination.as_mut_ptr() as *mut c_void,
                destination.len(),
            ))
        }
    }

    pub fn zero(&self, destination: DevicePtr, bytes: usize) -> Result<()> {
        let state = self.lock();
        let reference = Self::find(&state, destination, bytes)?;
        if reference.buffer.is_null() {
            return Err(Error("zero requires a GPU allocation".into()));
        }
        let pattern: u8 = 0;
        // Safety: the span was bounds-checked, and the pattern outlives the call.
        unsafe {
            check(sys::hrx_stream_fill_buffer(
                state.stream,
                reference.buffer,
                reference.offset,
                bytes,
                &pattern as *const u8 as *const c_void,
                1,
            ))?;
            check(sys::hrx_stream_execution_barrier(state.stream))
        }
    }

    pub(crate) fn dispatch(
        &self,
        executable: sys::hrx_executable_t,
        ordinal: u32,
        config: &sys::hrx_dispatch_config_t,
        arguments: &[u8],
    ) -> Result<()> {
        let state = self.lock();
        // Safety: the executable outlives the dispatch (the Kernel holds it),
        // and the arguments are copied by the driver during the call.
        unsafe {
            check(sys::hrx_stream_dispatch(
                state.stream,
                executable,
                ordinal,
                config,
                arguments.as_ptr() as *const c_void,
                arguments.len(),
                std::ptr::null(),
                0,
                sys::HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
            ))?;
            check(sys::hrx_stream_execution_barrier(state.stream))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut state = self.lock();
        // Safety: drain the stream before releasing its allocations and device.
        // Errors cannot be returned from Drop.
        unsafe {
            sys::hrx_status_ignore(sys::hrx_stream_synchronize(state.stream));
            sys::hrx_stream_release(state.stream);
            for (_, allocation) in std::mem::take(&mut state.allocations) {
                sys::hrx_buffer_release(allocation.buffer);
            }
            sys::hrx_device_release(self.device);
            if self.owns_gpu {
                sys::hrx_status_ignore(sys::hrx_gpu_shutdown());
            }
        }
    }
}

/// The first gfx1151, releasing every other device it had to open to look.
///
/// # Safety
/// Call with the GPU accelerator initialized.
unsafe fn find_gfx1151(count: c_int) -> Result<sys::hrx_device_t> {
    for index in 0..count {
        let mut candidate: sys::hrx_device_t = std::ptr::null_mut();
        check(sys::hrx_gpu_device_get(index, &mut candidate))?;
        let mut architecture = [0 as c_char; 64];
        let property = check(sys::hrx_device_get_property(
            candidate,
            sys::HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.as_mut_ptr() as *mut c_void,
            architecture.len(),
        ));
        if let Err(error) = property {
            sys::hrx_device_release(candidate);
            return Err(error);
        }
        if CStr::from_ptr(architecture.as_ptr()).to_bytes() == b"gfx1151" {
            return Ok(candidate);
        }
        sys::hrx_device_release(candidate);
    }
    Err(Error("gfx1151 GPU required".into()))
}

/// `<this library's directory>/runtime/libhsa-runtime64.so.1`, if it is there.
fn provider_beside_this_library() -> Option<std::path::PathBuf> {
    #[repr(C)]
    struct DlInfo {
        file_name: *const c_char,
        base: *mut c_void,
        symbol_name: *const c_char,
        symbol_address: *mut c_void,
    }
    extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }
    // Safety: dladdr fills the struct for an address inside this library; the
    // string it returns points into the loader's own tables and is only read.
    unsafe {
        let mut info = DlInfo {
            file_name: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol_name: std::ptr::null(),
            symbol_address: std::ptr::null_mut(),
        };
        if dladdr(provider_beside_this_library as *const c_void, &mut info) == 0
            || info.file_name.is_null()
        {
            return None;
        }
        let library = std::path::PathBuf::from(CStr::from_ptr(info.file_name).to_str().ok()?);
        let path = std::fs::canonicalize(library)
            .ok()?
            .parent()?
            .join("runtime/libhsa-runtime64.so.1");
        path.exists().then_some(path)
    }
}

/// Kept for the C ABI's error strings, which quote paths verbatim.
pub(crate) fn c_string(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error("path contains a NUL byte".into()))
}
