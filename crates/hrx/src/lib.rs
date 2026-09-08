//! gfx1151 dispatch through the public HRX C API. No HIP, no GPU math library.
//!
//! One process-wide [`Device`]: one gfx1151 device, one ordered stream, and a
//! registry of every live allocation. Kernels are dispatched with custom direct
//! arguments and an execution barrier after each launch, so operations are
//! ordered without a host synchronization between them.
//!
//! Device addresses are handed out as [`DevicePtr`], a `Copy` address that is
//! never dereferenced on the host. Offsetting one stays legal, and the registry
//! resolves any address inside an allocation back to (buffer, offset) at
//! dispatch time — which is what lets tensors be views into one big allocation.
pub mod sys;

mod args;
mod device;
mod kernel;

pub use args::Args;
pub use device::{device, try_device, Buffer, Device, DevicePtr};
pub use kernel::Kernel;

/// Anything libhrx or this crate rejects. The message is HRX's own where there
/// is one, so failures read the same as they did from the C++ host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Error(message.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns a libhrx status into a `Result`, taking ownership of its message.
pub(crate) fn check(status: sys::hrx_status_t) -> Result<()> {
    if sys::is_ok(status) {
        return Ok(());
    }
    // Safety: the status is non-null, so it carries a message payload; HRX
    // allocates the string and we hand it straight back.
    let text = unsafe {
        let mut message: *mut std::ffi::c_char = std::ptr::null_mut();
        let mut size: usize = 0;
        sys::hrx_status_ignore(sys::hrx_status_to_string(status, &mut message, &mut size));
        let text = if message.is_null() {
            "unknown HRX error".to_string()
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(message as *const u8, size))
                .into_owned()
        };
        sys::hrx_status_free_message(message);
        sys::hrx_status_ignore(status);
        text
    };
    Err(Error(text))
}
