//! A compiled Loom export, ready to dispatch.
use std::sync::Arc;

use crate::device::c_string;
use crate::{check, device, sys, Args, Error, Result};

struct Executable(sys::hrx_executable_t);

// The handle is only used through &Device, under its mutex.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        // Safety: the executable is released once, after draining any work
        // that may still be running from it.
        unsafe {
            let _ = device().synchronize();
            sys::hrx_executable_release(self.0);
        }
    }
}

/// One export of one HSACO. Cloning shares the loaded executable.
#[derive(Clone)]
pub struct Kernel {
    executable: Arc<Executable>,
    ordinal: u32,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel(export {})", self.ordinal)
    }
}

impl Kernel {
    /// Loads `path` for gfx1151 and looks up `symbol`.
    pub fn load(path: &std::path::Path, symbol: &str) -> Result<Kernel> {
        let path_text =
            path.to_str().ok_or_else(|| Error(format!("{} is not UTF-8", path.display())))?;
        let path_c = c_string(path_text)?;
        let symbol_c = c_string(symbol)?;
        let family = c_string("amdgpu")?;
        let target = c_string("gfx1151")?;
        // Safety: every string outlives the call; HRX writes the handle on
        // success, and the ordinal lookup happens on a loaded executable.
        unsafe {
            let mut executable: sys::hrx_executable_t = std::ptr::null_mut();
            check(sys::hrx_executable_load_file(
                device().raw(),
                path_c.as_ptr(),
                family.as_ptr(),
                target.as_ptr(),
                &mut executable,
            ))?;
            let executable = Executable(executable);
            let mut ordinal: u32 = 0;
            check(sys::hrx_executable_lookup_export_by_name(
                executable.0,
                symbol_c.as_ptr(),
                &mut ordinal,
            ))?;
            Ok(Kernel { executable: Arc::new(executable), ordinal })
        }
    }

    /// One dispatch: `grid` workgroups of `block` work items, subgroup size 32.
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], args: &Args) -> Result<()> {
        let workgroup = u64::from(block[0]) * u64::from(block[1]) * u64::from(block[2]);
        if grid.iter().chain(block.iter()).any(|&n| n == 0) || workgroup > 1024 {
            return Err(Error("invalid GPU launch".into()));
        }
        let config = sys::hrx_dispatch_config_t {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: 32,
        };
        device().dispatch(self.executable.0, self.ordinal, &config, args.as_bytes())
    }

    /// The common case: a 2-D grid of 1-D workgroups.
    pub fn launch_2d(&self, grid_x: u32, grid_y: u32, threads: u32, args: &Args) -> Result<()> {
        self.launch([grid_x, grid_y, 1], [threads, 1, 1], args)
    }
}
