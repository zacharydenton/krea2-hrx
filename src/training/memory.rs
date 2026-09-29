//! System RAM checks for a GPU that shares physical memory with the desktop.
use crate::{Error, Result};

const GIB: usize = 1 << 30;
// One system reserve at startup and during execution. Allocation estimates
// account for training's temporary host buffers separately; this is a policy
// floor for the desktop/other processes, not a measured model requirement.
const SYSTEM_RESERVE: usize = 8 * GIB;

fn available_bytes(meminfo: &str) -> Result<usize> {
    let value = meminfo.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == "MemAvailable:").then(|| {
            let kib = fields.next()?.parse::<usize>().ok()?;
            (fields.next()? == "kB").then_some(kib)?.checked_mul(1024)
        })?
    });
    value.ok_or_else(|| Error::invalid("missing or invalid MemAvailable in /proc/meminfo"))
}

fn require(available: usize, allocations: usize, reserve: usize) -> Result<()> {
    if available.saturating_sub(reserve) < allocations || available < reserve {
        return Err(Error::invalid(format!(
            "insufficient system RAM: {:.2} GiB available, {:.2} GiB planned plus {:.0} GiB headroom; release other memory-heavy work before training",
            available as f64 / GIB as f64,
            allocations as f64 / GIB as f64,
            reserve as f64 / GIB as f64,
        )));
    }
    Ok(())
}

fn available() -> Result<usize> {
    available_bytes(&std::fs::read_to_string("/proc/meminfo").map_err(crate::lora::io)?)
}

pub(super) fn before_load(planned: usize) -> Result<()> {
    require(available()?, planned, SYSTEM_RESERVE)
}

pub(super) fn during_run() -> Result<()> {
    require(available()?, 0, SYSTEM_RESERVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_available_ram_without_counting_swap_as_gpu_capacity() {
        let info = "MemFree: 100 kB\nSwapFree: 999999999 kB\nMemAvailable: 4096 kB\n";
        assert_eq!(available_bytes(info).unwrap(), 4 * 1024 * 1024);
        assert!(available_bytes("MemFree: 999999999 kB\n").is_err());
        assert!(available_bytes("MemAvailable: 20 MB\n").is_err());
        assert!(available_bytes("MemAvailable: invalid kB\n").is_err());
        assert!(available_bytes(&format!("MemAvailable: {} kB\n", usize::MAX)).is_err());
    }

    #[test]
    fn leaves_headroom_and_rejects_low_memory_without_overflow() {
        let planned = 64 * GIB;
        let available = planned + SYSTEM_RESERVE;
        assert!(require(available, planned, SYSTEM_RESERVE).is_ok());
        assert!(require(available - 1, planned, SYSTEM_RESERVE).is_err());
        // Allocating the complete plan leaves exactly the runtime floor.
        assert!(require(available - planned, 0, SYSTEM_RESERVE).is_ok());
        assert!(require(available - planned - 1, 0, SYSTEM_RESERVE).is_err());
        assert!(require(0, 0, SYSTEM_RESERVE).is_err());
        assert!(require(usize::MAX, usize::MAX, SYSTEM_RESERVE).is_err());
    }
}
