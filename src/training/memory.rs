//! System RAM checks for a GPU that shares physical memory with the desktop.
use crate::{Error, Result};
use hrx::{
    inference::ModelContext,
    residency::{MemoryReservation, ResidencyManager},
};

const GIB: usize = 1 << 30;
// One system reserve at startup and during execution. Allocation estimates
// account for training's temporary host buffers separately; this is a policy
// floor for the desktop/other processes, not a measured model requirement.
const SYSTEM_RESERVE: usize = 8 * GIB;

/// A private allocation ceiling backed by an admission reservation in the caller's
/// shared budget. HRX 0.8 supports one budget per stream, so reserve the whole
/// ceiling before replacing the stream's allocation counter. This never charges
/// the same bytes twice to the global budget, or silently bypasses that budget.
///
/// Declare before streams/buffers in local scopes, and AFTER them in owner structs:
/// its reservation must outlive all allocations, staging and queued uses.
pub(super) struct RunBudget {
    local: ResidencyManager,
    parent: Option<MemoryReservation>,
    context: ModelContext,
}

impl RunBudget {
    pub(super) fn new(context: &ModelContext, bytes: usize) -> Result<Self> {
        let local = ResidencyManager::new(bytes)?;
        let parent = context.runtime().memory_budget().map(|b| b.reserve(bytes)).transpose()?;
        Ok(Self { local, parent, context: context.clone() })
    }

    pub(super) fn stream(&self) -> Result<hrx::Stream> {
        Ok(crate::context::native_stream(&self.context)?
            .with_memory_budget(self.local.budget()))
    }
}

impl Drop for RunBudget {
    fn drop(&mut self) {
        // HRX can quarantine backing and its charges after an uncertain device
        // failure. Never release the enclosing reservation while any such
        // storage remains live. Normal owners drain and drop everything first.
        if self.local.budget().reserved_bytes() != 0
            && let Some(parent) = self.parent.take()
        {
            std::mem::forget(parent);
        }
    }
}

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

    fn context(parent: &ResidencyManager) -> ModelContext {
        ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(parent.budget()),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn shared_admission_preserves_both_limits_and_rolls_back_errors() {
        let parent = ResidencyManager::new(1024).unwrap();
        let ctx = context(&parent);
        let sibling = parent.budget().reserve(256).unwrap();
        assert!(RunBudget::new(&ctx, 769).is_err());
        assert_eq!(parent.budget().reserved_bytes(), 256);
        {
            let run = RunBudget::new(&ctx, 512).unwrap();
            assert_eq!(parent.budget().reserved_bytes(), 768);
            assert!(run.local.budget().reserve(513).is_err());
            let private = run.local.budget().reserve(512).unwrap();
            assert_eq!(parent.budget().reserved_bytes(), 768);
            assert!(parent.budget().reserve(257).is_err());
            drop(private);
        }
        assert_eq!(parent.budget().reserved_bytes(), 256);
        let fail = || -> Result<()> {
            let run = RunBudget::new(&ctx, 768)?;
            let _allocation = run.local.budget().reserve(42)?;
            Err(Error::invalid("injected model load failure"))
        };
        assert!(fail().is_err());
        assert_eq!(parent.budget().reserved_bytes(), 256);
        drop(sibling);
        assert_eq!(parent.budget().reserved_bytes(), 0);
    }

    #[test]
    fn live_or_quarantined_charges_never_release_the_parent_allowance() {
        let parent = ResidencyManager::new(1024).unwrap();
        let run = RunBudget::new(&context(&parent), 512).unwrap();
        let live = run.local.budget().reserve(64).unwrap();
        drop(run);
        assert_eq!(parent.budget().reserved_bytes(), 512);
        drop(live);
        // Once an owner escapes its scope, conservatively retain admission.
        assert_eq!(parent.budget().reserved_bytes(), 512);
    }

    #[test]
    fn standalone_budget_still_enforces_its_ceiling_without_a_parent() {
        let ctx = ModelContext::new(Default::default()).unwrap();
        let run = RunBudget::new(&ctx, 512).unwrap();
        let allocation = run.local.budget().reserve(512).unwrap();
        assert!(run.local.budget().reserve(1).is_err());
        drop(allocation);
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn native_training_allocations_keep_global_admission_until_drained() {
        let parent = ResidencyManager::new(4 << 20).unwrap();
        let ctx = context(&parent);
        let peer = crate::context::native_stream(&ctx).unwrap();
        let peer_buffer = peer.allocate(4096).unwrap();
        {
            let run = RunBudget::new(&ctx, 2 << 20).unwrap();
            let mut stream = run.stream().unwrap();
            assert_eq!(stream.device_id(), peer.device_id());
            assert_eq!(parent.budget().reserved_bytes(), (2 << 20) + 4096);
            assert!(stream.allocate((2 << 20) + 1).is_err());
            assert!(peer.allocate(2 << 20).is_err());
            let buffer = stream.allocate(4096).unwrap();
            stream.fill(buffer.binding(), 0x5a).unwrap();
            let mut out = [0u8; 4096];
            stream.read_blocking(buffer.binding(), &mut out).unwrap();
            assert_eq!(out, [0x5a; 4096]);
            // Queued work and transfer staging are drained on scope exit.
        }
        assert_eq!(parent.budget().reserved_bytes(), 4096);
        drop(peer_buffer);
        drop(peer);
        assert_eq!(parent.budget().reserved_bytes(), 0);
    }

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
