//! Native streams on a caller's selected device and allocation domain.
use crate::Result;
use hrx::{Stream, inference::ModelContext};

pub(crate) fn native_stream(context: &ModelContext) -> Result<Stream> {
    let mut stream = hrx::Device::open(context.runtime().gpu()?.index())?.stream()?;
    if let Some(budget) = context.runtime().memory_budget() {
        stream = stream.with_memory_budget(budget.clone());
    }
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires gfx1151"]
    fn native_clients_share_the_context_device_and_allocation_counter() {
        let manager = hrx::residency::ResidencyManager::new(8192).unwrap();
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(manager.budget()),
            ..Default::default()
        })
        .unwrap();
        let first = native_stream(&context).unwrap();
        let second = native_stream(&context).unwrap();
        assert_eq!(first.device_id(), second.device_id());
        let a = first.allocate(4096).unwrap();
        let b = second.allocate(4096).unwrap();
        assert_eq!(manager.budget().reserved_bytes(), 8192);
        assert!(second.allocate(1).is_err());
        drop(a);
        assert_eq!(manager.budget().reserved_bytes(), 4096);
        drop(b);
        drop(first);
        drop(second);
        assert_eq!(manager.budget().reserved_bytes(), 0);
    }
}
