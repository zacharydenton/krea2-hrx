//! Synchronized stage timings on stderr, enabled by `KREA2_NATIVE_PROFILE=1`.
//! Synchronization affects latency, so disable profiling for performance measurements.
use hrx::device;

use crate::Result;

pub struct Profile {
    scope: &'static str,
    last: Option<std::time::Instant>,
}

impl Profile {
    /// Enabled only by `KREA2_NATIVE_PROFILE=1`, read per scope so a long-lived
    /// pipeline picks up the setting without being rebuilt.
    pub fn new(scope: &'static str) -> Profile {
        let enabled =
            std::env::var_os("KREA2_NATIVE_PROFILE").is_some_and(|value| value == "1");
        if !enabled {
            return Profile { scope, last: None };
        }
        // A first mark that did not wait would bill the previous stage's tail.
        let _ = device().synchronize();
        Profile { scope, last: Some(std::time::Instant::now()) }
    }

    pub fn mark(&mut self, stage: &str) -> Result<()> {
        let Some(last) = self.last else {
            return Ok(());
        };
        device().synchronize()?;
        let now = std::time::Instant::now();
        eprintln!(
            "native profile {} / {stage}: {:.3} ms",
            self.scope,
            (now - last).as_secs_f64() * 1e3
        );
        self.last = Some(now);
        Ok(())
    }
}
