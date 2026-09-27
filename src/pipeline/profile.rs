//! Synchronized stage timings on stderr, enabled by `KREA2_NATIVE_PROFILE=1`.
//! Synchronization affects latency, so disable profiling for performance measurements.
use hrx::Stream;

use super::Result;

/// One scope's stage clock, or nothing when profiling is off.
pub struct Profile {
    scope: &'static str,
    last: Option<std::time::Instant>,
}

impl Profile {
    /// Enabled only by `KREA2_NATIVE_PROFILE=1`, read per scope so a long-lived
    /// pipeline picks up the setting without being rebuilt.
    pub fn new(stream: &mut Stream, scope: &'static str) -> Profile {
        if !crate::kernels::native_profile() {
            return Profile { scope, last: None };
        }
        // A first mark that did not wait would bill the previous stage's tail.
        let _ = stream.synchronize();
        Profile { scope, last: Some(std::time::Instant::now()) }
    }

    /// Ends `stage` and prints its time, after waiting for its work.
    pub fn mark(&mut self, stream: &mut Stream, stage: &str) -> Result<()> {
        let Some(last) = self.last else {
            return Ok(());
        };
        stream.synchronize()?;
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
