//! Owned device-copy boundaries between auxiliary native models and prepared
//! shared-context blocks. Pipeline's state lock excludes concurrent native use.
use super::*;
use hrx::{
    execution::GpuAccess,
    inference::PreparedModel,
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
    Access, PooledBuffer,
};
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) fn native_stream(context: &ModelContext) -> Result<Stream> {
    let mut stream = hrx::Device::open(context.runtime().gpu()?.index())?.stream()?;
    if let Some(budget) = context.runtime().memory_budget() {
        stream = stream.with_memory_budget(budget.clone());
    }
    Ok(stream)
}

/// A failed or panicking native handoff must never reuse its stream.
pub(super) struct Bridge {
    stream: Mutex<Stream>,
    healthy: AtomicBool,
}
impl Bridge {
    pub fn new(stream: Stream) -> Self {
        Self { stream: Mutex::new(stream), healthy: AtomicBool::new(true) }
    }

    pub fn check(&self) -> hrx::Result<()> {
        if !self.healthy.load(Ordering::Acquire) {
            return Err(hrx::Error::DeviceLost("native bridge is quarantined".into()));
        }
        Ok(())
    }

    fn run(&self, copy: impl FnOnce(&mut Stream) -> hrx::Result<()>) -> hrx::Result<()> {
        let mut stream = self
            .stream
            .lock()
            .map_err(|_| hrx::Error::DeviceLost("native bridge is poisoned".into()))?;
        self.check()?;
        // A panic also leaves the bridge unavailable. The scheduler quarantines
        // the callback and all captured owners on an uncertain native failure.
        self.healthy.store(false, Ordering::Release);
        copy(&mut stream)?;
        stream.synchronize()?;
        self.healthy.store(true, Ordering::Release);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlockShape {
    pub width: usize,
    pub height: usize,
    pub text_tokens: usize,
}
impl BlockShape {
    pub fn tokens(self) -> usize {
        self.width / PATCH * (self.height / PATCH) + self.text_tokens
    }
}
pub(super) struct Blocks {
    pub plan: PreparedModel,
    residual: DeviceTensor,
    modulation: DeviceTensor,
}
pub(super) type BlockCache = hrx::plan_cache::PlanCache<BlockShape, Blocks>;
impl Blocks {
    pub fn new(context: &ModelContext, plan: PreparedModel, tokens: usize) -> Result<Self> {
        Ok(Self {
            plan,
            residual: context.allocate(
                TensorDesc::new(DType::BF16, vec![tokens, WIDTH])?.with_layout(Layout::Rows)?,
            )?,
            modulation: context.allocate(TensorDesc::new(DType::F32, vec![28, 6, WIDTH])?)?,
        })
    }
}

impl Pipeline {
    pub(super) fn run_blocks(
        &self,
        stream: &mut Stream,
        blocks: &Blocks,
        residual: &Tensor,
        modulation: PooledBuffer,
    ) -> Result<()> {
        // Native producers use their private stream. Drain before handing those
        // owners to a worker; no host readback is involved in this boundary.
        stream.synchronize()?;
        let bindings = [&blocks.residual, &blocks.modulation].map(|t| GpuAccess {
            view: t.binding().expect("nonempty input"),
            access: Access::Write,
        });
        let source = residual.clone();
        let bridge = self.bridge.clone();
        let mut upload = self.context.runtime().graph();
        // SAFETY: Pipeline's state lock excludes all other native uses of these
        // sources. The closure retains pooled buffers until it drains copies,
        // and all tracked writes are declared. Uncertain failures quarantine
        // both native owners and destination bindings together.
        unsafe {
            upload.gpu_scoped(&bindings, move |views| {
                bridge.run(|stream| {
                    stream.copy(
                        views[0],
                        source.binding().map_err(|e| hrx::Error::Message(e.to_string()))?,
                    )?;
                    stream.copy(views[1], modulation.binding())
                })
            })?;
        }
        let copied = upload.prepare()?.submit()?;
        // No native source may escape to its pool while a handoff is pending,
        // including when tensor validation or slot admission below fails.
        copied.wait()?;
        let inputs = [&blocks.residual, &blocks.modulation]
            .into_iter()
            .map(|t| {
                self.context.tensor(
                    t.desc().clone(),
                    t.binding().expect("nonempty input"),
                    copied.clone(),
                )
            })
            .collect::<hrx::Result<Vec<_>>>()?;
        let result = blocks.plan.submit(&inputs)?;
        let binding = GpuAccess {
            view: result.outputs()[0].binding().expect("nonempty output"),
            access: Access::Read,
        };
        let destination = residual.clone();
        let retained_output = result.outputs()[0].clone();
        let bridge = self.bridge.clone();
        let mut output = self.context.runtime().graph();
        // SAFETY: The destination remains inaccessible to native operations
        // until the wait below succeeds. The closure owns its allocation and
        // reusable stream, retains the prepared output slot, and drains copies.
        unsafe {
            output.gpu_scoped(&[binding], move |views| {
                // Retain the slot lease, not merely its underlying allocation,
                // until completion (or indefinitely on quarantine).
                let _keep_slot = &retained_output;
                bridge.run(|stream| {
                    stream.copy(
                        destination
                            .binding()
                            .map_err(|e| hrx::Error::Message(e.to_string()))?,
                        views[0],
                    )
                })
            })?;
        }
        output.prepare()?.submit_after(std::slice::from_ref(result.completion()))?.wait()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a provisioned GPU runtime"]
    fn failed_or_panicking_bridge_never_runs_again() {
        for panic in [false, true] {
            let bridge = Bridge::new(Stream::open().unwrap());
            bridge.run(|_| Ok(())).unwrap();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bridge.run(|_| {
                    if panic {
                        panic!("injected handoff panic");
                    }
                    Err(hrx::Error::Message("injected handoff failure".into()))
                })
            }));
            if panic {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.unwrap().is_err());
            }
            assert!(bridge.check().is_err());
            let mut ran = false;
            assert!(bridge
                .run(|_| {
                    ran = true;
                    Ok(())
                })
                .is_err());
            assert!(!ran);
        }
    }
}
