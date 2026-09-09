//! BF16 device matrices and pooled allocations.
//! A [`Pool`] reuses released buffers up to its cache limit. Tensors and views
//! retain ownership of their backing allocation.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use hrx::{device, Buffer, DevicePtr};

use crate::{Error, Result};

/// Cached device memory, capped so a long run does not hoard it.
const LIMIT: usize = 512 << 20;

#[derive(Default)]
struct FreeList {
    cached: usize,
    blocks: BTreeMap<usize, Vec<Buffer>>,
}

/// A source of device buffers that reuses what it has been given back.
#[derive(Default)]
pub struct Pool {
    free: Mutex<FreeList>,
    stream: std::sync::OnceLock<Arc<hrx::Device>>,
}

impl Pool {
    pub fn new() -> Arc<Pool> {
        Arc::new(Pool::default())
    }

    /// A buffer of at least `bytes`. A cached block is taken when it is not
    /// more than twice the size asked for, which is what keeps the free list
    /// from returning a 100 MB block for a 1 KB tensor.
    pub(crate) fn check_stream(&self) -> Result<()> {
        let current = hrx::try_device()?;
        let owner = self.stream.get_or_init(|| current.clone());
        if !Arc::ptr_eq(owner, &current) {
            return Err(Error("a tensor pool must only be used on its owning stream".into()));
        }
        Ok(())
    }

    fn take(self: &Arc<Self>, bytes: usize) -> Result<Pooled> {
        self.check_stream()?;
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let reusable = free
            .blocks
            .range_mut(bytes..)
            .next()
            .filter(|(&size, _)| size - bytes <= bytes)
            .map(|(&size, blocks)| (size, blocks.pop()));
        if let Some((size, Some(buffer))) = reusable {
            free.cached -= size;
            if free.blocks[&size].is_empty() {
                free.blocks.remove(&size);
            }
            return Ok(Pooled { buffer: Some(buffer), pool: self.clone() });
        }
        drop(free);
        Ok(Pooled { buffer: Some(device().allocate(bytes)?), pool: self.clone() })
    }

    fn give_back(&self, buffer: Buffer) {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.len() > LIMIT.saturating_sub(free.cached) {
            return; // dropping the buffer releases it
        }
        free.cached += buffer.len();
        free.blocks.entry(buffer.len()).or_default().push(buffer);
    }

    /// Raw pooled bytes, for the one intermediate that is not bf16: attention
    /// scores, which the GEMM writes as float32.
    pub fn scratch(self: &Arc<Self>, bytes: usize) -> Result<Scratch> {
        if bytes == 0 {
            return Err(Error("empty scratch".into()));
        }
        Ok(Scratch(self.take(bytes)?))
    }
}

/// Pooled device bytes with no shape.
pub struct Scratch(Pooled);

impl Scratch {
    pub fn ptr(&self) -> DevicePtr {
        self.0.ptr()
    }
}

/// A buffer that returns to its pool when dropped.
struct Pooled {
    buffer: Option<Buffer>,
    pool: Arc<Pool>,
}

impl Pooled {
    fn ptr(&self) -> DevicePtr {
        self.buffer.as_ref().expect("a live buffer").ptr()
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.give_back(buffer);
        }
    }
}

/// Backing allocation retained by a tensor and its views.
enum Storage {
    Owned(Pooled),
    /// The allocation is held only to keep it alive; the address is what the
    /// kernels get.
    Shared {
        _alive: Arc<Buffer>,
        at: DevicePtr,
    },
}

impl Storage {
    fn ptr(&self) -> DevicePtr {
        match self {
            Storage::Owned(pooled) => pooled.ptr(),
            Storage::Shared { at, .. } => *at,
        }
    }
}

/// BF16 device matrix with a checked, read-only shape.
/// Cloning shares storage; [`Tensor::view`] creates a window without copying.
#[derive(Clone)]
pub struct Tensor {
    rows: usize,
    cols: usize,
    storage: Arc<Storage>,
    offset: usize,
}

impl Tensor {
    /// Allocates uninitialized storage. Initialize every element before reading it;
    /// call [`Tensor::zero`] when zero-filled storage is required.
    pub fn new(pool: &Arc<Pool>, rows: usize, cols: usize) -> Result<Tensor> {
        let storage = pool.take(bytes(rows, cols)?)?;
        Ok(Tensor { rows, cols, storage: Arc::new(Storage::Owned(storage)), offset: 0 })
    }

    /// A matrix over part of an allocation someone else made — a weight, read
    /// by an operation that takes tensors. The tensor holds a share of the
    /// allocation, so the view cannot outlive it.
    pub fn shared(
        buffer: &Arc<Buffer>,
        at: DevicePtr,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let wanted = bytes(rows, cols)?;
        let inside = at
            .address()
            .checked_sub(buffer.ptr().address())
            .and_then(|offset| offset.checked_add(wanted))
            .is_some_and(|end| end <= buffer.len());
        if !inside {
            return Err(Error("a view outside its allocation".into()));
        }
        Ok(Tensor {
            rows,
            cols,
            storage: Arc::new(Storage::Shared { _alive: Arc::clone(buffer), at }),
            offset: 0,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn from_slice(
        pool: &Arc<Pool>,
        values: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        if values.len() != rows * cols {
            return Err(Error("upload size".into()));
        }
        let tensor = Tensor::new(pool, rows, cols)?;
        device().write(tensor.ptr(), values)?;
        Ok(tensor)
    }

    pub fn size(&self) -> usize {
        // Checked when the tensor was made, so this cannot overflow here.
        self.rows * self.cols
    }

    pub fn ptr(&self) -> DevicePtr {
        self.storage.ptr().offset(self.offset)
    }

    /// A window of `rows * cols` elements, `skip` elements into this tensor.
    pub fn view(&self, rows: usize, cols: usize, skip: usize) -> Result<Tensor> {
        let wanted = elements(rows, cols)?;
        let offset = skip
            .checked_mul(2)
            .and_then(|skipped| self.offset.checked_add(skipped))
            .ok_or_else(|| Error("tensor view".into()))?;
        if skip.checked_add(wanted).is_none_or(|end| end > self.size()) {
            return Err(Error("tensor view".into()));
        }
        Ok(Tensor { rows, cols, storage: self.storage.clone(), offset })
    }

    pub fn download(&self) -> Result<Vec<u16>> {
        let mut values = vec![0u16; self.size()];
        device().read(&mut values, self.ptr())?;
        Ok(values)
    }

    pub fn zero(&self) -> Result<()> {
        device().zero(self.ptr(), self.size() * 2)?;
        Ok(())
    }
}

/// A shape's element count, refusing an empty or unrepresentable one.
fn elements(rows: usize, cols: usize) -> Result<usize> {
    match rows.checked_mul(cols) {
        Some(0) | None => Err(Error("tensor shape".into())),
        Some(count) => Ok(count),
    }
}

/// Checked byte count, including the two bytes per bf16 element.
fn bytes(rows: usize, cols: usize) -> Result<usize> {
    elements(rows, cols)?.checked_mul(2).ok_or_else(|| Error("tensor shape".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_tensor_round_trips_through_the_device() {
        let pool = Pool::new();
        let values: Vec<u16> = (0..64u16).map(|i| i.wrapping_mul(577)).collect();
        let tensor = Tensor::from_slice(&pool, &values, 8, 8).expect("upload");
        assert_eq!(tensor.download().expect("download"), values);
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_view_reads_the_rows_it_names() {
        let pool = Pool::new();
        let values: Vec<u16> = (0..64u16).collect();
        let tensor = Tensor::from_slice(&pool, &values, 8, 8).expect("upload");
        let second = tensor.view(1, 8, 8).expect("the second row");
        assert_eq!(second.download().expect("download"), &values[8..16]);
        assert!(tensor.view(2, 8, 56).is_err(), "a view past the end is refused");
    }

    #[test]
    fn a_shape_that_cannot_be_indexed_is_refused() {
        // No device needed: these fail on arithmetic.
        assert!(elements(0, 8).is_err(), "an empty tensor is not a tensor");
        assert!(elements(8, 0).is_err());
        assert!(elements(usize::MAX, 2).is_err(), "rows * cols must not wrap");
        assert_eq!(elements(3, 4).expect("a shape"), 12);
        // A count that fits in a usize whose byte count does not: the bounds
        // are checked in bytes, so this has to fail before they are.
        assert!(elements(1, 1 << 63).is_ok(), "the element count itself fits");
        assert!(bytes(1, 1 << 63).is_err(), "twice it does not");
        assert_eq!(bytes(3, 4).expect("a shape"), 24);
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_shared_view_keeps_its_allocation_and_stays_inside_it() {
        let buffer = std::sync::Arc::new(device().allocate(64 * 2).expect("an allocation"));
        let base = buffer.ptr();
        // Past the end, and before the start: both are outside.
        assert!(Tensor::shared(&buffer, base, 8, 9).is_err(), "past the end");
        assert!(Tensor::shared(&buffer, base.offset(4), 8, 8).is_err(), "past the end");
        assert!(
            Tensor::shared(&buffer, DevicePtr::from_address(base.address() - 8), 1, 1).is_err(),
            "before the start"
        );

        // The byte count, not the element count, is what the bound is
        // against: 2^63 elements is a usize but 2^64 bytes is not.
        assert!(Tensor::shared(&buffer, base, 1, 1 << 63).is_err(), "byte count wraps");

        let view = Tensor::shared(&buffer, base.offset(16), 4, 4).expect("a view");
        assert_eq!((view.rows(), view.cols(), view.ptr()), (4, 4, base.offset(16)));
        // The view owns a share, so dropping the caller's handle keeps the
        // memory mapped and the address valid.
        drop(buffer);
        assert_eq!(view.ptr(), base.offset(16));
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn dropped_storage_comes_back_from_the_pool() {
        let pool = Pool::new();
        let first = Tensor::new(&pool, 16, 16).expect("a tensor").ptr();
        // Dropped above, so the same block should serve the next request.
        let second = Tensor::new(&pool, 16, 16).expect("a tensor");
        assert_eq!(second.ptr(), first, "the pool did not reuse the block");
        // A much smaller request must not take a much larger block.
        let small = Tensor::new(&pool, 1, 4).expect("a small tensor");
        assert_ne!(small.ptr(), first);
    }
}
