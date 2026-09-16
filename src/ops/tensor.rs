//! BF16 device matrices backed by HRX pooled allocations.
use std::sync::Arc;

use hrx::{Buffer, BufferPool, PooledBuffer, Stream, View};

use super::{Error, Result};

/// Backing allocation retained by a tensor and its views, and where in it this
/// tensor starts. Views are borrowed from the allocation rather than taken as
/// addresses, so a tensor can no longer outlive the memory it names.
enum Storage {
    Owned(PooledBuffer),
    Shared { buffer: Arc<Buffer>, base: usize },
}

impl Storage {
    fn buffer(&self) -> &Buffer {
        match self {
            Storage::Owned(pooled) => pooled.buffer(),
            Storage::Shared { buffer, .. } => buffer,
        }
    }

    fn base(&self) -> usize {
        match self {
            Storage::Owned(_) => 0,
            Storage::Shared { base, .. } => *base,
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
    pub fn new(
        pool: &Arc<BufferPool>,
        stream: &Stream,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let storage = pool.acquire(stream, bytes(rows, cols)?)?;
        Ok(Tensor { rows, cols, storage: Arc::new(Storage::Owned(storage)), offset: 0 })
    }

    /// A matrix over part of an allocation someone else made — a weight, read
    /// by an operation that takes tensors. The tensor holds a share of the
    /// allocation, so the view cannot outlive it.
    /// `base` is a byte offset into `buffer`, replacing the address arithmetic
    /// the address-based runtime required.
    pub fn shared(
        buffer: &Arc<Buffer>,
        base: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let wanted = bytes(rows, cols)?;
        if base.checked_add(wanted).is_none_or(|end| end > buffer.bytes()) {
            return Err(Error("a view outside its allocation".into()));
        }
        Ok(Tensor {
            rows,
            cols,
            storage: Arc::new(Storage::Shared { buffer: Arc::clone(buffer), base }),
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
        pool: &Arc<BufferPool>,
        stream: &mut Stream,
        values: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        if values.len() != rows * cols {
            return Err(Error("upload size".into()));
        }
        let tensor = Tensor::new(pool, stream, rows, cols)?;
        tensor.upload(stream, values)?;
        Ok(tensor)
    }

    /// Writes `values` over this tensor's span. Queued, so it is ordered before
    /// any later dispatch on the same stream without draining it.
    pub fn upload(&self, stream: &mut Stream, values: &[u16]) -> Result<()> {
        if values.len() != self.size() {
            return Err(Error("upload size".into()));
        }
        stream.upload(self.binding()?, bytemuck::cast_slice(values))?;
        Ok(())
    }

    pub fn size(&self) -> usize {
        // Checked when the tensor was made, so this cannot overflow here.
        self.rows * self.cols
    }

    /// This tensor's byte offset within its backing allocation.
    fn at(&self) -> usize {
        self.storage.base() + self.offset
    }

    /// This tensor's span, as a kernel binding.
    pub fn binding(&self) -> Result<View<'_>> {
        self.storage
            .buffer()
            .try_slice(self.at(), self.size() * 2)
            .map_err(|e| Error(e.to_string()))
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

    pub fn download(&self, stream: &mut Stream) -> Result<Vec<u16>> {
        let mut values = vec![0u16; self.size()];
        stream.read_blocking(self.binding()?, bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }

    pub fn zero(&self, stream: &mut Stream) -> Result<()> {
        stream.fill(self.binding()?, 0)?;
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
        let mut stream = Stream::open().expect("a stream");
        let pool = BufferPool::new();
        let values: Vec<u16> = (0..64u16).map(|i| i.wrapping_mul(577)).collect();
        let tensor = Tensor::from_slice(&pool, &mut stream, &values, 8, 8).expect("upload");
        assert_eq!(tensor.download(&mut stream).expect("download"), values);
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_view_reads_the_rows_it_names() {
        let mut stream = Stream::open().expect("a stream");
        let pool = BufferPool::new();
        let values: Vec<u16> = (0..64u16).collect();
        let tensor = Tensor::from_slice(&pool, &mut stream, &values, 8, 8).expect("upload");
        let second = tensor.view(1, 8, 8).expect("the second row");
        assert_eq!(second.download(&mut stream).expect("download"), &values[8..16]);
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
        let stream = Stream::open().expect("a stream");
        let buffer = std::sync::Arc::new(stream.allocate(64 * 2).expect("an allocation"));
        // Past the end, from the base and from an offset. There is no longer a
        // "before the start" case to test: the base is an unsigned offset into
        // the allocation, so an address before it cannot be named at all.
        assert!(Tensor::shared(&buffer, 0, 8, 9).is_err(), "past the end");
        assert!(Tensor::shared(&buffer, 8, 8, 8).is_err(), "past the end");

        // The byte count, not the element count, is what the bound is
        // against: 2^63 elements is a usize but 2^64 bytes is not.
        assert!(Tensor::shared(&buffer, 0, 1, 1 << 63).is_err(), "byte count wraps");

        let view = Tensor::shared(&buffer, 32, 4, 4).expect("a view");
        assert_eq!((view.rows(), view.cols(), view.at()), (4, 4, 32));
        // The view owns a share, so dropping the caller's handle keeps the
        // allocation mapped and the binding valid.
        drop(buffer);
        assert_eq!(view.binding().expect("a binding").len(), 32);
    }

    /// The hazard device-scoped buffers opened up: a block returns to the free
    /// list while the work using it is still queued, so reissuing it to a
    /// second stream races the first. HRX will not object -- the buffer belongs
    /// to the device -- so the pool has to.
    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_pool_refuses_a_second_stream_rather_than_racing_it() {
        let device = hrx::Device::open(0).expect("a device");
        let first = device.stream().expect("first stream");
        let second = device.stream().expect("second stream");
        let pool = BufferPool::new();

        let tensor = Tensor::new(&pool, &first, 16, 16).expect("a tensor");
        drop(tensor); // back on the free list, possibly with work still queued

        let Err(error) = Tensor::new(&pool, &second, 16, 16) else {
            panic!("a second stream must not be handed the recycled block");
        };
        assert!(error.0.contains("serves one stream"), "{}", error.0);
        // The stream it does serve is unaffected.
        assert!(Tensor::new(&pool, &first, 16, 16).is_ok());
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn dropped_storage_comes_back_from_the_pool() {
        let stream = Stream::open().expect("a stream");
        let pool = BufferPool::new();
        let first = Tensor::new(&pool, &stream, 16, 16).expect("a tensor");
        let shared = first.clone();
        drop(first);
        assert_eq!(pool.cached_bytes(), 0, "a live alias prevents recycling");
        drop(shared);
        assert_eq!(pool.cached_bytes(), 512);
        let second = Tensor::new(&pool, &stream, 8, 16).expect("a smaller tensor");
        assert_eq!(second.binding().unwrap().owner().bytes(), 512);
        assert_eq!(pool.cached_bytes(), 0, "the cached block was consumed");
        drop(second);
        assert_eq!(pool.cached_bytes(), 512);
    }
}
