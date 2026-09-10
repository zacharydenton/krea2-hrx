//! BF16 device matrices and pooled allocations.
//! A [`Pool`] reuses released buffers up to its cache limit. Tensors and views
//! retain ownership of their backing allocation.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use hrx::{Buffer, Stream, View};

use super::{Error, Result};

/// Cached device memory, capped so a long run does not hoard it.
const LIMIT: usize = 512 << 20;

#[derive(Default)]
struct FreeList {
    cached: usize,
    blocks: BTreeMap<usize, Vec<Buffer>>,
    /// The stream every block here came from, once one has. See [`Pool`].
    stream: Option<usize>,
}

/// A source of device buffers that reuses what it has been given back.
///
/// It exists because `Stream::recycle` needs a mutable stream, which a tensor
/// destructor does not have.
///
/// **One pool serves one stream.** A block goes back on the free list as soon
/// as its last owner drops, which is typically while the work reading it is
/// still queued. Reissuing it is safe only because the stream that runs that
/// work also runs whatever writes it next, in that order. Across streams there
/// is no such order, and the second stream would overwrite bytes the first has
/// not finished with -- events can order cross-stream *use*, but not reuse the
/// pool has already handed out. Buffers themselves are device-scoped and HRX
/// will not object, so the pool keeps this invariant itself: it records the
/// stream it first served and refuses another.
#[derive(Default)]
pub struct Pool {
    free: Mutex<FreeList>,
}

impl Pool {
    pub fn new() -> Arc<Pool> {
        Arc::new(Pool::default())
    }

    /// A buffer of at least `bytes`. A cached block is taken when it is not
    /// more than twice the size asked for, which is what keeps the free list
    /// from returning a 100 MB block for a 1 KB tensor.
    fn take(self: &Arc<Self>, stream: &Stream, bytes: usize) -> Result<Pooled> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        match free.stream {
            Some(owner) if owner != stream.id() => {
                return Err(Error(
                    "a buffer pool serves one stream: reuse is ordered by the queue that \
                     last used the block, and another stream is not in that order"
                        .into(),
                ))
            }
            Some(_) => {}
            None => free.stream = Some(stream.id()),
        }
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
        Ok(Pooled { buffer: Some(stream.allocate(bytes)?), pool: self.clone() })
    }

    fn give_back(&self, buffer: Buffer) {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.bytes() > LIMIT.saturating_sub(free.cached) {
            return; // dropping the buffer releases it
        }
        free.cached += buffer.bytes();
        free.blocks.entry(buffer.bytes()).or_default().push(buffer);
    }

    /// Raw pooled bytes, for the one intermediate that is not bf16: attention
    /// scores, which the GEMM writes as float32.
    pub fn scratch(self: &Arc<Self>, stream: &Stream, bytes: usize) -> Result<Scratch> {
        if bytes == 0 {
            return Err(Error("empty scratch".into()));
        }
        Ok(Scratch(self.take(stream, bytes)?))
    }
}

/// Pooled device bytes with no shape.
pub struct Scratch(Pooled);

impl Scratch {
    /// The whole scratch allocation, as a kernel binding.
    pub fn binding(&self) -> View<'_> {
        self.0.buffer().binding()
    }
}

/// A buffer that returns to its pool when dropped.
struct Pooled {
    buffer: Option<Buffer>,
    pool: Arc<Pool>,
}

impl Pooled {
    fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().expect("a live buffer")
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.give_back(buffer);
        }
    }
}

/// Backing allocation retained by a tensor and its views, and where in it this
/// tensor starts. Views are borrowed from the allocation rather than taken as
/// addresses, so a tensor can no longer outlive the memory it names.
enum Storage {
    Owned(Pooled),
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
    pub fn new(pool: &Arc<Pool>, stream: &Stream, rows: usize, cols: usize) -> Result<Tensor> {
        let storage = pool.take(stream, bytes(rows, cols)?)?;
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
        pool: &Arc<Pool>,
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
        let pool = Pool::new();
        let values: Vec<u16> = (0..64u16).map(|i| i.wrapping_mul(577)).collect();
        let tensor = Tensor::from_slice(&pool, &mut stream, &values, 8, 8).expect("upload");
        assert_eq!(tensor.download(&mut stream).expect("download"), values);
    }

    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX"]
    fn a_view_reads_the_rows_it_names() {
        let mut stream = Stream::open().expect("a stream");
        let pool = Pool::new();
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
        let pool = Pool::new();

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
        let pool = Pool::new();
        let first = Tensor::new(&pool, &stream, 16, 16).expect("a tensor");
        let shared = first.clone();
        drop(first);
        assert_eq!(pool.free.lock().unwrap().cached, 0, "a live alias prevents recycling");
        drop(shared);
        assert_eq!(pool.free.lock().unwrap().cached, 512);
        let second = Tensor::new(&pool, &stream, 8, 16).expect("a smaller tensor");
        assert_eq!(second.binding().unwrap().owner().bytes(), 512);
        assert_eq!(pool.free.lock().unwrap().cached, 0, "the cached block was consumed");
        drop(second);
        assert_eq!(pool.free.lock().unwrap().cached, 512);
    }
}
