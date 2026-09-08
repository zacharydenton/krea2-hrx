//! Device tensors and the pool they come from.
//!
//! Every intermediate is bf16 `[rows][cols]` in device memory. Allocations go
//! through a [`Pool`] because the pipeline makes and drops hundreds of them per
//! image and HRX allocation is not free: a dropped tensor's storage goes back
//! on a free list and the next request of a similar size takes it.
//!
//! The C++ did this with a thread-local pool and `shared_ptr` deleters. Here
//! the pool is owned by whoever owns the ops, and a tensor holds an `Arc` to
//! it, so the lifetime is in the type rather than in a convention.
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
}

impl Pool {
    pub fn new() -> Arc<Pool> {
        Arc::new(Pool::default())
    }

    /// A buffer of at least `bytes`. A cached block is taken when it is not
    /// more than twice the size asked for, which is what keeps the free list
    /// from returning a 100 MB block for a 1 KB tensor.
    fn take(self: &Arc<Self>, bytes: usize) -> Result<Pooled> {
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
        if free.cached + buffer.len() > LIMIT {
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

/// Where a tensor's elements live: its own pooled block, or someone else's.
enum Storage {
    Owned(Pooled),
    Borrowed(DevicePtr),
}

impl Storage {
    fn ptr(&self) -> DevicePtr {
        match self {
            Storage::Owned(pooled) => pooled.ptr(),
            Storage::Borrowed(pointer) => *pointer,
        }
    }
}

/// A bf16 matrix on the device. Cloning shares the storage; [`Tensor::view`]
/// makes a window onto it without copying.
#[derive(Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    storage: Arc<Storage>,
    offset: usize,
}

impl Tensor {
    /// An uninitialized tensor. Kernels write every element they read, as they
    /// did in the C++; a tensor that must start at zero is zeroed explicitly.
    pub fn new(pool: &Arc<Pool>, rows: usize, cols: usize) -> Result<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(Error("empty tensor".into()));
        }
        let storage = pool.take(rows * cols * 2)?;
        Ok(Tensor { rows, cols, storage: Arc::new(Storage::Owned(storage)), offset: 0 })
    }

    /// A matrix over memory this tensor does not own — a weight, read by an
    /// operation that takes tensors. The caller keeps the memory alive; a
    /// device address is never dereferenced here, so this stays safe code, in
    /// the same way the C++ `Tensor::view` over a weight was.
    pub fn borrowed(values: DevicePtr, rows: usize, cols: usize) -> Result<Tensor> {
        if rows == 0 || cols == 0 {
            return Err(Error("empty tensor".into()));
        }
        Ok(Tensor { rows, cols, storage: Arc::new(Storage::Borrowed(values)), offset: 0 })
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
        self.rows * self.cols
    }

    pub fn ptr(&self) -> DevicePtr {
        self.storage.ptr().offset(self.offset)
    }

    /// A window of `rows * cols` elements, `elements` into this tensor.
    pub fn view(&self, rows: usize, cols: usize, elements: usize) -> Result<Tensor> {
        if rows == 0 || cols == 0 || elements + rows * cols > self.size() {
            return Err(Error("tensor view".into()));
        }
        Ok(Tensor {
            rows,
            cols,
            storage: self.storage.clone(),
            offset: self.offset + elements * 2,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn usable() -> bool {
        hrx::try_device().is_ok()
    }

    #[test]
    fn a_tensor_round_trips_through_the_device() {
        if !usable() {
            return;
        }
        let pool = Pool::new();
        let values: Vec<u16> = (0..64u16).map(|i| i.wrapping_mul(577)).collect();
        let tensor = Tensor::from_slice(&pool, &values, 8, 8).expect("upload");
        assert_eq!(tensor.download().expect("download"), values);
    }

    #[test]
    fn a_view_reads_the_rows_it_names() {
        if !usable() {
            return;
        }
        let pool = Pool::new();
        let values: Vec<u16> = (0..64u16).collect();
        let tensor = Tensor::from_slice(&pool, &values, 8, 8).expect("upload");
        let second = tensor.view(1, 8, 8).expect("the second row");
        assert_eq!(second.download().expect("download"), &values[8..16]);
        assert!(tensor.view(2, 8, 56).is_err(), "a view past the end is refused");
    }

    #[test]
    fn dropped_storage_comes_back_from_the_pool() {
        if !usable() {
            return;
        }
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
