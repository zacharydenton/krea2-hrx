//! Loom's AMDGPU kernel argument ABI: scalars at their natural alignment in
//! declaration order, then one 8-byte device address per buffer operand.
//!
//! This is the only implementation in the tree; the C++ host had three, with
//! different capacities and one that left its padding uninitialized. Padding
//! here is always zero, which is what the largest of the three did.
use crate::DevicePtr;

/// The dispatch limit is 256 bytes of direct arguments.
const CAPACITY: usize = 256;

#[derive(Clone, Copy)]
pub struct Args {
    bytes: [u8; CAPACITY],
    size: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

impl Args {
    pub const fn new() -> Self {
        Args { bytes: [0; CAPACITY], size: 0 }
    }

    fn push(&mut self, value: &[u8], align: usize) -> &mut Self {
        self.size = (self.size + align - 1) & !(align - 1);
        assert!(self.size + value.len() <= CAPACITY, "kernel argument overflow");
        self.bytes[self.size..self.size + value.len()].copy_from_slice(value);
        self.size += value.len();
        self
    }

    pub fn i32(&mut self, value: i32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.push(&value.to_ne_bytes(), 4)
    }

    /// A device address, as the kernel's `buffer` operand.
    pub fn ptr(&mut self, value: DevicePtr) -> &mut Self {
        self.push(&(value.address() as u64).to_ne_bytes(), 8)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_pad_to_their_alignment_and_pointers_to_eight() {
        let mut args = Args::new();
        args.i32(1).ptr(DevicePtr::from_address(0x1000)).f32(2.0);
        // i32 at 0, four bytes of zero padding, the address at 8, the float at 16.
        let bytes = args.as_bytes();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[0..4], &1i32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "padding must be zero, not indeterminate");
        assert_eq!(&bytes[8..16], &0x1000u64.to_ne_bytes());
        assert_eq!(&bytes[16..20], &2.0f32.to_ne_bytes());
    }

    #[test]
    fn an_empty_argument_list_is_empty() {
        assert!(Args::new().as_bytes().is_empty());
    }

    #[test]
    #[should_panic(expected = "kernel argument overflow")]
    fn past_the_dispatch_limit_is_a_bug_not_an_error() {
        let mut args = Args::new();
        for _ in 0..33 {
            args.ptr(DevicePtr::from_address(8));
        }
    }
}
