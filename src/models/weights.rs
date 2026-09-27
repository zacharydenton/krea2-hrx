//! Dense checkpoint tensors in device memory.
//!
//! BF16 values are uploaded directly; F32 values retain a float32 copy and gain a
//! bf16 copy. Scaled fp8 values are dequantized to bf16. Single-image causal
//! convolutions use the last temporal tap. Packed 3×3 weights use `[out, ky, kx, in]`
//! values with logical `[out, in, ky, kx]` shapes; the layout travels with the weight.
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::checkpoint::{Checkpoint, DType, Tensor};
use crate::numerics::{fp8_e4m3_to_f32, from_f32_carrying};
use crate::ops::Weight;
use hrx::{Buffer, Stream};

use super::{Error, Result};

/// A checkpoint's tensors, renamed onto the names this runtime uses. Each
/// weight holds a share of the allocation its values live in.
pub struct Weights {
    values: BTreeMap<String, Weight>,
}

/// How a tensor is stored in the file, which decides how it is staged.
#[derive(Clone, Copy, PartialEq)]
enum Stored {
    /// Kept as float32 for the norms, and rounded to a bf16 copy.
    F32,
    Bf16,
    /// fp8 E4M3 with one float32 scale, dequantized to bf16.
    Fp8 {
        scale: f32,
    },
}

impl Stored {
    /// Bytes per element in the file.
    fn file_bytes(self) -> usize {
        match self {
            Stored::F32 => 4,
            Stored::Bf16 => 2,
            Stored::Fp8 { .. } => 1,
        }
    }

    /// Bytes per element on the device, in the main allocation.
    fn device_bytes(self) -> usize {
        match self {
            Stored::F32 => 4,
            Stored::Bf16 | Stored::Fp8 { .. } => 2,
        }
    }
}

/// One tensor to upload, and where it goes.
struct Item<'a> {
    name: String,
    tensor: Tensor<'a>,
    stored: Stored,
    /// The device shape: a causal 3-D convolution loses its temporal axis.
    shape: Vec<usize>,
    count: usize,
    offset: usize,
}

impl Item<'_> {
    /// `[O][I][T][H][W]`: a causal 3-D convolution, kept as its last tap.
    fn last_tap(&self) -> bool {
        self.tensor.shape.len() == 5
    }

    fn device_bytes(&self) -> usize {
        self.count * self.stored.device_bytes()
    }
}

impl Weights {
    /// Loads every tensor `rename` maps to a non-empty name.
    pub fn load(
        stream: &mut Stream,
        file: &Checkpoint,
        rename: impl Fn(&str) -> String,
    ) -> Result<Weights> {
        let mut items = Vec::new();
        let mut total = 0;
        for key in file.names() {
            if key.ends_with(".comfy_quant") || key.ends_with(".weight_scale") {
                continue;
            }
            let name = rename(key);
            if name.is_empty() {
                continue;
            }
            let tensor = file.get(key)?;
            let shape = match *tensor.shape {
                [outputs, inputs, _, height, width] => vec![outputs, inputs, height, width],
                ref shape => shape.to_vec(),
            };
            let count: usize = shape.iter().product();
            let stored = match tensor.dtype {
                DType::F32 => Stored::F32,
                DType::BF16 => Stored::Bf16,
                DType::F8_E4M3 => {
                    let scale = file.get(&format!("{key}_scale"))?;
                    match (scale.dtype, <[u8; 4]>::try_from(scale.bytes)) {
                        (DType::F32, Ok(bytes)) => {
                            Stored::Fp8 { scale: f32::from_le_bytes(bytes) }
                        }
                        _ => {
                            return Err(Error::invalid(format!(
                                "unsupported float8 scale for {key}"
                            )));
                        }
                    }
                }
                other => {
                    return Err(Error::invalid(format!(
                        "unsupported tensor dtype {other:?} for {key} in {}",
                        file.path().display()
                    )));
                }
            };
            let elements: usize = tensor.shape.iter().product();
            if tensor.bytes.len() != elements * stored.file_bytes() {
                return Err(Error::invalid(format!("tensor size mismatch for {key}")));
            }
            let item = Item { name, tensor, stored, shape, count, offset: total };
            total += item.device_bytes().div_ceil(256) * 256;
            items.push(item);
        }
        if items.is_empty() {
            return Err(Error::invalid("the checkpoint has none of the tensors this needs"));
        }

        let storage = Arc::new(stream.allocate(total)?);
        let mut values = BTreeMap::new();
        for item in &items {
            let staged = stage(item);
            let bytes = staged.as_deref().unwrap_or(&item.tensor.bytes[..item.device_bytes()]);
            // Repack both ordinary 4D and reduced causal 5D convolutions.
            let element = item.stored.device_bytes();
            let packed = is_square_convolution(&item.shape, 3)
                .then(|| channels_last(bytes, &item.shape, element));
            let layout = if packed.is_some() {
                crate::ops::Layout::ChannelsLast
            } else {
                crate::ops::Layout::RowMajor
            };
            // The F32 and BF16 copies must use the same packed layout.
            let bytes = packed.as_deref().unwrap_or(bytes);
            upload(stream, &storage, item.offset, bytes)?;
            let weight = if item.stored == Stored::F32 {
                // Everything but the norms consumes a float32 tensor as bf16,
                // rounded the way the checkpoint's own conversion rounds.
                let rounded: Vec<u16> =
                    read_f32(bytes, item.count).into_iter().map(from_f32_carrying).collect();
                let buffer = Arc::new(stream.allocate(rounded.len() * 2)?);
                upload(stream, &buffer, 0, bytemuck::cast_slice(&rounded))?;
                let float32 = Some((Arc::clone(&storage), item.offset));
                Weight::new(&buffer, 0, item.shape.clone(), item.count, float32)
            } else {
                Weight::new(&storage, item.offset, item.shape.clone(), item.count, None)
            };
            values.insert(item.name.clone(), weight.in_layout(layout));
        }
        Ok(Weights { values })
    }

    /// The tensor named `name`, or an invalid-argument error.
    pub fn get(&self, name: &str) -> Result<&Weight> {
        self.values.get(name).ok_or_else(|| Error::invalid(format!("missing tensor {name}")))
    }

    /// Whether the checkpoint has a tensor named `name`.
    pub fn has(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// The tensor named `name`, when the checkpoint has one.
    pub fn find(&self, name: &str) -> Option<&Weight> {
        self.values.get(name)
    }
}

/// `bytes` into `buffer` at `offset`, in staging-sized chunks.
fn upload(stream: &mut Stream, buffer: &Buffer, offset: usize, bytes: &[u8]) -> Result<()> {
    const CHUNK: usize = 16 << 20;
    for (index, chunk) in bytes.chunks(CHUNK).enumerate() {
        stream.upload(buffer.try_slice(offset + index * CHUNK, chunk.len())?, chunk)?;
    }
    Ok(())
}

/// The bytes to upload when the file's are not already what the device wants:
/// a float8 tensor to dequantise, or a convolution to reduce to its last tap.
fn stage(item: &Item) -> Option<Vec<u8>> {
    let scale = match item.stored {
        Stored::Fp8 { scale } => Some(scale),
        _ if item.last_tap() => None,
        _ => return None,
    };
    let (taps, plane, blocks) = match *item.tensor.shape {
        [outputs, inputs, taps, height, width] => (taps, height * width, outputs * inputs),
        _ => (1, item.count, 1),
    };
    let (element, out_element) = (item.stored.file_bytes(), item.stored.device_bytes());
    let mut staged = vec![0u8; item.device_bytes()];
    for block in 0..blocks {
        let start = ((block * taps) + (taps - 1)) * plane * element;
        let input = &item.tensor.bytes[start..start + plane * element];
        let out = &mut staged[block * plane * out_element..(block + 1) * plane * out_element];
        match scale {
            Some(scale) => {
                for (&byte, value) in input.iter().zip(out.as_chunks_mut::<2>().0) {
                    *value = from_f32_carrying(fp8_e4m3_to_f32(byte) * scale).to_le_bytes();
                }
            }
            None => out.copy_from_slice(input),
        }
    }
    Some(staged)
}

/// `[out][in][k][k]` with `k` as given.
fn is_square_convolution(shape: &[usize], k: usize) -> bool {
    shape.len() == 4 && shape[2] == k && shape[3] == k
}

/// `[out][in][ky][kx]` to `[out][ky][kx][in]`, so the innermost run is the
/// input channels the convolution reduces over four at a time.
fn channels_last(values: &[u8], shape: &[usize], element: usize) -> Vec<u8> {
    let (outputs, inputs, ky, kx) = (shape[0], shape[1], shape[2], shape[3]);
    let taps = ky * kx;
    let mut packed = vec![0u8; values.len()];
    for o in 0..outputs {
        for i in 0..inputs {
            for t in 0..taps {
                let from = ((o * inputs + i) * taps + t) * element;
                let to = ((o * taps + t) * inputs + i) * element;
                packed[to..to + element].copy_from_slice(&values[from..from + element]);
            }
        }
    }
    packed
}

fn read_f32(bytes: &[u8], count: usize) -> Vec<f32> {
    bytes[..count * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}
