//! Dense checkpoint tensors in device memory.
//!
//! BF16 values are uploaded directly; F32 values retain a float32 copy and gain a
//! bf16 copy. Scaled fp8 values are dequantized to bf16. Single-image causal
//! convolutions use the last temporal tap. Packed 3×3 weights use `[out, ky, kx, in]`
//! values with logical `[out, in, ky, kx]` shapes; the layout travels with the weight.
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::checkpoint::Checkpoint;
use crate::numerics::{fp8_e4m3_to_f32, from_f32_carrying};
use crate::ops::Weight;
use hrx::{Buffer, Stream};

use super::{Error, Result};

/// A checkpoint's tensors, renamed onto the names this runtime uses.
pub struct Weights {
    values: BTreeMap<String, Weight>,
    /// Every weight holds a share of whichever of these its values live in, so
    /// this is only here to keep the set together.
    _storage: Vec<Arc<Buffer>>,
}

struct Item {
    name: String,
    key: String,
    dtype: String,
    file_shape: Vec<usize>,
    shape: Vec<usize>,
    count: usize,
    bytes: usize,
    offset: usize,
    last_tap: bool,
    scale: Option<f32>,
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
            let file_shape = tensor.shape.to_vec();
            // [O][I][T][H][W]: a causal 3-D convolution, kept as its last tap.
            let last_tap = file_shape.len() == 5;
            let shape = if last_tap {
                vec![file_shape[0], file_shape[1], file_shape[3], file_shape[4]]
            } else {
                file_shape.clone()
            };
            let count: usize = shape.iter().product();
            let (bytes, scale) = match tensor.dtype {
                "F32" => (count * 4, None),
                "BF16" => (count * 2, None),
                "F8_E4M3" => {
                    let scale = file.get(&format!("{key}_scale"))?;
                    if scale.dtype != "F32" || scale.bytes.len() != 4 {
                        return Err(Error(format!("unsupported float8 scale for {key}")));
                    }
                    (
                        count * 2,
                        Some(f32::from_le_bytes(scale.bytes.try_into().expect("four bytes"))),
                    )
                }
                other => {
                    return Err(Error(format!(
                        "unsupported tensor dtype {other} for {key} in {}",
                        file.path().display()
                    )))
                }
            };
            let element_bytes = match tensor.dtype {
                "F32" => 4,
                "BF16" => 2,
                _ => 1,
            };
            let elements: usize = file_shape.iter().product();
            if tensor.bytes.len() != elements * element_bytes {
                return Err(Error(format!("tensor size mismatch for {key}")));
            }
            items.push(Item {
                name,
                key: key.to_string(),
                dtype: tensor.dtype.to_string(),
                file_shape,
                shape,
                count,
                bytes,
                offset: total,
                last_tap,
                scale,
            });
            total += bytes.div_ceil(256) * 256;
        }
        if items.is_empty() {
            return Err(Error("the checkpoint has none of the tensors this needs".into()));
        }

        let storage = Arc::new(stream.allocate(total)?);
        let mut float_storage = Vec::new();
        let mut values = BTreeMap::new();
        for item in &items {
            let tensor = file.get(&item.key)?;
            let base = item.offset;
            let staged = stage(&item.dtype, tensor.bytes, item)?;
            let bytes = staged.as_deref().unwrap_or(&tensor.bytes[..item.bytes]);
            // Repack both ordinary 4D and reduced causal 5D convolutions.
            let element = if item.dtype == "F32" { 4 } else { 2 };
            let packed = (is_square_convolution(&item.shape, 3)
                && crate::ops::pack_convolutions())
            .then(|| channels_last(bytes, &item.shape, element));
            let layout = match packed.is_some() {
                true => crate::ops::Layout::ChannelsLast,
                false => crate::ops::Layout::RowMajor,
            };
            // The F32 and BF16 copies must use the same packed layout.
            let bytes = packed.as_deref().unwrap_or(bytes);
            for (i, chunk) in bytes.chunks(16 << 20).enumerate() {
                stream.upload(storage.try_slice(base + i * (16 << 20), chunk.len())?, chunk)?;
            }
            let (bf16, holder) = if item.dtype == "F32" {
                // Everything but the norms consumes a float32 tensor as bf16,
                // rounded the way the checkpoint's own conversion rounds.
                let floats = read_f32(bytes, item.count);
                let rounded: Vec<u16> = floats.iter().map(|&v| from_f32_carrying(v)).collect();
                let buffer = Arc::new(stream.allocate(rounded.len() * 2)?);
                for (i, chunk) in
                    bytemuck::cast_slice::<u16, u8>(&rounded).chunks(16 << 20).enumerate()
                {
                    stream.upload(buffer.try_slice(i * (16 << 20), chunk.len())?, chunk)?;
                }
                float_storage.push(Arc::clone(&buffer));
                (0, buffer)
            } else {
                (base, Arc::clone(&storage))
            };
            values.insert(
                item.name.clone(),
                Weight::new(
                    &holder,
                    bf16,
                    item.shape.clone(),
                    item.count,
                    (item.dtype == "F32").then(|| (Arc::clone(&storage), item.offset)),
                )
                .in_layout(layout),
            );
        }
        float_storage.push(storage);
        Ok(Weights { values, _storage: float_storage })
    }

    pub fn get(&self, name: &str) -> Result<&Weight> {
        self.values.get(name).ok_or_else(|| Error(format!("missing tensor {name}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
}

/// The bytes to upload when the file's are not already what the device wants:
/// a float8 row to dequantise, or a convolution to reduce to its last tap.
fn stage(dtype: &str, source: &[u8], item: &Item) -> Result<Option<Vec<u8>>> {
    if !item.last_tap && item.scale.is_none() {
        return Ok(None);
    }
    let (taps, plane, blocks) = if item.last_tap {
        (
            item.file_shape[2],
            item.file_shape[3] * item.file_shape[4],
            item.file_shape[0] * item.file_shape[1],
        )
    } else {
        (1, item.count, 1)
    };
    let element = if item.scale.is_some() {
        1
    } else if dtype == "F32" {
        4
    } else {
        2
    };
    let out_element = if dtype == "F32" { 4 } else { 2 };
    let mut staged = vec![0u8; item.bytes];
    for block in 0..blocks {
        let start = ((block * taps) + (taps - 1)) * plane * element;
        let input = &source[start..start + plane * element];
        let out = &mut staged[block * plane * out_element..(block + 1) * plane * out_element];
        match item.scale {
            Some(scale) => {
                for (index, &byte) in input.iter().enumerate() {
                    let value = from_f32_carrying(fp8_e4m3_to_f32(byte) * scale);
                    out[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
            None => out.copy_from_slice(input),
        }
    }
    Ok(Some(staged))
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
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}
