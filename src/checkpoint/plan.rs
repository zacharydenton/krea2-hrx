//! Where every checkpoint row goes on the device.
//!
//! Per block the int8 ConvRot rows of wq, wk, wv and the attention gate become
//! one `qkvg` operand; the MLP gate and up rows are interleaved in 16-row
//! groups so the fused SwiGLU epilogue reads them together; wo and down stay as
//! they are. Each operand's rows are written at the pitch its kernel was
//! compiled for, the per-row f32 scales follow the same arrangement, and the
//! RMSNorm scales are the checkpoint's own f32 vectors.
use std::collections::BTreeMap;
use std::ops::Range;

use super::{Checkpoint, Error, Result};

/// Rows to copy from one checkpoint tensor, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The tensor these rows come from.
    pub tensor: String,
    /// Which of its rows, as a half-open range.
    pub rows: Range<usize>,
}

/// One destination on the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// Where it starts in the single weights allocation.
    pub device_offset: usize,
    /// How much of the allocation it occupies, at the device pitch.
    pub device_bytes: usize,
    /// Rows, and the bytes each occupies in the file and on the device. A
    /// device row is at least as wide as a file row; the padding stays zero.
    pub rows: usize,
    pub row_bytes: usize,
    pub device_row_bytes: usize,
    /// The checkpoint rows that fill it, in order.
    pub segments: Vec<Segment>,
    /// Bytes assembled on the host instead: the gathered f32 scale vectors.
    pub host: Vec<u8>,
}

impl Span {
    /// Bytes as the checkpoint stores them, before any padding.
    pub fn file_bytes(&self) -> usize {
        if self.host.is_empty() {
            self.rows * self.row_bytes
        } else {
            self.host.len()
        }
    }
}

/// The whole upload: named spans, and the size of the allocation they need.
#[derive(Debug, Default)]
pub struct Plan {
    pub spans: BTreeMap<String, Span>,
    pub total_bytes: usize,
    pub layers: usize,
    /// The operand width the GEMM kernels must be built for.
    pub bits: u32,
}

/// Operand row pitch in k elements, from `crate::kernels::shape`, in bytes for int8.
fn device_row_bytes(row_bytes: usize, bits: u32) -> usize {
    crate::kernels::shape::gemm_pitch(row_bytes as i32 * 8 / bits as i32, bits as i32) as usize
        * bits as usize
        / 8
}

fn int8_rows<'a>(file: &'a Checkpoint, name: &str) -> Result<super::Tensor<'a>> {
    let tensor = file.get(&format!("{name}.weight"))?;
    if tensor.dtype != "I8" || tensor.shape.len() != 2 {
        return Err(Error(format!(
            "{name}.weight is {}, not int8 ConvRot rows ({})",
            tensor.dtype,
            file.path().display()
        )));
    }
    if tensor.shape[0] == 0 || tensor.shape[1] == 0 || tensor.row_bytes()? != tensor.shape[1] {
        return Err(Error(format!("invalid weight shape or size in {name}")));
    }
    Ok(tensor)
}

fn scales<'a>(file: &'a Checkpoint, name: &str, rows: usize) -> Result<super::Tensor<'a>> {
    let tensor = file.get(&format!("{name}.weight_scale"))?;
    if tensor.dtype != "F32" {
        return Err(Error(format!("{name}.weight_scale is not float32")));
    }
    if tensor.bytes.len() != rows * 4 {
        return Err(Error(format!("scale count in {name}")));
    }
    Ok(tensor)
}

impl Plan {
    /// Works out the device layout for one checkpoint.
    pub fn for_checkpoint(file: &Checkpoint) -> Result<Plan> {
        let layers = file.block_count();
        if layers == 0 {
            return Err(Error(format!("no transformer blocks in {}", file.path().display())));
        }
        let bits = 8;
        let mut spans: BTreeMap<String, Span> = BTreeMap::new();
        for block in 0..layers {
            let p = format!("blocks.{block}");
            let (q, s) = operand(
                file,
                &format!("{p}.qkvg"),
                &[
                    format!("{p}.attn.wq"),
                    format!("{p}.attn.wk"),
                    format!("{p}.attn.wv"),
                    format!("{p}.attn.gate"),
                ],
                0,
                bits,
            )?;
            spans.insert(format!("{p}.qkvg.q"), q);
            spans.insert(format!("{p}.qkvg.s"), s);
            for (out, part) in
                [("wo", format!("{p}.attn.wo")), ("down", format!("{p}.mlp.down"))]
            {
                let (q, s) = operand(file, &format!("{p}.{out}"), &[part], 0, bits)?;
                spans.insert(format!("{p}.{out}.q"), q);
                spans.insert(format!("{p}.{out}.s"), s);
            }
            let (q, s) = operand(
                file,
                &format!("{p}.gu"),
                &[format!("{p}.mlp.gate"), format!("{p}.mlp.up")],
                16,
                bits,
            )?;
            spans.insert(format!("{p}.gu.q"), q);
            spans.insert(format!("{p}.gu.s"), s);
            for (out, name) in [
                ("prenorm", format!("{p}.prenorm.scale")),
                ("postnorm", format!("{p}.postnorm.scale")),
                ("qnorm", format!("{p}.attn.qknorm.qnorm.scale")),
                ("knorm", format!("{p}.attn.qknorm.knorm.scale")),
            ] {
                spans.insert(format!("{p}.{out}"), vector(file, &name)?);
            }
        }
        // Tensors start at 256-byte boundaries in name order; sessions use these offsets.
        let mut total = 0;
        for span in spans.values_mut() {
            span.device_offset = total;
            total += span.device_bytes.div_ceil(256) * 256;
        }
        Ok(Plan { spans, total_bytes: total, layers, bits })
    }

    pub fn span(&self, name: &str) -> Result<&Span> {
        self.spans.get(name).ok_or_else(|| Error(format!("no span named {name}")))
    }
}

/// One GEMM operand and its scale vector.
///
/// `group` 0 concatenates the parts' rows; `group` n interleaves them n rows at
/// a time, which is what the fused gate/up epilogue reads.
fn operand(
    file: &Checkpoint,
    out: &str,
    parts: &[String],
    group: usize,
    bits: u32,
) -> Result<(Span, Span)> {
    let tensors: Vec<_> =
        parts.iter().map(|part| int8_rows(file, part)).collect::<Result<_>>()?;
    let row_bytes = tensors[0].row_bytes()?;
    if tensors.iter().any(|t| t.shape[1] != tensors[0].shape[1]) {
        return Err(Error(format!("mismatched K in {out}")));
    }
    let rows: usize = tensors.iter().map(|t| t.shape[0]).sum();
    let mut weights = Span {
        device_offset: 0,
        device_bytes: rows * device_row_bytes(row_bytes, bits),
        rows,
        row_bytes,
        device_row_bytes: device_row_bytes(row_bytes, bits),
        segments: Vec::new(),
        host: Vec::new(),
    };
    let mut scale_bytes = vec![0u8; rows * 4];
    let mut written = 0;
    if group == 0 {
        for (part, tensor) in parts.iter().zip(&tensors) {
            let count = tensor.shape[0];
            weights.segments.push(Segment { tensor: format!("{part}.weight"), rows: 0..count });
            let source = scales(file, part, count)?;
            scale_bytes[written * 4..(written + count) * 4].copy_from_slice(source.bytes);
            written += count;
        }
    } else {
        if tensors.len() != 2
            || tensors[0].shape[0] != tensors[1].shape[0]
            || tensors[0].shape[0] % group != 0
        {
            return Err(Error(format!("interleave shape in {out}")));
        }
        let sources: Vec<_> = parts
            .iter()
            .map(|part| scales(file, part, tensors[0].shape[0]))
            .collect::<Result<_>>()?;
        for start in (0..tensors[0].shape[0]).step_by(group) {
            for (part, source) in parts.iter().zip(&sources) {
                weights.segments.push(Segment {
                    tensor: format!("{part}.weight"),
                    rows: start..start + group,
                });
                scale_bytes[written * 4..(written + group) * 4]
                    .copy_from_slice(&source.bytes[start * 4..(start + group) * 4]);
                written += group;
            }
        }
    }
    let scales = Span {
        device_offset: 0,
        device_bytes: scale_bytes.len(),
        rows: 0,
        row_bytes: 0,
        device_row_bytes: 0,
        segments: Vec::new(),
        host: scale_bytes,
    };
    Ok((weights, scales))
}

/// An f32 vector: the RMSNorm scales, copied as they are, or upcast exactly from
/// the bf16 a half-precision save stores them in.
fn vector(file: &Checkpoint, name: &str) -> Result<Span> {
    let tensor = file.get(name)?;
    if tensor.dtype == "BF16" {
        let host: Vec<u8> = tensor
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|bits| crate::numerics::to_f32(u16::from_le_bytes(*bits)).to_le_bytes())
            .collect();
        return Ok(Span {
            device_offset: 0,
            device_bytes: host.len(),
            rows: 0,
            row_bytes: 0,
            device_row_bytes: 0,
            segments: Vec::new(),
            host,
        });
    }
    if tensor.dtype != "F32" {
        return Err(Error(format!("{name} is {}, not float32", tensor.dtype)));
    }
    Ok(Span {
        device_offset: 0,
        device_bytes: tensor.bytes.len(),
        rows: 1,
        row_bytes: tensor.bytes.len(),
        device_row_bytes: tensor.bytes.len(),
        segments: vec![Segment { tensor: name.to_string(), rows: 0..1 }],
        host: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> Checkpoint {
        let path = std::env::var_os("KREA2_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                crate::models::hub::file(
                    crate::models::hub::REPO,
                    "diffusion_models/krea2_turbo_int8_convrot.safetensors",
                    true,
                )
                .expect("cache the Turbo checkpoint or set KREA2_MODEL")
            });
        Checkpoint::open(&path).expect("set KREA2_MODEL to a local checkpoint")
    }

    #[test]
    #[ignore = "requires a local Krea checkpoint"]
    fn the_plan_lays_out_krea_twos_blocks() {
        let file = checkpoint();
        let plan = Plan::for_checkpoint(&file).expect("a plan");
        assert_eq!(plan.layers, 28);
        assert_eq!(plan.bits, 8);

        // qkv|gate: 48 query heads, 12 key and value heads, and the gate, all
        // of width 128, over the 6144-wide residual stream.
        let qkvg = plan.span("blocks.0.qkvg.q").expect("qkvg");
        assert_eq!(qkvg.rows, 48 * 128 + 12 * 128 + 12 * 128 + 6144);
        assert_eq!(qkvg.row_bytes, 6144);
        assert_eq!(qkvg.device_row_bytes, 6144, "K = 6144 rows stay dense");
        assert_eq!(qkvg.segments.len(), 4);

        // The down projection's rows are 16384 bytes, a multiple of 8192, so
        // they take one 64-byte step of padding to stop cache aliasing.
        let down = plan.span("blocks.0.down.q").expect("down");
        assert_eq!((down.rows, down.row_bytes), (6144, 16384));
        assert_eq!(down.device_row_bytes, 16448);
        assert_eq!(down.device_bytes, 6144 * 16448);

        // Every span starts at a 256-byte boundary and none overlap.
        let mut spans: Vec<_> = plan.spans.values().collect();
        spans.sort_by_key(|span| span.device_offset);
        let mut end = 0;
        for span in spans {
            assert_eq!(span.device_offset % 256, 0);
            assert!(span.device_offset >= end, "spans overlap");
            end = span.device_offset + span.device_bytes;
        }
        assert!(plan.total_bytes >= end);
    }

    #[test]
    #[ignore = "requires a local Krea checkpoint"]
    fn the_gate_and_up_rows_interleave_in_sixteens() {
        let file = checkpoint();
        let plan = Plan::for_checkpoint(&file).expect("a plan");
        let gu = plan.span("blocks.0.gu.q").expect("gu");
        let half = gu.rows / 2;
        assert_eq!(gu.segments.len(), half / 16 * 2);
        assert_eq!(
            gu.segments[0],
            Segment { tensor: "blocks.0.mlp.gate.weight".into(), rows: 0..16 }
        );
        assert_eq!(
            gu.segments[1],
            Segment { tensor: "blocks.0.mlp.up.weight".into(), rows: 0..16 }
        );
        assert_eq!(
            gu.segments[2],
            Segment { tensor: "blocks.0.mlp.gate.weight".into(), rows: 16..32 }
        );
        // The scales follow the rows: gate 0..16, up 0..16, gate 16..32, ...
        let scales = plan.span("blocks.0.gu.s").expect("gu scales");
        assert_eq!(scales.host.len(), gu.rows * 4);
        let gate = file.get("blocks.0.mlp.gate.weight_scale").expect("gate scales");
        let up = file.get("blocks.0.mlp.up.weight_scale").expect("up scales");
        assert_eq!(&scales.host[..64], &gate.bytes[..64]);
        assert_eq!(&scales.host[64..128], &up.bytes[..64]);
        assert_eq!(&scales.host[128..192], &gate.bytes[64..128]);
    }

    #[test]
    #[ignore = "requires a local Krea checkpoint"]
    fn the_concatenated_scales_follow_their_rows() {
        let file = checkpoint();
        let plan = Plan::for_checkpoint(&file).expect("a plan");
        let scales = plan.span("blocks.0.qkvg.s").expect("qkvg scales");
        let mut at = 0;
        for part in ["attn.wq", "attn.wk", "attn.wv", "attn.gate"] {
            let source =
                file.get(&format!("blocks.0.{part}.weight_scale")).expect("a scale vector");
            assert_eq!(&scales.host[at..at + source.bytes.len()], source.bytes, "{part}");
            at += source.bytes.len();
        }
        assert_eq!(at, scales.host.len());
    }
}
