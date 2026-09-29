//! Single-frame Qwen VAE encoding; temporal downsamplers are inactive for frame zero.
use crate::checkpoint::Checkpoint;
use crate::kernels::Scalars;
use crate::models::Weights;
use crate::ops::{Binary, Norm, Ops, Tensor, config};
use crate::{Error, Result};
use hrx::Stream;
use std::path::Path;

/// Frozen VAE encoder loaded separately from text conditioning and the DiT.
pub struct Encoder {
    weights: Weights,
}

impl Encoder {
    /// Load encoder and posterior-projection weights from the native Qwen VAE file.
    pub fn load(stream: &mut Stream, path: &Path) -> Result<Self> {
        let weights = Weights::load(stream, &Checkpoint::open(path)?, |name| {
            if (name.starts_with("encoder.") || name.starts_with("conv1."))
                && !name.contains("time_conv")
            {
                name.into()
            } else {
                String::new()
            }
        })?;
        weights.get("encoder.conv1.weight")?;
        weights.get("conv1.weight")?;
        Ok(Self { weights })
    }

    fn conv(
        &self,
        ops: &Ops,
        stream: &Stream,
        x: &Tensor,
        h: usize,
        w: usize,
        name: &str,
    ) -> Result<Tensor> {
        ops.conv(
            stream,
            x,
            h,
            w,
            self.weights.get(&format!("{name}.weight"))?,
            Some(self.weights.get(&format!("{name}.bias"))?.values()?),
        )
    }

    fn residual(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        x: &Tensor,
        h: usize,
        w: usize,
        name: &str,
    ) -> Result<Tensor> {
        let skip = if self.weights.has(&format!("{name}.shortcut.weight")) {
            self.conv(ops, stream, x, h, w, &format!("{name}.shortcut"))?
        } else {
            x.clone()
        };
        let y =
            ops.norm_silu(stream, x, self.weights.get(&format!("{name}.residual.0.gamma"))?)?;
        let y = self.conv(ops, stream, &y, h, w, &format!("{name}.residual.2"))?;
        let y =
            ops.norm_silu(stream, &y, self.weights.get(&format!("{name}.residual.3.gamma"))?)?;
        let y = self.conv(ops, stream, &y, h, w, &format!("{name}.residual.6"))?;
        ops.binary(stream, &y, &skip, Binary::Add)
    }

    /// Return unnormalized `[latent_height * latent_width, 32]` posterior moments.
    /// The first 16 channels are means and the last 16 are log variances.
    pub fn encode(
        &self,
        ops: &Ops,
        stream: &mut Stream,
        image: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        if image.rows() != width * height
            || image.cols() != 3
            || !width.is_multiple_of(16)
            || !height.is_multiple_of(16)
        {
            return Err(Error::invalid("VAE encoder image dimensions"));
        }
        let (mut h, mut w) = (height, width);
        let mut x = self.conv(ops, stream, image, h, w, "encoder.conv1")?;
        for index in 0..11 {
            let name = format!("encoder.downsamples.{index}");
            if [2, 5, 8].contains(&index) {
                // Same-padding convolution sampled at odd coordinates equals
                // bottom/right zero padding followed by a stride-two convolution.
                let full = self.conv(ops, stream, &x, h, w, &format!("{name}.resample.1"))?;
                x = stride_two(ops, stream, &full, h, w)?;
                h /= 2;
                w /= 2;
            } else {
                x = self.residual(ops, stream, &x, h, w, &name)?;
            }
        }
        x = self.residual(ops, stream, &x, h, w, "encoder.middle.0")?;
        let normed = ops.norm(
            stream,
            &x,
            self.weights.get("encoder.middle.1.norm.gamma")?,
            Norm::Group,
            1e-5,
        )?;
        let qkv = self.conv(ops, stream, &normed, h, w, "encoder.middle.1.to_qkv")?;
        let dim = x.cols();
        let q = columns(ops, stream, &qkv, 0, dim)?;
        let k = columns(ops, stream, &qkv, dim, dim)?;
        let v = columns(ops, stream, &qkv, dim * 2, dim)?;
        let attended = ops.attention(stream, &q, &k, &v, 1, h * w, 1, 1, dim, false)?;
        let projected = self.conv(ops, stream, &attended, h, w, "encoder.middle.1.proj")?;
        x = ops.binary(stream, &x, &projected, Binary::Add)?;
        x = self.residual(ops, stream, &x, h, w, "encoder.middle.2")?;
        x = ops.norm_silu(stream, &x, self.weights.get("encoder.head.0.gamma")?)?;
        x = self.conv(ops, stream, &x, h, w, "encoder.head.2")?;
        self.conv(ops, stream, &x, h, w, "conv1")
    }
}

/// Extract contiguous columns from every row.
pub fn columns(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    start: usize,
    cols: usize,
) -> Result<Tensor> {
    if start.checked_add(cols).is_none_or(|n| n > x.cols()) || cols == 0 {
        return Err(Error::invalid("column extraction dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), cols)?;
    // SAFETY: checked source column interval, with one output element per lane.
    unsafe {
        ops.launch_1d(
            stream,
            "columns",
            config(&[
                ("xsize", x.size()),
                ("width", x.cols()),
                ("start1", start + 1),
                ("cols", cols),
            ]),
            &Scalars::new().index(out.size()),
            &[x.binding()?, out.binding()?],
            out.size(),
        )?;
    }
    Ok(out)
}

fn stride_two(ops: &Ops, stream: &Stream, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    if !h.is_multiple_of(2) || !w.is_multiple_of(2) || x.rows() != h * w {
        return Err(Error::invalid("VAE downsample dimensions"));
    }
    let out = ops.tensor(stream, h / 2 * (w / 2), x.cols())?;
    // SAFETY: odd coordinates lie within each even-sized input axis.
    unsafe {
        ops.launch_1d(
            stream,
            "train_stride_two",
            config(&[("width", w), ("cols", x.cols()), ("size", x.size())]),
            &Scalars::new().index(out.size()),
            &[x.binding()?, out.binding()?],
            out.size(),
        )?;
    }
    Ok(out)
}
