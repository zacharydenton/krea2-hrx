//! Device operations used by the model-specific reverse pass.
use std::sync::Arc;

use hrx::{Buffer, PooledBuffer, Stream, View};

use crate::kernels::Scalars;
use crate::ops::{Ops, Tensor, config};
use crate::{Error, Result};

/// A row-major FP32 device matrix, used for gradients and optimizer state.
#[derive(Clone)]
pub struct FloatTensor {
    rows: usize,
    cols: usize,
    buffer: Arc<FloatStorage>,
}

enum FloatStorage {
    Owned(Buffer),
    Pooled(PooledBuffer),
}

impl FloatTensor {
    /// Number of rows, fixed when the allocation is created.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns, fixed when the allocation is created.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Allocate a zero-initialized FP32 matrix.
    pub fn zero(stream: &Stream, rows: usize, cols: usize) -> Result<Self> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n > 0 && *n <= 4 * 1073741824usize)
            .ok_or_else(|| Error::invalid("FP32 tensor dimensions"))?;
        let buffer = stream.allocate(bytes)?;
        stream.fill(buffer.binding(), 0)?;
        Ok(Self { rows, cols, buffer: Arc::new(FloatStorage::Owned(buffer)) })
    }

    /// Uninitialized scratch: callers must write every logical element before reading.
    pub(crate) fn scratch(
        ops: &Ops,
        stream: &Stream,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n > 0 && *n <= 4 * 1073741824usize)
            .ok_or_else(|| Error::invalid("FP32 tensor dimensions"))?;
        Ok(Self {
            rows,
            cols,
            buffer: Arc::new(FloatStorage::Pooled(ops.pool().acquire(stream, bytes)?)),
        })
    }

    /// Upload row-major FP32 values.
    pub fn from_slice(
        stream: &mut Stream,
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> Result<Self> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(Error::invalid("FP32 upload dimensions"));
        }
        let tensor = Self::zero(stream, rows, cols)?;
        stream.upload(tensor.binding(), bytemuck::cast_slice(values))?;
        Ok(tensor)
    }

    /// The checked allocation view.
    pub fn binding(&self) -> View<'_> {
        let buffer = match &*self.buffer {
            FloatStorage::Owned(buffer) => buffer,
            FloatStorage::Pooled(buffer) => buffer.buffer(),
        };
        // The pool can return excess capacity; consumers see only the logical matrix.
        buffer.try_slice(0, self.size() * 4).expect("validated FP32 storage")
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Read after queued operations complete.
    pub fn download(&self, stream: &mut Stream) -> Result<Vec<f32>> {
        let mut values = vec![0.0; self.size()];
        stream.read_blocking(self.binding(), bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }

    /// Clear an accumulated gradient.
    pub fn clear(&self, stream: &Stream) -> Result<()> {
        stream.fill(self.binding(), 0).map_err(Error::from)
    }
}

/// BF16 matrix product with transposed right operand and FP32 accumulation.
pub fn matmul(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    if a.cols() != b.cols() || !alpha.is_finite() {
        return Err(Error::invalid("training GEMM dimensions"));
    }
    let out = ops.tensor(stream, a.rows(), b.rows())?;
    if a.rows() >= 512 && b.rows().is_multiple_of(64) && a.cols().is_multiple_of(64) {
        // SAFETY: this kernel permits a ragged M, with full 64-column N/K tiles.
        // Both operands are contiguous row-major BF16 matrices; the output is M x N.
        unsafe {
            ops.launch(
                stream,
                "train_gemm",
                config(&[
                    ("m", a.rows()),
                    ("n", b.rows()),
                    ("k", a.cols()),
                    ("asize", a.size()),
                    ("bsize", b.size()),
                    ("csize", out.size()),
                    ("astride", a.size()),
                    ("bstride", b.size()),
                ]),
                &Scalars::new().index(a.rows()).float(alpha),
                &[a.binding()?, b.binding()?, out.binding()?],
                b.rows() / 64,
                a.rows().div_ceil(128),
                256,
            )?;
        }
        return Ok(out);
    }
    ops.matmul(
        stream,
        "gemm_bf16_bf16_nt",
        a.binding()?,
        b.binding()?,
        out.binding()?,
        a.rows(),
        b.rows(),
        a.cols(),
        1,
        alpha,
        None,
    )?;
    Ok(out)
}

/// BF16 matrix product with an ordinary row-major right operand.
/// The frozen-weight reverse pass uses this to avoid transposing the whole model.
pub fn matmul_nn(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    if a.cols() != b.rows() || !alpha.is_finite() {
        return Err(Error::invalid("training NN GEMM dimensions"));
    }
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    let out = ops.tensor(stream, m, n)?;
    if m >= 512 && n.is_multiple_of(64) && k.is_multiple_of(64) {
        // SAFETY: the matrices are M x K, K x N and M x N; N/K have full 64-wide tiles.
        unsafe {
            ops.launch(
                stream,
                "train_gemm_nn",
                config(&[
                    ("m", m),
                    ("n", n),
                    ("k", k),
                    ("asize", a.size()),
                    ("bsize", b.size()),
                    ("csize", out.size()),
                    ("astride", a.size()),
                    ("bstride", b.size()),
                ]),
                &Scalars::new().index(m).float(alpha),
                &[a.binding()?, b.binding()?, out.binding()?],
                n / 64,
                m.div_ceil(128),
                256,
            )?;
        }
    } else {
        ops.matmul(
            stream,
            "gemm_bf16_bf16_nn",
            a.binding()?,
            b.binding()?,
            out.binding()?,
            m,
            n,
            k,
            1,
            alpha,
            None,
        )?;
    }
    Ok(out)
}

/// FP32 result for an adapter parameter gradient.
pub fn matmul_float(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    alpha: f32,
) -> Result<FloatTensor> {
    if a.cols() != b.cols() || !alpha.is_finite() {
        return Err(Error::invalid("gradient GEMM dimensions"));
    }
    let out = FloatTensor::scratch(ops, stream, a.rows(), b.rows())?;
    ops.matmul(
        stream,
        "gemm_bf16_f32_nt",
        a.binding()?,
        b.binding()?,
        out.binding(),
        a.rows(),
        b.rows(),
        a.cols(),
        1,
        alpha,
        None,
    )?;
    Ok(out)
}

/// FP32 flow MSE and its BF16 activation gradient. Only reduction partials read back.
pub fn flow_loss(
    ops: &Ops,
    stream: &mut Stream,
    prediction: &Tensor,
    target: &FloatTensor,
) -> Result<(f64, Tensor)> {
    if prediction.rows() != target.rows || prediction.cols() != target.cols {
        return Err(Error::invalid("flow loss dimensions"));
    }
    let parts = prediction.size().div_ceil(1024);
    let partials = FloatTensor::scratch(ops, stream, 1, parts)?;
    let gradient = ops.tensor(stream, prediction.rows(), prediction.cols())?;
    // SAFETY: each wave owns at most 1024 elements and one FP32 partial; both inputs
    // and the gradient have the checked matrix shape.
    unsafe {
        ops.launch(
            stream,
            "train_loss",
            config(&[("parts", parts)]),
            &Scalars::new().index(prediction.size()),
            &[prediction.binding()?, target.binding(), gradient.binding()?, partials.binding()],
            parts,
            1,
            32,
        )?;
    }
    let mut sum = 0.0;
    for value in partials.download(stream)? {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid("nonfinite flow loss; update cancelled"));
        }
        sum += f64::from(value);
    }
    Ok((sum / prediction.size() as f64, gradient))
}

/// Transpose a BF16 matrix without a host round trip.
pub fn transpose(ops: &Ops, stream: &Stream, x: &Tensor) -> Result<Tensor> {
    let out = ops.tensor(stream, x.cols(), x.rows())?;
    // SAFETY: each group transposes a 32x32 tile, guarding both ragged edges.
    unsafe {
        ops.launch(
            stream,
            "train_transpose",
            config(&[("rows", x.rows()), ("cols", x.cols())]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, out.binding()?],
            x.rows().div_ceil(32) * x.cols().div_ceil(32),
            1,
            256,
        )?;
    }
    Ok(out)
}

/// Add a scaled BF16 tensor, preserving a single final rounding.
pub fn add_scaled(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    y: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    if x.rows() != y.rows() || x.cols() != y.cols() || !scale.is_finite() {
        return Err(Error::invalid("training add dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: all three bindings have exactly count BF16 elements.
    unsafe {
        ops.launch_1d(
            stream,
            "train_add",
            config(&[]),
            &Scalars::new().index(x.size()).float(scale),
            &[x.binding()?, y.binding()?, out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

/// Add an FP32 parameter gradient into its persistent accumulation buffer.
pub fn accumulate(
    ops: &Ops,
    stream: &Stream,
    dst: &FloatTensor,
    src: &FloatTensor,
) -> Result<()> {
    if dst.rows != src.rows || dst.cols != src.cols {
        return Err(Error::invalid("gradient accumulation dimensions"));
    }
    // SAFETY: both views hold count floats; each thread exclusively owns its output.
    unsafe {
        ops.launch_1d(
            stream,
            "train_accumulate",
            config(&[]),
            &Scalars::new().index(dst.size()),
            &[dst.binding(), src.binding()],
            dst.size(),
        )
    }
}

/// Round FP32 master parameters into an execution copy.
pub fn cast(ops: &Ops, stream: &Stream, x: &FloatTensor) -> Result<Tensor> {
    cast_view(ops, stream, x.binding(), x.rows, x.cols)
}

/// Convert the leading FP32 matrix in a view, allowing extra pooled capacity.
pub fn cast_view(
    ops: &Ops,
    stream: &Stream,
    x: View<'_>,
    rows: usize,
    cols: usize,
) -> Result<Tensor> {
    let bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .filter(|n| *n > 0 && *n <= x.len())
        .ok_or_else(|| Error::invalid("FP32 cast dimensions"))?;
    let x = x.slice(0, bytes)?;
    let out = ops.tensor(stream, rows, cols)?;
    // SAFETY: the input and output contain the same number of elements.
    unsafe {
        ops.launch_1d(
            stream,
            "train_cast",
            config(&[]),
            &Scalars::new().index(rows * cols),
            &[x, out.binding()?],
            rows * cols,
        )?;
    }
    Ok(out)
}

/// Three-axis Krea RoPE, or its transpose for a reverse pass.
pub fn rope(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    cos: &FloatTensor,
    sin: &FloatTensor,
    inverse: bool,
) -> Result<Tensor> {
    if !x.cols().is_multiple_of(128)
        || cos.rows != x.rows()
        || cos.cols != 128
        || sin.rows != cos.rows
        || sin.cols != cos.cols
    {
        return Err(Error::invalid("training rotary dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: x/out match; cos/sin contain 128 coefficients per token.
    unsafe {
        ops.launch_1d(
            stream,
            "train_rope",
            config(&[("cols", x.cols()), ("tables", cos.size())]),
            &Scalars::new().index(x.size()).float(if inverse { -1.0 } else { 1.0 }),
            &[x.binding()?, cos.binding(), sin.binding(), out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

/// Add one to a BF16 vector at the model's rounding boundary.
pub fn one_plus(ops: &Ops, stream: &Stream, x: &Tensor) -> Result<Tensor> {
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: equal-sized BF16 buffers.
    unsafe {
        ops.launch_1d(
            stream,
            "unary_one",
            config(&[]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

/// Pointwise activation derivative multiplied by an incoming gradient.
pub fn activation_backward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    grad: &Tensor,
    sigmoid: bool,
) -> Result<Tensor> {
    if x.rows() != grad.rows() || x.cols() != grad.cols() {
        return Err(Error::invalid("activation gradient dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: matching BF16 matrices, one output element per lane.
    unsafe {
        ops.launch_1d(
            stream,
            if sigmoid { "train_sigmoid_backward" } else { "train_silu_backward" },
            config(&[]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, grad.binding()?, out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

/// Backward through zero-centered RMSNorm with frozen FP32 scales.
pub fn norm_backward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    grad: &Tensor,
    scale: View<'_>,
    eps: f32,
) -> Result<Tensor> {
    if x.rows() != grad.rows()
        || x.cols() != grad.cols()
        || scale.len() != x.cols() * 4
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::invalid("RMSNorm backward dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: matching matrices and one scale per column; one wave owns each row.
    unsafe {
        ops.launch(
            stream,
            "train_norm_backward",
            config(&[("cols", x.cols()), ("size", x.size())]),
            &Scalars::new().index(x.rows()).float(eps),
            &[x.binding()?, grad.binding()?, scale, out.binding()?],
            x.rows(),
            1,
            32,
        )?;
    }
    Ok(out)
}

/// Attention output and unrounded statistics retained for its reverse pass.
pub struct Attention {
    /// BF16 attended values in token-major order.
    pub output: Tensor,
    exact: FloatTensor,
    lse: FloatTensor,
    heads: usize,
    kv: usize,
}

fn attention_config(tokens: usize, heads: usize, kv: usize) -> crate::kernels::Config {
    config(&[
        ("tokens", tokens),
        ("heads", heads),
        ("kv", kv),
        ("qsize", tokens * heads * 128),
        ("ksize", tokens * kv * 128),
        ("stats", tokens * heads),
    ])
}

/// Streaming GQA with linear auxiliary storage and FP32 softmax accumulation.
pub fn attention(
    ops: &Ops,
    stream: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Attention> {
    let heads = q.cols() / 128;
    let kv = k.cols() / 128;
    if kv == 0
        || !heads.is_multiple_of(kv)
        || !q.cols().is_multiple_of(128)
        || !k.cols().is_multiple_of(128)
        || k.cols() != v.cols()
        || q.rows() != k.rows()
        || q.rows() != v.rows()
    {
        return Err(Error::invalid("training attention dimensions"));
    }
    let output = ops.tensor(stream, q.rows(), q.cols())?;
    let exact = FloatTensor::scratch(ops, stream, q.rows(), q.cols())?;
    let lse = FloatTensor::scratch(ops, stream, q.rows(), heads)?;
    // SAFETY: dimensions above match the configured GQA layout; a wave owns one query/head.
    unsafe {
        ops.launch(
            stream,
            "train_attention",
            attention_config(q.rows(), heads, kv),
            &Scalars::new().index(q.rows()),
            &[
                q.binding()?,
                k.binding()?,
                v.binding()?,
                output.binding()?,
                exact.binding(),
                lse.binding(),
            ],
            q.rows() * heads,
            1,
            32,
        )?;
    }
    Ok(Attention { output, exact, lse, heads, kv })
}

/// Compute dQ, dK and dV without atomic accumulation or repeated KV storage.
pub fn attention_backward(
    ops: &Ops,
    stream: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    forward: &Attention,
    grad: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let Attention { heads, kv, .. } = *forward;
    if q.rows() != grad.rows()
        || q.cols() != grad.cols()
        || q.cols() != heads * 128
        || k.cols() != kv * 128
        || v.cols() != k.cols()
        || k.rows() != q.rows()
        || v.rows() != q.rows()
        || forward.exact.rows != q.rows()
        || forward.exact.cols != q.cols()
        || forward.lse.rows != q.rows()
        || forward.lse.cols != heads
    {
        return Err(Error::invalid("attention backward dimensions"));
    }
    let delta = FloatTensor::scratch(ops, stream, q.rows(), heads)?;
    let dq = ops.tensor(stream, q.rows(), q.cols())?;
    let dk = ops.tensor(stream, k.rows(), k.cols())?;
    let dv = ops.tensor(stream, v.rows(), v.cols())?;
    let conf = attention_config(q.rows(), heads, kv);
    let scalars = Scalars::new().index(q.rows());
    // SAFETY: each kernel owns disjoint output rows; all shapes match conf and share the stream.
    unsafe {
        ops.launch(
            stream,
            "train_attention_delta",
            conf.clone(),
            &scalars,
            &[grad.binding()?, forward.exact.binding(), delta.binding()],
            q.rows() * heads,
            1,
            32,
        )?;
        let inputs = [
            q.binding()?,
            k.binding()?,
            v.binding()?,
            grad.binding()?,
            forward.lse.binding(),
            delta.binding(),
        ];
        let mut args = inputs.to_vec();
        args.extend([dq.binding()?, dq.binding()?]);
        ops.launch(
            stream,
            "train_attention_dq",
            conf.clone(),
            &scalars,
            &args,
            q.rows() * heads,
            1,
            32,
        )?;
        args[6] = dk.binding()?;
        args[7] = dv.binding()?;
        ops.launch(stream, "train_attention_dkv", conf, &scalars, &args, q.rows() * kv, 1, 32)?;
    }
    Ok((dq, dk, dv))
}
