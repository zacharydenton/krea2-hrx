//! Device operations used by the model-specific reverse pass.
use std::sync::Arc;

use hrx::{Buffer, PooledBuffer, Stream, View};

use crate::kernels::Scalars;
use crate::ops::{Binary, Ops, Tensor, Unary, Weight, config};
use crate::{Error, Result};

#[derive(Default, serde::Serialize)]
struct HostTiming {
    calls: usize,
    milliseconds: f64,
}

thread_local! {
    static HOST_TIMINGS: std::cell::RefCell<std::collections::BTreeMap<&'static str, HostTiming>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

fn host_profile() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("KREA2_HOST_PROFILE").is_some_and(|value| value == "1")
    });
    *ENABLED
}

pub(crate) fn profile<T>(name: &'static str, run: impl FnOnce() -> Result<T>) -> Result<T> {
    if !host_profile() {
        return run();
    }
    let started = std::time::Instant::now();
    let result = run();
    let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
    HOST_TIMINGS.with_borrow_mut(|timings| {
        let row = timings.entry(name).or_default();
        row.calls += 1;
        row.milliseconds += milliseconds;
    });
    result
}

pub(crate) fn clear_host_timings() {
    if host_profile() {
        HOST_TIMINGS.with_borrow_mut(|timings| timings.clear());
    }
}

pub(crate) fn report_host_timings() {
    if host_profile() {
        HOST_TIMINGS.with_borrow(|timings| {
            eprintln!("training host operations: {}", serde_json::to_string(timings).unwrap());
        });
    }
}

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
    profile("matmul", || matmul_inner(ops, stream, a, b, alpha))
}

fn matmul_inner(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    if a.cols() != b.cols() || !alpha.is_finite() {
        return Err(Error::invalid("training GEMM dimensions"));
    }
    let out = profile("bf16_allocation", || ops.tensor(stream, a.rows(), b.rows()))?;
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
    profile("matmul_nn", || matmul_nn_inner(ops, stream, a, b, alpha))
}

fn matmul_nn_inner(
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
    let out = profile("bf16_allocation", || ops.tensor(stream, m, n))?;
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

/// Add a rank-sized matrix product to a base tensor, rounding the product to BF16 first.
/// `transposed` selects an N x K right operand; otherwise it is K x N.
pub fn matmul_add(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    base: &Tensor,
    scale: f32,
    transposed: bool,
) -> Result<Tensor> {
    let (m, k) = (a.rows(), a.cols());
    let (bk, n) = if transposed { (b.cols(), b.rows()) } else { (b.rows(), b.cols()) };
    if k != bk || (base.rows(), base.cols()) != (m, n) || !scale.is_finite() {
        return Err(Error::invalid("training GEMM add dimensions"));
    }
    if !n.is_multiple_of(64) || !k.is_multiple_of(32) || k > 64 {
        let product = if transposed {
            matmul(ops, stream, a, b, 1.0)?
        } else {
            matmul_nn(ops, stream, a, b, 1.0)?
        };
        return add_scaled(ops, stream, base, &product, scale);
    }
    let out = ops.tensor(stream, m, n)?;
    // SAFETY: full N/K tiles, guarded ragged M; base/out are M x N, A is M x K,
    // and the specialized right-operand layout matches B's validated dimensions.
    unsafe {
        ops.launch(
            stream,
            "train_gemm_lora_add",
            config(&[
                ("m", m),
                ("n", n),
                ("k", k),
                ("asize", a.size()),
                ("bsize", b.size()),
                ("csize", out.size()),
                ("transposed", usize::from(transposed)),
            ]),
            &Scalars::new().index(m).float(scale),
            &[a.binding()?, b.binding()?, base.binding()?, out.binding()?],
            n / 64,
            m.div_ceil(64),
            256,
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
    profile("matmul_float", || matmul_float_inner(ops, stream, a, b, alpha))
}

fn matmul_float_inner(
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

/// Accumulate `alpha * a^T * b` directly into an FP32 parameter gradient.
/// Inputs stay in their original row-major activation layouts.
pub fn matmul_tn_accumulate(
    ops: &Ops,
    stream: &Stream,
    a: &Tensor,
    b: &Tensor,
    dst: &FloatTensor,
    alpha: f32,
) -> Result<()> {
    if a.rows() != b.rows()
        || dst.rows != a.cols()
        || dst.cols != b.cols()
        || !alpha.is_finite()
    {
        return Err(Error::invalid("gradient TN GEMM dimensions"));
    }
    profile("matmul_tn_accumulate", || {
        // SAFETY: A is K x M, B is K x N, and dst is M x N. Each group
        // exclusively owns a 32x32 output tile; all ragged edges are guarded.
        unsafe {
            ops.launch(
                stream,
                "train_gemm_tn_accumulate",
                config(&[("m", a.cols()), ("n", b.cols()), ("k", a.rows())]),
                &Scalars::new().index(dst.size()).float(alpha),
                &[a.binding()?, b.binding()?, dst.binding()],
                b.cols().div_ceil(32),
                a.cols().div_ceil(32),
                128,
            )
        }
    })
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
    profile("transpose", || transpose_inner(ops, stream, x))
}

fn transpose_inner(ops: &Ops, stream: &Stream, x: &Tensor) -> Result<Tensor> {
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

/// Frozen Q/K RMSNorm followed by rotary, retaining normalization rounding in registers.
pub fn norm_rope(
    ops: &Ops,
    stream: &mut Stream,
    x: &Tensor,
    weight: &Weight,
    cos: &FloatTensor,
    sin: &FloatTensor,
    eps: f32,
) -> Result<Tensor> {
    if !x.cols().is_multiple_of(128)
        || weight.count != 128
        || (cos.rows, cos.cols) != (x.rows(), 128)
        || (sin.rows, sin.cols) != (cos.rows, cos.cols)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::invalid("norm rotary dimensions"));
    }
    let heads = x.cols() / 128;
    let rows = x.rows() * heads;
    let scale = weight.f32_values(stream)?;
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: eight waves per workgroup, each guarding one 128-channel head;
    // matching input/output matrices, 128 weights, and 128 table values per token.
    unsafe {
        ops.launch(
            stream,
            "train_norm_rope",
            config(&[("size", x.size()), ("heads", heads), ("tables", cos.size())]),
            &Scalars::new().index(rows).float(eps),
            &[x.binding()?, scale, cos.binding(), sin.binding(), out.binding()?],
            rows.div_ceil(8),
            1,
            256,
        )?;
    }
    Ok(out)
}

/// Inverse rotary followed by frozen Q/K RMSNorm backward, with BF16 rotary rounding.
#[allow(clippy::too_many_arguments)]
pub fn rope_norm_backward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    grad: &Tensor,
    scale: View<'_>,
    cos: &FloatTensor,
    sin: &FloatTensor,
    eps: f32,
) -> Result<Tensor> {
    if !x.cols().is_multiple_of(128)
        || (grad.rows(), grad.cols()) != (x.rows(), x.cols())
        || scale.len() != 128 * 4
        || (cos.rows, cos.cols) != (x.rows(), 128)
        || (sin.rows, sin.cols) != (cos.rows, cos.cols)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::invalid("rotary norm backward dimensions"));
    }
    let heads = x.cols() / 128;
    let rows = x.rows() * heads;
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: one wave per 128-channel head, with matching matrices and rotary tables.
    unsafe {
        ops.launch(
            stream,
            "train_rope_norm_backward",
            config(&[
                ("cols", 128),
                ("size", x.size()),
                ("heads", heads),
                ("tables", cos.size()),
            ]),
            &Scalars::new().index(rows).float(eps),
            &[
                x.binding()?,
                grad.binding()?,
                scale,
                cos.binding(),
                sin.binding(),
                out.binding()?,
            ],
            rows,
            1,
            32,
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

/// SiLU/sigmoid times a value, returning the rounded activation and product.
pub fn gated_forward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    value: &Tensor,
    sigmoid: bool,
) -> Result<(Tensor, Tensor)> {
    if (value.rows(), value.cols()) != (x.rows(), x.cols()) {
        return Err(Error::invalid("gated forward dimensions"));
    }
    if !x.size().is_multiple_of(4) {
        let activation =
            ops.unary(stream, x, if sigmoid { Unary::Sigmoid } else { Unary::Silu })?;
        let out = ops.binary(stream, &activation, value, Binary::Mul)?;
        return Ok((activation, out));
    }
    let activation = ops.tensor(stream, x.rows(), x.cols())?;
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: four BF16 elements per thread, matching matrix extents divisible by four.
    unsafe {
        ops.launch(
            stream,
            "train_gated_forward",
            config(&[("sigmoid", usize::from(sigmoid))]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, value.binding()?, activation.binding()?, out.binding()?],
            x.size().div_ceil(1024),
            1,
            256,
        )?;
    }
    Ok((activation, out))
}

/// Broadcast a gate over a branch and add the residual, rounding the product first.
pub fn residual_gate(
    ops: &Ops,
    stream: &Stream,
    residual: &Tensor,
    value: &Tensor,
    modulation: &Tensor,
) -> Result<Tensor> {
    if (value.rows(), value.cols()) != (residual.rows(), residual.cols())
        || (modulation.rows(), modulation.cols()) != (1, value.cols())
    {
        return Err(Error::invalid("residual gate dimensions"));
    }
    if !value.cols().is_multiple_of(4) {
        let product = ops.binary(stream, value, modulation, Binary::Mul)?;
        return ops.binary(stream, residual, &product, Binary::Add);
    }
    let out = ops.tensor(stream, value.rows(), value.cols())?;
    // SAFETY: four contiguous elements per thread cannot cross a broadcast row;
    // matrices match, with one modulation value per column and columns divisible by four.
    unsafe {
        ops.launch(
            stream,
            "train_residual_gate",
            config(&[("cols", value.cols())]),
            &Scalars::new().index(value.size()),
            &[residual.binding()?, value.binding()?, modulation.binding()?, out.binding()?],
            value.size().div_ceil(1024),
            1,
            256,
        )?;
    }
    Ok(out)
}

/// Reverse a cached SiLU/sigmoid gate times a value, returning `(dx, dvalue)`.
/// Preserve the BF16 product rounding before applying the activation derivative.
pub fn gated_backward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    activation: &Tensor,
    value: &Tensor,
    grad: &Tensor,
    sigmoid: bool,
) -> Result<(Tensor, Tensor)> {
    if [activation, value, grad].iter().any(|t| t.rows() != x.rows() || t.cols() != x.cols()) {
        return Err(Error::invalid("gated gradient dimensions"));
    }
    let dx = ops.tensor(stream, x.rows(), x.cols())?;
    let dv = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: matching BF16 matrices and two distinct, fully written outputs.
    unsafe {
        ops.launch_1d(
            stream,
            "train_gated_backward",
            config(&[("sigmoid", usize::from(sigmoid))]),
            &Scalars::new().index(x.size()),
            &[
                x.binding()?,
                activation.binding()?,
                value.binding()?,
                grad.binding()?,
                dx.binding()?,
                dv.binding()?,
            ],
            x.size(),
        )?;
    }
    Ok((dx, dv))
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
    let kernel =
        if x.cols() == 128 { "train_norm_backward_head" } else { "train_norm_backward" };
    // SAFETY: matching matrices and one scale per column; one wave owns each row.
    unsafe {
        ops.launch(
            stream,
            kernel,
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

/// RMSNorm with broadcast modulation, retaining normalized values for modulation gradients.
/// Preserve the separate normalization, one-plus, multiplication and addition roundings.
pub fn norm_modulated(
    ops: &Ops,
    stream: &mut Stream,
    x: &Tensor,
    weight: &Weight,
    modulation: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if weight.count != x.cols()
        || [modulation, shift].iter().any(|t| (t.rows(), t.cols()) != (1, x.cols()))
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::invalid("modulated RMSNorm dimensions"));
    }
    let scale = weight.f32_values(stream)?;
    let norm = ops.tensor(stream, x.rows(), x.cols())?;
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: one workgroup per row, three matching BF16 matrices, and three
    // per-column vectors: FP32 norm weights and BF16 modulation/shift.
    unsafe {
        ops.launch(
            stream,
            "train_norm_modulated",
            config(&[("cols", x.cols()), ("xsize", x.size())]),
            &Scalars::new().index(x.rows()).float(eps),
            &[
                x.binding()?,
                scale,
                norm.binding()?,
                modulation.binding()?,
                shift.binding()?,
                out.binding()?,
            ],
            x.rows(),
            1,
            256,
        )?;
    }
    Ok((norm, out))
}

/// Apply modulation scaling, RMSNorm backward and a residual gradient in one dispatch.
/// Preserve the BF16 boundaries of `one_plus`, multiplication, norm and addition.
#[allow(clippy::too_many_arguments)]
pub fn norm_modulated_backward(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    grad: &Tensor,
    scale: View<'_>,
    modulation: &Tensor,
    residual: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if [grad, residual].iter().any(|t| (t.rows(), t.cols()) != (x.rows(), x.cols()))
        || (modulation.rows(), modulation.cols()) != (1, x.cols())
        || scale.len() != x.cols() * 4
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::invalid("modulated RMSNorm backward dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: one wave per row; matching BF16 matrices and per-column scale vectors.
    unsafe {
        ops.launch(
            stream,
            "train_norm_modulated_backward",
            config(&[("cols", x.cols()), ("size", x.size())]),
            &Scalars::new().index(x.rows()).float(eps),
            &[
                x.binding()?,
                grad.binding()?,
                scale,
                modulation.binding()?,
                residual.binding()?,
                out.binding()?,
            ],
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
    sequence: usize,
    packed: Option<Box<AttentionInputs>>,
}

struct AttentionInputs {
    sources: [Tensor; 3],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    qt: Tensor,
    kt: Tensor,
}

fn attention_config(
    tokens: usize,
    heads: usize,
    kv: usize,
    sequence: usize,
) -> crate::kernels::Config {
    config(&[
        ("tokens", tokens),
        ("sequence", sequence),
        ("heads", heads),
        ("kv", kv),
        ("qsize", tokens * heads * 128),
        ("ksize", tokens * kv * 128),
        ("stats", tokens * heads),
    ])
}

fn pack_attention_input(
    ops: &Ops,
    stream: &Stream,
    input: &Tensor,
    capacity: usize,
) -> Result<(Tensor, Tensor)> {
    if capacity < input.rows() {
        return Err(Error::invalid("attention packing capacity"));
    }
    let out = ops.tensor(stream, capacity, input.cols())?;
    let transposed = ops.tensor(stream, input.cols(), capacity)?;
    // SAFETY: guarded input reads preserve BF16 bits. The tiled kernel writes
    // both complete output matrices, explicitly zeroing all padded rows.
    unsafe {
        ops.launch(
            stream,
            "train_attention_pack",
            config(&[("rows", input.rows()), ("cols", input.cols()), ("capacity", capacity)]),
            &Scalars::new().index(input.size()),
            &[input.binding()?, out.binding()?, transposed.binding()?],
            capacity.div_ceil(32) * input.cols().div_ceil(32),
            1,
            256,
        )?;
    }
    Ok((out, transposed))
}

/// Streaming GQA with linear auxiliary storage and FP32 softmax accumulation.
pub fn attention(
    ops: &Ops,
    stream: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Attention> {
    attention_batched(ops, stream, q, k, v, q.rows())
}

/// Independent attention sequences packed along the row dimension.
pub fn attention_batched(
    ops: &Ops,
    stream: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sequence: usize,
) -> Result<Attention> {
    let heads = q.cols() / 128;
    let kv = k.cols() / 128;
    if sequence == 0
        || !q.rows().is_multiple_of(sequence)
        || kv == 0
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
    if sequence == q.rows()
        && (16..=65536).contains(&q.rows())
        && q.cols() <= 32768
        && heads == kv * 4
    {
        let capacity = q.rows().div_ceil(16) * 16 + 16;
        let (qp, qt) = pack_attention_input(ops, stream, q, capacity)?;
        let (kp, kt) = pack_attention_input(ops, stream, k, capacity)?;
        let (vp, vt) = pack_attention_input(ops, stream, v, capacity)?;
        // SAFETY: operands have zero-filled 16-token headroom; each workgroup owns
        // sixteen queries and four heads, writing only valid rows to all outputs.
        unsafe {
            ops.launch(
                stream,
                "train_attention_flash",
                config(&[
                    ("q_stride", q.cols()),
                    ("kv_stride", k.cols()),
                    ("out_stride", q.cols()),
                    ("tokens", q.rows()),
                    ("token_capacity", capacity),
                ]),
                &Scalars::new().index(q.rows()),
                &[
                    qp.binding()?,
                    kp.binding()?,
                    vt.binding()?,
                    output.binding()?,
                    exact.binding(),
                    lse.binding(),
                ],
                q.rows().div_ceil(16),
                kv,
                128,
            )?;
        }
        return Ok(Attention {
            output,
            exact,
            lse,
            heads,
            kv,
            sequence,
            packed: Some(Box::new(AttentionInputs {
                sources: [q.clone(), k.clone(), v.clone()],
                q: qp,
                k: kp,
                v: vp,
                qt,
                kt,
            })),
        });
    }
    // SAFETY: dimensions above match the configured GQA layout; a wave owns one query/head.
    unsafe {
        ops.launch(
            stream,
            "train_attention",
            attention_config(q.rows(), heads, kv, sequence),
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
    Ok(Attention { output, exact, lse, heads, kv, sequence, packed: None })
}

/// Compute dQ, dK and dV without atomic accumulation or repeated KV storage.
/// Inputs must be the unchanged tensors used by the corresponding forward pass.
pub fn attention_backward(
    ops: &Ops,
    stream: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    forward: &Attention,
    grad: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let Attention { heads, kv, sequence, .. } = *forward;
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
    if let Some(packed) = &forward.packed
        && !packed.sources.iter().zip([q, k, v]).all(|(a, b)| a.same_view(b))
    {
        return Err(Error::invalid("attention backward requires its forward inputs"));
    }
    let delta = FloatTensor::scratch(ops, stream, q.rows(), heads)?;
    let dq = ops.tensor(stream, q.rows(), q.cols())?;
    let dk = ops.tensor(stream, k.rows(), k.cols())?;
    let dv = ops.tensor(stream, v.rows(), v.cols())?;
    let conf = attention_config(q.rows(), heads, kv, sequence);
    let scalars = Scalars::new().index(q.rows());
    let tiled = sequence == q.rows()
        && (16..=65536).contains(&q.rows())
        && q.cols() <= 32768
        && heads == kv * 4;
    // SAFETY: each kernel owns disjoint output rows; all shapes match conf and share the stream.
    unsafe {
        ops.launch(
            stream,
            if tiled { "train_attention_flash_delta" } else { "train_attention_delta" },
            if tiled {
                config(&[("tokens", q.rows()), ("heads", heads), ("q_stride", q.cols())])
            } else {
                conf.clone()
            },
            &scalars,
            &[grad.binding()?, forward.exact.binding(), delta.binding()],
            if tiled { q.rows().div_ceil(16) } else { q.rows() * heads },
            if tiled { heads } else { 1 },
            32,
        )?;
        if let Some(packed) = &forward.packed {
            let capacity = packed.q.rows();
            let (gp, gt) = pack_attention_input(ops, stream, grad, capacity)?;
            let tiled = config(&[
                ("tokens", q.rows()),
                ("token_capacity", capacity),
                ("q_stride", q.cols()),
                ("kv_stride", k.cols()),
                ("heads", heads),
                ("kv", kv),
            ]);
            let mut args = vec![
                packed.q.binding()?,
                packed.k.binding()?,
                packed.v.binding()?,
                gp.binding()?,
                forward.lse.binding(),
                delta.binding(),
                packed.kt.binding()?,
                gt.binding()?,
                dq.binding()?,
                dq.binding()?,
            ];
            // Padded inputs cover every 16-row tile. Each wave owns its output
            // rows/head; dKV reduces the four query heads locally without atomics.
            ops.launch(
                stream,
                "train_attention_flash_dq",
                tiled.clone(),
                &scalars,
                &args,
                q.rows().div_ceil(16),
                heads,
                32,
            )?;
            args[6] = packed.qt.binding()?;
            args[8] = dk.binding()?;
            args[9] = dv.binding()?;
            ops.launch(
                stream,
                "train_attention_flash_dkv",
                tiled,
                &scalars,
                &args,
                q.rows().div_ceil(16),
                kv,
                32,
            )?;
            return Ok((dq, dk, dv));
        }
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

/// FP32 row reduction, accumulated into one row of a small shared gradient.
pub fn sum_rows_accumulate(
    ops: &Ops,
    stream: &Stream,
    x: &Tensor,
    dst: &FloatTensor,
    row: usize,
) -> Result<()> {
    if dst.cols() != x.cols() || row >= dst.rows() {
        return Err(Error::invalid("row reduction dimensions"));
    }
    // SAFETY: each column owns one output; the slice is exactly the selected FP32 row.
    unsafe {
        ops.launch_1d(
            stream,
            "train_sum_rows",
            config(&[("rows", x.rows()), ("cols", x.cols()), ("size", x.size())]),
            &Scalars::new().index(x.cols()),
            &[x.binding()?, dst.binding().slice(row * x.cols() * 4, x.cols() * 4)?],
            x.cols(),
        )
    }
}

/// Accumulate scale, shift and residual-gate gradients into three consecutive rows.
/// Product rounding and FP32 reduction order match the separate operations.
#[allow(clippy::too_many_arguments)]
pub fn modulation_backward(
    ops: &Ops,
    stream: &Stream,
    residual_grad: &Tensor,
    branch: &Tensor,
    affine_grad: &Tensor,
    normalized: &Tensor,
    dst: &FloatTensor,
    row: usize,
) -> Result<()> {
    let x = affine_grad;
    if [residual_grad, branch, normalized]
        .iter()
        .any(|t| (t.rows(), t.cols()) != (x.rows(), x.cols()))
        || dst.cols() != x.cols()
        || dst.rows() < 3
        || row > dst.rows() - 3
    {
        return Err(Error::invalid("modulation gradient dimensions"));
    }
    // SAFETY: matching BF16 inputs; each column owns its three FP32 output cells.
    unsafe {
        ops.launch_1d(
            stream,
            "train_modulation_backward",
            config(&[("rows", x.rows()), ("cols", x.cols()), ("size", x.size())]),
            &Scalars::new().index(x.cols()),
            &[
                residual_grad.binding()?,
                branch.binding()?,
                x.binding()?,
                normalized.binding()?,
                dst.binding().slice(row * x.cols() * 4, 3 * x.cols() * 4)?,
            ],
            x.cols(),
        )
    }
}

/// Sum a broadcast gradient in FP32, rounding once at the activation boundary.
pub fn sum_rows(ops: &Ops, stream: &Stream, x: &Tensor) -> Result<Tensor> {
    let sum = FloatTensor::scratch(ops, stream, 1, x.cols())?;
    sum.clear(stream)?;
    sum_rows_accumulate(ops, stream, x, &sum, 0)?;
    cast(ops, stream, &sum)
}

/// Derivative of the same tanh GELU used in Krea's time/text projections.
pub fn gelu_backward(ops: &Ops, stream: &Stream, x: &Tensor, grad: &Tensor) -> Result<Tensor> {
    if (x.rows(), x.cols()) != (grad.rows(), grad.cols()) {
        return Err(Error::invalid("GELU gradient dimensions"));
    }
    let out = ops.tensor(stream, x.rows(), x.cols())?;
    // SAFETY: equal-sized BF16 operands and a guarded pointwise output.
    unsafe {
        ops.launch_1d(
            stream,
            "train_gelu_backward",
            config(&[]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, grad.binding()?, out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

/// Exchange the layer/channel axes of token-major Qwen taps without rounding.
pub fn permute_taps(ops: &Ops, stream: &Stream, x: &Tensor, reverse: bool) -> Result<Tensor> {
    if !x.size().is_multiple_of(12 * 2560) {
        return Err(Error::invalid("tap permutation dimensions"));
    }
    let tokens = x.size() / (12 * 2560);
    let out = if reverse {
        ops.tensor(stream, tokens * 12, 2560)?
    } else {
        ops.tensor(stream, tokens * 2560, 12)?
    };
    // SAFETY: the permutation maps every element bijectively within a token's 12x2560 slab.
    unsafe {
        ops.launch_1d(
            stream,
            "train_taps",
            config(&[("reverse", if reverse { 2 } else { 1 })]),
            &Scalars::new().index(x.size()),
            &[x.binding()?, out.binding()?],
            x.size(),
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a gfx1151 GPU"]
    fn attention_pack_preserves_bits_and_zeroes_padding() {
        let mut stream = Stream::open().unwrap();
        let ops = Ops::new(hrx::BufferPool::new());
        for (rows, cols) in [(1, 1), (7, 33), (16, 128), (17, 128), (32, 1536), (33, 6144)] {
            let bits = [0u16, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f80, 0x7fc1, 0xffff];
            let values = (0..rows * cols).map(|i| bits[i % bits.len()]).collect::<Vec<_>>();
            let input =
                Tensor::from_slice(ops.pool(), &mut stream, &values, rows, cols).unwrap();
            let capacity = rows.div_ceil(16) * 16 + 16;
            let (padded, transposed) =
                pack_attention_input(&ops, &stream, &input, capacity).unwrap();
            let padded = padded.download(&mut stream).unwrap();
            let transposed = transposed.download(&mut stream).unwrap();
            for r in 0..capacity {
                for c in 0..cols {
                    let expected = if r < rows { values[r * cols + c] } else { 0 };
                    assert_eq!(
                        padded[r * cols + c],
                        expected,
                        "row-major {rows}x{cols} at {r},{c}"
                    );
                    assert_eq!(
                        transposed[c * capacity + r],
                        expected,
                        "transposed {rows}x{cols} at {r},{c}"
                    );
                }
            }
            assert!(pack_attention_input(&ops, &stream, &input, rows - 1).is_err());
        }
    }
}
