#include "native_ops.h"
namespace krea_native {
__global__ void bias_cast(const float *x, const B *bias, B *y, size_t n,
                          int cols) {
  size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i < n)
    y[i] = B(x[i] + float(bias[i % cols]));
}
__global__ void unary_kernel(const B *x, B *y, size_t n, int op) {
  size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n)
    return;
  float a = float(x[i]);
  y[i] =
      B(op == 0 ? a / (1 + expf(-a))
        : op == 1
            ? .5f * a *
                  (1 + tanhf(.7978845608028654f * (a + .044715f * a * a * a)))
            : 1 / (1 + expf(-a)));
}
__global__ void binary_kernel(const B *x, const B *y, B *z, size_t n, size_t yn,
                              int op) {
  size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i < n)
    z[i] = B(op == 0 ? float(x[i]) + float(y[i % yn])
                     : float(x[i]) * float(y[i % yn]));
}
__global__ void norm_kernel(const B *x, const B *w, B *y, int cols, int mode,
                            float eps) {
  int row = blockIdx.x, lane = threadIdx.x;
  float sum = 0;
  for (int j = lane; j < cols; j += 256) {
    float a = float(x[size_t(row) * cols + j]);
    sum += a * a;
  }
  __shared__ float sums[256];
  sums[lane] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (lane < d)
      sums[lane] += sums[lane + d];
    __syncthreads();
  }
  float inv = mode == 2 ? 1 / fmaxf(sqrtf(sums[0]), 1e-12f)
                        : rsqrtf(sums[0] / cols + eps);
  for (int j = lane; j < cols; j += 256) {
    float a = float(x[size_t(row) * cols + j]) * inv;
    if (mode == 0)
      a *= 1 + float(w[j]); // Krea zero-centered fp32 norm.
    else if (mode == 1)
      a = float(B(a)) * float(w[j]); // Qwen rounds before its learned scale.
    else
      a = float(B(float(B(a)) * sqrtf(float(cols)))) *
          float(w[j]); // VAE L2 norm.
    y[size_t(row) * cols + j] = B(a);
  }
}
__global__ void head_pack(const B *x, B *y, int tokens, int heads, int kv,
                          int dim, size_t n, bool unpack) {
  size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n)
    return;
  int d = i % dim, s = (i / dim) % tokens, h = (i / dim / tokens) % heads,
      b = i / dim / tokens / heads;
  size_t j = ((size_t(b) * tokens + s) * kv + h / (heads / kv)) * dim + d;
  if (unpack)
    y[j] = x[i];
  else
    y[i] = x[j];
}
__global__ void softmax_kernel(const float *x, B *y, int tokens, bool causal) {
  int row = blockIdx.x, lane = threadIdx.x, q = row % tokens;
  float mx = -INFINITY;
  for (int k = lane; k < tokens; k += 256)
    if (!causal || k <= q)
      mx = fmaxf(mx, x[size_t(row) * tokens + k]);
  __shared__ float buf[256];
  buf[lane] = mx;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (lane < d)
      buf[lane] = fmaxf(buf[lane], buf[lane + d]);
    __syncthreads();
  }
  mx = buf[0];
  __syncthreads();
  float sum = 0;
  for (int k = lane; k < tokens; k += 256)
    if (!causal || k <= q)
      sum += expf(x[size_t(row) * tokens + k] - mx);
  buf[lane] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (lane < d)
      buf[lane] += buf[lane + d];
    __syncthreads();
  }
  for (int k = lane; k < tokens; k += 256)
    y[size_t(row) * tokens + k] =
        B((!causal || k <= q) ? expf(x[size_t(row) * tokens + k] - mx) / buf[0]
                              : 0);
}
__global__ void rope_kernel(const B *x, B *y, size_t n, int heads, int dim,
                            float theta) {
  size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n)
    return;
  int d = i % dim, pos = i / dim / heads;
  int other = d < dim / 2 ? d + dim / 2 : d - dim / 2;
  float angle = pos * powf(theta, -2.f * (d % (dim / 2)) / dim);
  B c = B(cosf(angle)), s = B(sinf(angle));
  y[i] = B(float(B(float(x[i]) * float(c))) +
           float(B((d < dim / 2 ? -1.f : 1.f) * float(x[i - d + other]) *
                   float(s))));
}
__global__ void im2col(const B *x, B *y, int h, int w, int channels,
                       int kernel) {
  size_t n = size_t(h) * w * channels * kernel * kernel,
         i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n)
    return;
  int k = i % (channels * kernel * kernel),
      p = i / (channels * kernel * kernel), c = k / (kernel * kernel);
  int yy = p / w + (k / kernel) % kernel - kernel / 2,
      xx = p % w + k % kernel - kernel / 2;
  y[i] = (yy < 0 || yy >= h || xx < 0 || xx >= w)
             ? B(0.f)
             : x[(size_t(yy) * w + xx) * channels + c];
}
__global__ void upsample_kernel(const B *x, B *y, int h, int w, int c) {
  size_t i = blockIdx.x * 256 + threadIdx.x, n = size_t(h) * w * 4 * c;
  if (i < n) {
    int p = i / c;
    y[i] = x[(size_t(p / (2 * w) / 2) * w + (p % (2 * w)) / 2) * c + i % c];
  }
}
Tensor Ops::linear(const Tensor &x, const Weight &w, const B *bias) {
  int n = w.shape[0], k = x.cols;
  if (w.t.size() != size_t(n) * k)
    throw std::invalid_argument("linear dimensions");
  Tensor y(x.rows, n);
  float alpha = 1, beta = 0;
  void *out = y.ptr;
  std::shared_ptr<void> temp;
  if (bias) {
    temp = device_storage(y.size() * 4);
    out = temp.get();
  }
  blas_check(hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, x.rows, k,
                           &alpha, w.t.ptr, HIP_R_16BF, k, x.ptr, HIP_R_16BF, k,
                           &beta, out, bias ? HIP_R_32F : HIP_R_16BF, n,
                           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  if (bias)
    bias_cast<<<(y.size() + 255) / 256, 256>>>((float *)out, bias, y.ptr,
                                               y.size(), n);
  hip_check(hipGetLastError());
  return y;
}
Tensor Ops::norm(const Tensor &x, const Tensor &w, int mode, float eps) {
  if (w.size() != size_t(x.cols))
    throw std::invalid_argument("norm width");
  Tensor y(x.rows, x.cols);
  norm_kernel<<<x.rows, 256>>>(x.ptr, w.ptr, y.ptr, x.cols, mode, eps);
  hip_check(hipGetLastError());
  return y;
}
Tensor Ops::unary(const Tensor &x, int op) {
  Tensor y(x.rows, x.cols);
  unary_kernel<<<(x.size() + 255) / 256, 256>>>(x.ptr, y.ptr, x.size(), op);
  hip_check(hipGetLastError());
  return y;
}
Tensor Ops::binary(const Tensor &x, const Tensor &y, int op) {
  if (!y.size() || x.size() % y.size())
    throw std::invalid_argument("binary broadcast");
  Tensor z(x.rows, x.cols);
  binary_kernel<<<(x.size() + 255) / 256, 256>>>(x.ptr, y.ptr, z.ptr, x.size(),
                                                 y.size(), op);
  hip_check(hipGetLastError());
  return z;
}
Tensor Ops::rope(const Tensor &x, int tokens, int heads, float theta,
                 bool interleaved) {
  if (interleaved)
    throw std::invalid_argument("split-half rotary required");
  Tensor y(x.rows, x.cols);
  rope_kernel<<<(x.size() + 255) / 256, 256>>>(x.ptr, y.ptr, x.size(), heads,
                                               x.cols / heads, theta);
  hip_check(hipGetLastError());
  (void)tokens;
  return y;
}
Tensor Ops::attention(const Tensor &q, const Tensor &k, const Tensor &v,
                      int batch, int s, int heads, int kv, int d, bool causal) {
  if (batch < 1 || s < 1 || heads < 1 || kv < 1 || d < 1 || heads % kv ||
      q.size() != size_t(batch) * s * heads * d ||
      k.size() != size_t(batch) * s * kv * d || v.size() != k.size())
    throw std::invalid_argument("attention dimensions");
  Tensor qp(batch * heads * s, d), kp(batch * heads * s, d),
      vp(batch * heads * s, d);
  for (auto pair : {std::pair{&q, &qp}, std::pair{&k, &kp}, std::pair{&v, &vp}})
    head_pack<<<(qp.size() + 255) / 256, 256>>>(
        pair.first->ptr, pair.second->ptr, s, heads,
        pair.first == &q ? heads : kv, d, qp.size(), false);
  size_t count = size_t(batch) * heads * s * s;
  auto scores = device_storage(count * 4);
  void *p = scores.get();
  float alpha = 1 / sqrtf(float(d)), beta = 0;
  blas_check(hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, s, s, d, &alpha, kp.ptr, HIP_R_16BF,
      d, int64_t(s) * d, qp.ptr, HIP_R_16BF, d, int64_t(s) * d, &beta, p,
      HIP_R_32F, s, int64_t(s) * s, batch * heads, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
  Tensor probs(batch * heads * s, s), op(batch * heads * s, d),
      out(batch * s, heads * d);
  softmax_kernel<<<batch * heads * s, 256>>>((float *)p, probs.ptr, s, causal);
  alpha = 1;
  blas_check(hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_N, HIPBLAS_OP_N, d, s, s, &alpha, vp.ptr, HIP_R_16BF,
      d, int64_t(s) * d, probs.ptr, HIP_R_16BF, s, int64_t(s) * s, &beta,
      op.ptr, HIP_R_16BF, d, int64_t(s) * d, batch * heads, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
  head_pack<<<(op.size() + 255) / 256, 256>>>(op.ptr, out.ptr, s, heads, heads,
                                              d, op.size(), true);
  hip_check(hipGetLastError());
  return out;
}
Tensor Ops::conv(const Tensor &x, int h, int w, const Weight &weight,
                 const B *bias) {
  if (weight.shape.size() != 4 || weight.shape[1] != x.cols ||
      weight.shape[2] != weight.shape[3] || weight.shape[2] % 2 != 1 || h < 1 ||
      w < 1 || x.rows != h * w)
    throw std::invalid_argument("convolution dimensions");
  int kernel = weight.shape[2];
  if (kernel == 1)
    return linear(x, weight, bias);
  Tensor patches(h * w, x.cols * kernel * kernel);
  im2col<<<(patches.size() + 255) / 256, 256>>>(x.ptr, patches.ptr, h, w,
                                                x.cols, kernel);
  return linear(patches, weight, bias);
}
Tensor Ops::upsample(const Tensor &x, int h, int w) {
  Tensor y(h * w * 4, x.cols);
  upsample_kernel<<<(y.size() + 255) / 256, 256>>>(x.ptr, y.ptr, h, w, x.cols);
  hip_check(hipGetLastError());
  return y;
}
// CUDA-resident scalar promotion rounds dt to the bf16 velocity dtype. The
// product rounds again before the scheduler adds it to the fp32 sample.
__global__ void euler_kernel(B *sample, const B *velocity, float delta, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    B product = B(float(B(delta)) * float(velocity[i]));
    sample[i] = B(float(sample[i]) + float(product));
  }
}
void Ops::euler_step(Tensor &sample, const Tensor &velocity, float delta) {
  if (sample.rows != velocity.rows || sample.cols != velocity.cols)
    throw std::invalid_argument("scheduler tensor dimensions");
  euler_kernel<<<(sample.size() + 255) / 256, 256>>>(sample.ptr, velocity.ptr,
                                                  delta, sample.size());
  hip_check(hipGetLastError());
}
} // namespace krea_native
