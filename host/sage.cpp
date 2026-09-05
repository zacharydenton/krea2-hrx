#include "sage.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <stdexcept>
namespace {
void check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
void check_blas(hipblasStatus_t e) {
  if (e != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Sage hipBLAS error " + std::to_string(e));
}
__global__ void key_partial(const __half *k, float *partial, int tokens,
                            int heads) {
  int head = blockIdx.x, tile = blockIdx.y, channel = threadIdx.x;
  float sum = 0;
  for (int s = tile * 64; s < min((tile + 1) * 64, tokens); ++s)
    sum += float(k[(size_t(s) * heads + head) * 128 + channel]);
  partial[(size_t(tile) * heads + head) * 128 + channel] = sum;
}
__global__ void key_mean(const float *partial, float *mean, int tokens,
                         int heads) {
  int head = blockIdx.x, channel = threadIdx.x;
  float sum = 0;
  for (int tile = 0; tile < (tokens + 63) / 64; ++tile)
    sum += partial[(size_t(tile) * heads + head) * 128 + channel];
  mean[head * 128 + channel] = sum / tokens;
}
// Keep the same sequential float32 mean as the original fused kernel. Splitting
// mean and packing avoids a 33 KiB LDS allocation and a 512-thread barrier
// tail.
__global__ void query_mean(const __half *q, float *mean, __half *half_mean,
                           int tokens, int heads, int tiles) {
  int tile = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
  int count = min(64, tokens - tile * 64);
  float sum = 0;
  for (int r = 0; r < count; ++r)
    sum += float(q[(size_t(tile * 64 + r) * heads + head) * 128 + channel]);
  float m = sum / count;
  size_t at = (size_t(head) * tiles + tile) * 128 + channel;
  mean[at] = m;
  half_mean[at] = __float2half(m);
}
__global__ void quantize_queries(const __half *q, const float *mean,
                                 uint8_t *packed, float *scales, int tokens,
                                 int heads, int tiles) {
  int row = blockIdx.x * 8 + threadIdx.x / 32, head = blockIdx.y;
  int lane = threadIdx.x % 32, channel = lane * 4;
  if (row >= tokens)
    return; // whole waves; padded rows were zeroed at allocation
  float values[4], maximum = 0;
  for (int j = 0; j < 4; ++j) {
    values[j] = float(q[(size_t(row) * heads + head) * 128 + channel + j]) -
                mean[(size_t(head) * tiles + row / 64) * 128 + channel + j];
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, delta, 32));
  float scale = fmaxf(maximum, 1e-30f) / 7.f;
  uint16_t bits = 0;
  for (int j = 0; j < 4; ++j)
    bits |= (max(-7, min(7, int(nearbyintf(values[j] / scale)))) & 15)
            << (4 * j);
  if (!lane)
    scales[size_t(row) * heads + head] = scale;
  ((uint16_t *)packed)[(size_t(row) * heads + head) * 32 + lane] = bits;
}
__global__ void quantize_keys(const __half *input, const float *mean,
                              uint8_t *packed, float *scales, __half *centered,
                              int tokens, int heads, int capacity) {
  int row = blockIdx.x, head = blockIdx.y, channel = threadIdx.x * 4;
  float values[4] = {}, maximum = 0;
  for (int j = 0; j < 4; ++j) {
    if (row < tokens) {
      values[j] =
          float(input[(size_t(row) * heads + head) * 128 + channel + j]) -
          mean[head * 128 + channel + j];
    }
    centered[(size_t(head) * capacity + row) * 128 + channel + j] =
        __float2half(values[j]);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, delta, 32));
  float scale = fmaxf(maximum, 1e-30f) / 7.f;
  uint16_t bits = 0;
  for (int j = 0; j < 4; ++j) {
    int code = max(-7, min(7, int(nearbyintf(values[j] / scale))));
    bits |= (code & 15) << (4 * j);
  }
  if (!channel)
    scales[size_t(row) * heads + head] = scale;
  ((uint16_t *)packed)[(size_t(row) * heads + head) * 32 + threadIdx.x] = bits;
}
// Once per block, instead of scattering V into LDS in every attention tile.
__global__ void transpose_values(const __half *input, __half *output,
                                 int tokens, int capacity, int channels) {
  __shared__ __half tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    tile[threadIdx.y + j][threadIdx.x] =
        y + j < tokens ? input[size_t(y + j) * channels + x] : __float2half(0);
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8)
    output[size_t(y + j) * capacity + x] = tile[threadIdx.x][threadIdx.y + j];
}
} // namespace
void SagePreparation::allocate(void *&p, size_t bytes) {
  check(hipMalloc(&p, bytes));
  try {
    allocations_.push_back(p);
  } catch (...) {
    (void)hipFree(p);
    p = nullptr;
    throw;
  }
}
SagePreparation::SagePreparation(int tokens, int capacity, int heads,
                                 int kv_heads)
    : tokens_(tokens), capacity_(capacity), heads_(heads), kv_heads_(kv_heads),
      tiles_((tokens + 63) / 64) {
  if (tokens < 16 || tokens > 16896 || capacity < (tokens + 63) / 64 * 64 ||
      capacity % 32 || heads < 1 || kv_heads < 1 || heads / kv_heads != 4 ||
      heads % kv_heads)
    throw std::invalid_argument("unsupported Sage attention dimensions");
  try {
    check_blas(hipblasCreate(&handle_));
    allocate(q4, size_t(capacity) * heads * 64);
    allocate(k4, size_t(capacity) * kv_heads * 64);
    allocate(qscale, size_t(capacity) * heads * 4);
    allocate(kscale, size_t(capacity) * kv_heads * 4);
    allocate(kpartial_, size_t((tokens + 63) / 64) * kv_heads * 128 * 4);
    allocate(kmean_, size_t(kv_heads) * 128 * 4);
    allocate(qmean_, size_t(heads) * tiles_ * 128 * 4);
    allocate(qmean_half_, size_t(heads) * tiles_ * 128 * 2);
    allocate(centered_k_, size_t(kv_heads) * capacity * 128 * 2);
    check(hipMemset(q4, 0, size_t(capacity) * heads * 64));
    check(hipMemset(qscale, 0, size_t(capacity) * heads * 4));
    allocate(correction, size_t(heads) * tiles_ * capacity * 4);
    allocate(v_transposed, size_t(kv_heads) * capacity * 128 * 2);
  } catch (...) {
    release();
    throw;
  }
}
void SagePreparation::release() noexcept {
  for (void *p : allocations_)
    (void)hipFree(p);
  if (handle_)
    (void)hipblasDestroy(handle_);
}
SagePreparation::~SagePreparation() { release(); }
void SagePreparation::run(const void *q, const void *k, const void *v,
                          bool profile) {
  if (!q || !k || !v)
    throw std::invalid_argument("missing Sage input");
  auto start = std::chrono::steady_clock::now();
  auto mark = [&](const char *name) {
    if (!profile)
      return;
    check(hipDeviceSynchronize());
    auto now = std::chrono::steady_clock::now();
    fprintf(stderr, "sage prep %s %.3f ms\n", name,
            std::chrono::duration<double, std::milli>(now - start).count());
    start = now;
  };
  mark("sync");
  key_partial<<<dim3(kv_heads_, (tokens_ + 63) / 64), 128>>>(
      (const __half *)k, (float *)kpartial_, tokens_, kv_heads_);
  key_mean<<<kv_heads_, 128>>>((float *)kpartial_, (float *)kmean_, tokens_,
                               kv_heads_);
  mark("K mean");
  query_mean<<<dim3(tiles_, heads_), 128>>>((const __half *)q, (float *)qmean_,
                                            (__half *)qmean_half_, tokens_,
                                            heads_, tiles_);
  mark("Q mean");
  quantize_queries<<<dim3((tokens_ + 7) / 8, heads_), 256>>>(
      (const __half *)q, (float *)qmean_, (uint8_t *)q4, (float *)qscale,
      tokens_, heads_, tiles_);
  mark("Q quant");
  quantize_keys<<<dim3(capacity_, kv_heads_), 32>>>(
      (const __half *)k, (float *)kmean_, (uint8_t *)k4, (float *)kscale,
      (__half *)centered_k_, tokens_, kv_heads_, capacity_);
  check(hipGetLastError());
  mark("K quant + centered copy");
  {
    transpose_values<<<dim3(kv_heads_ * 128 / 32, capacity_ / 32),
                       dim3(32, 8)>>>((const __half *)v, (__half *)v_transposed,
                                      tokens_, capacity_, kv_heads_ * 128);
    check(hipGetLastError());
    mark("V transpose");
  }
  float alpha = 1, beta = 0;
  // Four consecutive query heads share each K head. Fold their mean tiles
  // into GEMM's N dimension so centered K is stored only once per KV head.
  int grouped_tiles = tiles_ * (heads_ / kv_heads_);
  check_blas(hipblasGemmStridedBatchedEx(
      handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, capacity_, grouped_tiles, 128,
      &alpha, centered_k_, HIP_R_16F, 128, int64_t(capacity_) * 128,
      qmean_half_, HIP_R_16F, 128, int64_t(grouped_tiles) * 128, &beta,
      correction, HIP_R_32F, capacity_, int64_t(capacity_) * grouped_tiles,
      kv_heads_, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  mark("correction GEMM");
}
