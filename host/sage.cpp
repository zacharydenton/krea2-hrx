#include "sage.h"
#include "native_kernels.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>
using krea_native::native_launch;
void SagePreparation::allocate(void *&p, size_t bytes) {
  p = gpu::allocate(bytes);
  try {
    allocations_.push_back(p);
  } catch (...) {
    gpu::release(p);
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
    throw std::invalid_argument("unsupported Sage dimensions");
  try {
    allocate(q4, size_t(capacity) * heads * 64);
    allocate(k4, size_t(capacity) * kv_heads * 64);
    allocate(qscale, size_t(capacity) * heads * 4);
    allocate(kscale, size_t(capacity) * kv_heads * 4);
    allocate(kpartial_, size_t(tiles_) * kv_heads * 128 * 4);
    allocate(kmean_, size_t(kv_heads) * 128 * 4);
    allocate(qmean_, size_t(heads) * tiles_ * 128 * 4);
    allocate(qmean_half_, size_t(heads) * tiles_ * 128 * 2);
    allocate(centered_k_, size_t(kv_heads) * capacity * 128 * 2);
    gpu::zero(q4, size_t(capacity) * heads * 64);
    gpu::zero(qscale, size_t(capacity) * heads * 4);
    allocate(correction, size_t(heads) * tiles_ * capacity * 4);
    allocate(v_transposed, size_t(kv_heads) * capacity * 128 * 2);
    gpu::zero(v_transposed, size_t(kv_heads) * capacity * 128 * 2);
  } catch (...) {
    release();
    throw;
  }
}
void SagePreparation::release() noexcept {
  for (void *p : allocations_)
    gpu::release(p);
  allocations_.clear();
}
SagePreparation::~SagePreparation() { release(); }
void SagePreparation::run(const void *q, const void *k, const void *v,
                          bool profile) {
  if (!q || !k || !v)
    throw std::invalid_argument("missing Sage input");
  const size_t T = tokens_, C = capacity_, H = heads_, KV = kv_heads_,
               tiles = tiles_;
  auto start = std::chrono::steady_clock::now();
  auto mark = [&](const char *name) {
    if (!profile)
      return;
    gpu::synchronize();
    auto now = std::chrono::steady_clock::now();
    fprintf(stderr, "sage prep %s %.3f ms\n", name,
            std::chrono::duration<double, std::milli>(now - start).count());
    start = now;
  };
  mark("sync");
  {
    gpu::Args a;
    a.i32(T).ptr(k).ptr(kpartial_);
    native_launch("sage_key_partial",
                  {{"tokens", T},
                   {"heads", KV},
                   {"tiles", tiles},
                   {"xsize", C * KV * 128},
                   {"ysize", tiles * KV * 128}},
                  a, KV, tiles, 128);
    gpu::Args b;
    b.i32(T).ptr(kpartial_).ptr(kmean_);
    native_launch("sage_key_mean",
                  {{"tokens", T},
                   {"heads", KV},
                   {"tiles", tiles},
                   {"xsize", tiles * KV * 128},
                   {"ysize", KV * 128}},
                  b, KV, 1, 128);
  }
  mark("K mean");
  {
    gpu::Args a;
    a.i32(T).ptr(q).ptr(qmean_).ptr(qmean_half_);
    native_launch("sage_query_mean",
                  {{"tokens", T},
                   {"heads", H},
                   {"tiles", tiles},
                   {"xsize", C * H * 128},
                   {"ysize", H * tiles * 128}},
                  a, tiles, H, 128);
  }
  mark("Q mean");
  {
    gpu::Args a;
    a.i32(T).ptr(q).ptr(qmean_).ptr(q4).ptr(qscale);
    native_launch("sage_quant_q",
                  {{"tokens", T},
                   {"heads", H},
                   {"tiles", tiles},
                   {"capacity", C},
                   {"xsize", C * H * 128},
                   {"msize", H * tiles * 128},
                   {"psize", C * H * 32},
                   {"ssize", C * H}},
                  a, (T + 7) / 8, H);
    gpu::Args b;
    b.i32(T).ptr(k).ptr(kmean_).ptr(k4).ptr(kscale).ptr(centered_k_);
    native_launch("sage_quant_k",
                  {{"tokens", T},
                   {"heads", KV},
                   {"tiles", tiles},
                   {"capacity", C},
                   {"xsize", C * KV * 128},
                   {"msize", KV * 128},
                   {"psize", C * KV * 32},
                   {"ssize", C * KV}},
                  b, (C + 7) / 8, KV);
  }
  mark("quantization");
  {
    gpu::Args a;
    a.i32(T).ptr(v).ptr(v_transposed);
    native_launch("sage_transpose", {{"width", KV * 128}, {"row_capacity", C}},
                  a, (T + 31) / 32, KV * 128 / 32);
  }
  mark("V transpose");
  {
    const size_t m = tiles * 4, n = C, as = m * 128, bs = n * 128;
    gpu::Args a;
    a.i32(m).f32(1).ptr(qmean_half_).ptr(centered_k_).ptr(correction);
    native_launch("gemm_f16_f32_nt",
                  {{"m", m},
                   {"n", n},
                   {"k", 128},
                   {"asize", as * KV},
                   {"bsize", bs * KV},
                   {"csize", m * n * KV},
                   {"astride", as},
                   {"bstride", bs}},
                  a, (n + 63) / 64, KV * ((m + 63) / 64));
  }
  mark("correction GEMM");
}
