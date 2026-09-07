#pragma once
#include "gpu.h"
#include "safetensors.h"
#include <cmath>
#include <functional>
#include <limits>
#include "native_kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace krea_native {
// Host representation of IEEE BF16, round-to-nearest ties-to-even.
struct B {
  uint16_t bits = 0;
  B() = default;
  explicit B(float value) {
    uint32_t u;
    std::memcpy(&u, &value, 4);
    bits = uint16_t(((u & 0x7fffffff) > 0x7f800000)
                        ? ((u >> 16) | 0x40)
                        : ((u + 0x7fff + ((u >> 16) & 1)) >> 16));
  }
  operator float() const {
    uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
};
static_assert(sizeof(B) == 2);
using json = nlohmann::json;
// Temporary buffers are reused only within one serialized pipeline session.
// Every operation uses the ordered HRX stream, so subsequent uses are ordered
// after earlier kernels without a release synchronization between each
// operation.
struct BufferPool {
  static constexpr size_t limit = 512ull * 1024 * 1024;
  size_t cached = 0;
  std::multimap<size_t, void *> free;
  ~BufferPool() {
    for (auto [bytes, pointer] : free)
      (void)gpu::release(pointer);
  }
  void *take(size_t &bytes) {
    auto it = free.lower_bound(bytes);
    if (it != free.end() && it->first - bytes <= bytes) {
      bytes = it->first;
      void *pointer = it->second;
      cached -= bytes;
      free.erase(it);
      return pointer;
    }
    void *pointer = nullptr;
    pointer = gpu::allocate(bytes);
    return pointer;
  }
  void put(void *pointer, size_t bytes) noexcept {
    if (bytes <= limit - cached) {
      try {
        free.emplace(bytes, pointer);
        cached += bytes;
        return;
      } catch (...) {
      } // Allocation failure must not escape a buffer deleter.
    }
    (void)gpu::release(pointer);
  }
};
inline thread_local std::shared_ptr<BufferPool> active_pool;
struct PoolScope {
  std::shared_ptr<BufferPool> previous;
  explicit PoolScope(const std::shared_ptr<BufferPool> &pool)
      : previous(std::move(active_pool)) {
    active_pool = pool;
  }
  ~PoolScope() { active_pool = std::move(previous); }
};
inline std::shared_ptr<void> device_storage(size_t bytes) {
  if (active_pool) {
    auto pool = active_pool;
    void *pointer = pool->take(bytes);
    return {pointer, [pool, bytes](void *p) { pool->put(p, bytes); }};
  }
  void *pointer = nullptr;
  pointer = gpu::allocate(bytes);
  return {pointer, [](void *p) { (void)gpu::release(p); }};
}
struct Tensor {
  int rows = 0, cols = 0;
  std::shared_ptr<void> storage;
  B *ptr = nullptr;
  Tensor() = default;
  Tensor(int r, int c) : rows(r), cols(c) {
    if (r < 1 || c < 1)
      throw std::invalid_argument("empty native tensor");
    storage = device_storage(size_t(r) * c * sizeof(B));
    ptr = (B *)storage.get();
  }
  size_t size() const { return size_t(rows) * cols; }
  Tensor view(int r, int c, size_t offset = 0) const {
    if (r < 1 || c < 1 || offset > size() || size_t(r) * c > size() - offset)
      throw std::out_of_range("tensor view");
    Tensor t;
    t.rows = r;
    t.cols = c;
    t.storage = storage;
    t.ptr = ptr + offset;
    return t;
  }
  static Tensor upload(const std::vector<B> &v, int r, int c) {
    if (v.size() != size_t(r) * c)
      throw std::invalid_argument("upload size");
    Tensor t(r, c);
    gpu::copy(t.ptr, v.data(), v.size() * 2);
    return t;
  }
  std::vector<B> download() const {
    std::vector<B> v(size());
    gpu::copy(v.data(), ptr, size() * 2);
    return v;
  }
};
inline float fp8_e4m3_to_float(uint8_t v) {
  const int sign = v >> 7, exponent = (v >> 3) & 15, mantissa = v & 7;
  float value;
  if (exponent == 15 && mantissa == 7)
    value = std::numeric_limits<float>::quiet_NaN();
  else if (exponent == 0)
    value = std::ldexp(float(mantissa), -9); // subnormal: m/8 * 2^-6
  else
    value = std::ldexp(1.f + mantissa / 8.f, exponent - 7);
  return sign ? -value : value;
}
inline uint16_t float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

struct Weight {
  Tensor t; // bf16 storage (ptr is null for a float32 tensor)
  std::vector<int> shape;
  const float *f32 = nullptr; // float32 storage, when the checkpoint keeps it
  mutable std::shared_ptr<void> f32_cache;
  std::shared_ptr<void> bf16_copy; // a float32 tensor's rounded bf16 view (t)
  size_t count() const { return size_t(t.rows) * t.cols; }
  // A float32 tensor is consumed as bf16 by everything but the norms (ComfyUI
  // and diffusers run those tensors in bf16): make that view from host data.
  void bf16_of_f32(const char *host_f32, size_t n) {
    std::vector<uint16_t> v(n);
    const float *f = reinterpret_cast<const float *>(host_f32);
    for (size_t i = 0; i < n; ++i)
      v[i] = float_to_bf16(f[i]);
    bf16_copy = device_storage(n * 2);
    gpu::copy(bf16_copy.get(), v.data(), n * 2);
    t.ptr = static_cast<B *>(bf16_copy.get());
  }
  // The tensor as float32 on the device: the bundle's float32 data, or a
  // lossless upcast of its bf16 data made once. The norm kernels take this.
  const float *as_f32() const {
    if (f32)
      return f32;
    if (!f32_cache) {
      auto b = t.download();
      std::vector<float> v(b.size());
      for (size_t i = 0; i < b.size(); ++i)
        v[i] = float(b[i]);
      auto storage = device_storage(v.size() * sizeof(float));
      gpu::copy(storage.get(), v.data(), v.size() * sizeof(float));
      f32_cache = storage;
    }
    return static_cast<const float *>(f32_cache.get());
  }
};
struct Weights {
  std::map<std::string, Weight> values;
  // A ComfyUI checkpoint's tensors as they are, renamed (an empty name skips
  // the tensor): bf16 and float32 keep their dtype, float8 e4m3fn rows with a
  // per-tensor weight_scale are dequantised to bf16, and 5-D causal-conv
  // weights are reduced to their last temporal tap (the single-image frame).
  Weights(const SafeTensors &file,
          const std::function<std::string(const std::string &)> &rename) {
    struct Item {
      std::string name;
      const SafeTensors::Entry *entry, *scale = nullptr;
      bool f32 = false, last_tap = false;
      size_t count = 0, bytes = 0, offset = 0;
      std::vector<int> shape;
    };
    auto ends_with = [](const std::string &s, const std::string &t) {
      return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    std::vector<Item> items;
    size_t total = 0;
    for (const auto &[key, entry] : file.entries) {
      if (ends_with(key, ".comfy_quant") || ends_with(key, ".weight_scale"))
        continue;
      std::string name = rename(key);
      if (name.empty())
        continue;
      Item it{name, &entry};
      for (auto d : entry.shape)
        it.shape.push_back(int(d));
      if (it.shape.size() == 5) { // [O][I][T][H][W] -> the last tap
        it.last_tap = true;
        it.shape = {it.shape[0], it.shape[1], it.shape[3], it.shape[4]};
      }
      it.count = 1;
      for (int d : it.shape)
        it.count *= size_t(d);
      const size_t elements = entry.elements();
      if (entry.dtype == "F32") {
        it.f32 = true;
        it.bytes = it.count * 4;
      } else if (entry.dtype == "BF16") {
        it.bytes = it.count * 2;
      } else if (entry.dtype == "F8_E4M3") {
        it.scale = &file.at(key + "_scale");
        if (it.scale->dtype != "F32" || it.scale->elements() != 1)
          throw std::runtime_error("unsupported float8 scale for " + key);
        it.bytes = it.count * 2;
      } else {
        throw std::runtime_error("unsupported tensor dtype " + entry.dtype +
                                 " for " + key + " in " + file.path);
      }
      const size_t element_bytes = entry.dtype == "F32" ? 4 : entry.dtype == "BF16" ? 2 : 1;
      if (entry.bytes != elements * element_bytes)
        throw std::runtime_error("tensor size mismatch for " + key);
      it.offset = total;
      total += (it.bytes + 255) / 256 * 256;
      items.push_back(std::move(it));
    }
    auto storage = device_storage(total);
    std::vector<char> staging;
    for (const auto &it : items) {
      char *dst = static_cast<char *>(storage.get()) + it.offset;
      const char *src = file.data(*it.entry);
      if (!it.last_tap && !it.scale) {
        gpu::copy(dst, src, it.bytes);
      } else {
        staging.resize(it.bytes);
        const auto &shape = it.entry->shape;
        const size_t taps = it.last_tap ? size_t(shape[2]) : 1,
                     plane = it.last_tap ? size_t(shape[3]) * size_t(shape[4]) : it.count,
                     blocks = it.last_tap ? size_t(shape[0]) * size_t(shape[1]) : 1;
        const float scale = it.scale ? *reinterpret_cast<const float *>(file.data(*it.scale)) : 1.f;
        const size_t esz = it.scale ? 1 : it.f32 ? 4 : 2;
        for (size_t b = 0; b < blocks; ++b) {
          const char *in = src + ((b * taps) + (taps - 1)) * plane * esz;
          char *out = staging.data() + b * plane * (it.f32 ? 4 : 2);
          if (it.scale) {
            auto *o = reinterpret_cast<uint16_t *>(out);
            for (size_t i = 0; i < plane; ++i)
              o[i] = float_to_bf16(fp8_e4m3_to_float(uint8_t(in[i])) * scale);
          } else {
            std::memcpy(out, in, plane * esz);
          }
        }
        gpu::copy(dst, staging.data(), it.bytes);
      }
      Tensor view;
      view.rows = 1;
      view.cols = int(it.count);
      view.storage = storage;
      Weight weight{std::move(view), it.shape};
      if (it.f32) {
        weight.f32 = reinterpret_cast<const float *>(dst);
        weight.bf16_of_f32(it.last_tap ? staging.data() : src, it.count);
      } else {
        weight.t.ptr = reinterpret_cast<B *>(dst);
      }
      values.emplace(it.name, std::move(weight));
    }
  }
  const Weight &operator[](const std::string &name) const {
    return values.at(name);
  }
  bool has(const std::string &name) const { return values.count(name); }
};
struct Ops {
  Tensor linear(const Tensor &x, const Weight &w, const B *bias = nullptr);
  // RMSNorm with float32 scales (mode 0: x * (1 + w), 1: x * w, 2: GroupNorm-style).
  Tensor norm(const Tensor &x, const Weight &weight, int mode = 0,
              float eps = 1e-5f);
  Tensor unary(const Tensor &x, int op);
  Tensor binary(const Tensor &x, const Tensor &y, int op);
  Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, int batch,
                   int tokens, int heads, int kv, int dim, bool causal = false);
  Tensor rope(const Tensor &x, int tokens, int heads, float theta,
              bool interleaved = false);
  Tensor conv(const Tensor &x, int height, int width, const Weight &w,
              const B *bias);
  Tensor upsample(const Tensor &x, int height, int width);
  void euler_step(Tensor &sample, const Tensor &velocity, float delta);
  // cond = cond + scale * (cond - uncond), diffusers' bf16 rounding per operation.
  void guidance(Tensor &cond, const Tensor &uncond, float scale);
};
} // namespace krea_native
