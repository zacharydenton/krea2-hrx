#pragma once
#include <cmath>
#include <fstream>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace krea_native {
using B = hip_bfloat16;
using json = nlohmann::json;
inline void hip_check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
inline void blas_check(hipblasStatus_t e) {
  if (e != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("hipBLAS error " + std::to_string(e));
}
// Temporary buffers are reused only within one serialized pipeline session.
// Every operation uses HIP's default stream, so subsequent uses are ordered
// after earlier kernels without a hipFree synchronization between each
// operation.
struct BufferPool {
  static constexpr size_t limit = 512ull * 1024 * 1024;
  size_t cached = 0;
  std::multimap<size_t, void *> free;
  ~BufferPool() {
    for (auto [bytes, pointer] : free)
      (void)hipFree(pointer);
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
    hip_check(hipMalloc(&pointer, bytes));
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
    (void)hipFree(pointer);
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
  hip_check(hipMalloc(&pointer, bytes));
  return {pointer, [](void *p) { (void)hipFree(p); }};
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
    hip_check(hipMemcpy(t.ptr, v.data(), v.size() * 2, hipMemcpyHostToDevice));
    return t;
  }
  std::vector<B> download() const {
    std::vector<B> v(size());
    hip_check(hipMemcpy(v.data(), ptr, size() * 2, hipMemcpyDeviceToHost));
    return v;
  }
};
struct Weight {
  Tensor t;
  std::vector<int> shape;
};
struct Weights {
  std::map<std::string, Weight> values;
  explicit Weights(const std::string &directory) {
    std::ifstream meta(directory + "/weights.json");
    if (!meta)
      throw std::runtime_error("cannot read " + directory + "/weights.json");
    json entries;
    meta >> entries;
    std::ifstream f(directory + "/weights.bin",
                    std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("cannot open weights: " + directory);
    size_t bytes = f.tellg();
    f.seekg(0);
    for (auto &[name, entry] : entries.items()) {
      auto dims = entry["shape"].get<std::vector<int>>();
      if (dims.empty())
        throw std::runtime_error("empty weight shape: " + name);
      size_t count = 1;
      for (int n : dims) {
        if (n < 1 || count > size_t(INT32_MAX) / size_t(n))
          throw std::runtime_error("invalid weight dimensions");
        count *= n;
      }
      size_t offset = entry["offset"], length = entry["bytes"];
      if (length != count * 2 || offset > bytes || length > bytes - offset ||
          count > INT32_MAX)
        throw std::runtime_error("invalid weight span: " + name);
      std::vector<B> data(count);
      f.seekg(offset);
      f.read((char *)data.data(), length);
      if (!f)
        throw std::runtime_error("truncated weights: " + name);
      values.emplace(name, Weight{Tensor::upload(data, 1, int(count)), dims});
    }
  }
  const Weight &operator[](const std::string &name) const {
    return values.at(name);
  }
  bool has(const std::string &name) const { return values.count(name); }
};
struct Ops {
  hipblasHandle_t handle{};
  Ops() { blas_check(hipblasCreate(&handle)); }
  ~Ops() { hipblasDestroy(handle); }
  Tensor linear(const Tensor &x, const Weight &w, const B *bias = nullptr);
  Tensor norm(const Tensor &x, const Tensor &weight, int mode = 0,
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
};
} // namespace krea_native
