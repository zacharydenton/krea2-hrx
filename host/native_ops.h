#pragma once
#include "gpu.h"
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
    const auto path = directory + "/weights.bin";
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error("cannot open " + path);
    struct File {
      int fd;
      ~File() { close(fd); }
    } file{fd};
    struct stat info;
    if (fstat(fd, &info) || info.st_size <= 0 || info.st_size % sizeof(B))
      throw std::runtime_error("invalid weight file: " + path);
    size_t bytes = size_t(info.st_size);
    // Validate every view before allocating or uploading the bundle.
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
      if (length != count * 2 || offset % sizeof(B) || offset > bytes ||
          length > bytes - offset || count > INT32_MAX)
        throw std::runtime_error("invalid weight span: " + name);
      Tensor view;
      view.rows = 1;
      view.cols = int(count);
      values.emplace(name, Weight{std::move(view), std::move(dims)});
    }
    void *mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED)
      throw std::runtime_error("cannot map " + path);
    struct Mapping {
      void *p;
      size_t bytes;
      ~Mapping() { munmap(p, bytes); }
    } mapping{mapped, bytes};
    auto storage = device_storage(bytes);
    gpu::copy(storage.get(), mapped, bytes);
    for (auto &[name, weight] : values) {
      size_t offset = entries.at(name).at("offset");
      weight.t.storage = storage;
      weight.t.ptr =
          reinterpret_cast<B *>(static_cast<char *>(storage.get()) + offset);
    }
  }

  const Weight &operator[](const std::string &name) const {
    return values.at(name);
  }
  bool has(const std::string &name) const { return values.count(name); }
};
struct Ops {
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
