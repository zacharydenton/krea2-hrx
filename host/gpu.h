#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

// gfx1151 dispatch through the public HRX C ABI. No HIP or GPU math library.
namespace gpu {
void initialize();
void *allocate(size_t bytes);
void release(void *pointer) noexcept;
void copy(void *destination, const void *source, size_t bytes);
void zero(void *destination, size_t bytes);
void synchronize();

struct Args {
  alignas(16) unsigned char bytes[256]{};
  size_t size = 0;
  template <class T> Args &add(T value) {
    size = (size + alignof(T) - 1) & ~(alignof(T) - 1);
    if (size + sizeof(T) > sizeof(bytes))
      throw std::runtime_error("kernel argument overflow");
    std::memcpy(bytes + size, &value, sizeof(T));
    size += sizeof(T);
    return *this;
  }
  Args &i32(int value) { return add(int32_t(value)); }
  Args &f32(float value) { return add(value); }
  Args &ptr(const void *value) { return add(value); }
};
class Kernel {
  struct Impl;
  std::shared_ptr<Impl> impl;

public:
  Kernel() = default;
  Kernel(const std::string &path, const std::string &symbol);
  void launch(unsigned gx, unsigned gy, unsigned bx, const void *args,
              size_t size, unsigned gz = 1, unsigned by = 1,
              unsigned bz = 1) const;
  void launch(unsigned gx, unsigned gy, unsigned bx, const Args &a) const {
    launch(gx, gy, bx, a.bytes, a.size);
  }
};
} // namespace gpu
