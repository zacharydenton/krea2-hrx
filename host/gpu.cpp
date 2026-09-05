#include "gpu.h"
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <hrx_runtime.h>
#include <map>
#include <mutex>

namespace gpu {
namespace {
void check(hrx_status_t status) {
  if (hrx_status_is_ok(status))
    return;
  char *message = nullptr;
  size_t size = 0;
  hrx_status_ignore(hrx_status_to_string(status, &message, &size));
  std::string text = message ? std::string(message, size) : "unknown HRX error";
  hrx_status_free_message(message);
  hrx_status_ignore(status);
  throw std::runtime_error(text);
}
struct Allocation {
  hrx_buffer_t buffer;
  size_t size;
};
struct Runtime {
  bool owns_gpu = false;
  hrx_device_t device = nullptr;
  hrx_stream_t stream = nullptr;
  std::map<uintptr_t, Allocation> allocations;
  std::mutex mutex;
  Runtime() {
    try {
      // HRX loads the GPU driver directly. Prefer the provider packaged beside
      // this library, without changing the caller's library search path.
      if (!std::getenv("IREE_HAL_AMDGPU_LIBHSA_PATH")) {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void *>(&initialize), &info) &&
            info.dli_fname) {
          auto path = std::filesystem::absolute(info.dli_fname).parent_path() /
                      "runtime/libhsa-runtime64.so.1";
          if (std::filesystem::exists(path))
            setenv("IREE_HAL_AMDGPU_LIBHSA_PATH", path.c_str(), 0);
        }
      }
      auto initialization = hrx_gpu_initialize(0);
      if (hrx_status_code(initialization) == HRX_STATUS_ALREADY_EXISTS)
        hrx_status_ignore(initialization);
      else {
        check(initialization);
        owns_gpu = true;
      }
      int count = 0;
      check(hrx_gpu_device_count(&count));
      for (int i = 0; i < count; ++i) {
        hrx_device_t candidate;
        check(hrx_gpu_device_get(i, &candidate));
        char arch[64]{};
        try {
          check(hrx_device_get_property(
              candidate, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
        } catch (...) {
          hrx_device_release(candidate);
          throw;
        }
        if (std::string(arch) == "gfx1151") {
          device = candidate;
          break;
        }
        hrx_device_release(candidate);
      }
      if (!device)
        throw std::runtime_error("gfx1151 GPU required");
      check(hrx_stream_create(device, 0, &stream));
    } catch (...) {
      cleanup();
      throw;
    }
  }
  ~Runtime() { cleanup(); }
  void cleanup() noexcept {
    if (stream) {
      hrx_status_ignore(hrx_stream_synchronize(stream));
      hrx_stream_release(stream);
    }
    for (auto &[address, allocation] : allocations)
      hrx_buffer_release(allocation.buffer);
    if (device)
      hrx_device_release(device);
    if (owns_gpu)
      hrx_status_ignore(hrx_gpu_shutdown());
  }
  hrx_buffer_ref_t find(const void *pointer, size_t bytes) {
    uintptr_t p = reinterpret_cast<uintptr_t>(pointer);
    auto i = allocations.upper_bound(p);
    if (i == allocations.begin())
      return {};
    --i;
    const size_t offset = p - i->first;
    if (offset >= i->second.size)
      return {};
    if (bytes > i->second.size - offset)
      throw std::out_of_range("GPU buffer span");
    return {i->second.buffer, offset, bytes};
  }
};
Runtime &runtime() {
  static Runtime r;
  return r;
}
} // namespace
void initialize() { (void)runtime(); }
void synchronize() {
  auto &r = runtime();
  std::lock_guard lock(r.mutex);
  check(hrx_stream_synchronize(r.stream));
}
void *allocate(size_t bytes) {
  auto &r = runtime();
  std::lock_guard lock(r.mutex);
  hrx_buffer_t buffer = nullptr;
  bytes = bytes ? bytes : 4;
  check(hrx_buffer_allocate(
      r.stream, bytes,
      HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_HOST_VISIBLE,
      HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &buffer));
  void *p = nullptr;
  try {
    check(hrx_buffer_get_device_ptr(buffer, &p));
    r.allocations.emplace(reinterpret_cast<uintptr_t>(p),
                          Allocation{buffer, bytes});
  } catch (...) {
    hrx_buffer_release(buffer);
    throw;
  }
  return p;
}
void release(void *p) noexcept {
  if (!p)
    return;
  try {
    auto &r = runtime();
    std::lock_guard lock(r.mutex);
    auto i = r.allocations.find(reinterpret_cast<uintptr_t>(p));
    if (i == r.allocations.end())
      return;
    check(hrx_stream_synchronize(r.stream));
    hrx_buffer_release(i->second.buffer);
    r.allocations.erase(i);
  } catch (...) {
  }
}
void copy(void *dst, const void *src, size_t bytes) {
  auto &r = runtime();
  std::lock_guard lock(r.mutex);
  auto d = r.find(dst, bytes), s = r.find(src, bytes);
  if (d.buffer && s.buffer) {
    check(hrx_stream_copy_buffer(r.stream, s.buffer, s.offset, d.buffer,
                                 d.offset, bytes));
    check(hrx_stream_execution_barrier(r.stream));
  } else {
    check(hrx_stream_synchronize(r.stream));
    if (d.buffer)
      check(hrx_synchronous_h2d(r.device, src, d.buffer, d.offset, bytes));
    else if (s.buffer)
      check(hrx_synchronous_d2h(r.device, s.buffer, s.offset, dst, bytes));
    else
      throw std::invalid_argument("copy requires a GPU allocation");
  }
}
void zero(void *dst, size_t bytes) {
  auto &r = runtime();
  std::lock_guard lock(r.mutex);
  auto d = r.find(dst, bytes);
  if (!d.buffer)
    throw std::invalid_argument("zero requires a GPU allocation");
  const uint8_t value = 0;
  check(hrx_stream_fill_buffer(r.stream, d.buffer, d.offset, bytes, &value, 1));
  check(hrx_stream_execution_barrier(r.stream));
}
struct Kernel::Impl {
  hrx_executable_t executable = nullptr;
  uint32_t ordinal = 0;
  ~Impl() {
    if (executable) {
      try {
        synchronize();
      } catch (...) {
      }
      hrx_executable_release(executable);
    }
  }
};
Kernel::Kernel(const std::string &path, const std::string &symbol)
    : impl(std::make_shared<Impl>()) {
  auto &r = runtime();
  check(hrx_executable_load_file(r.device, path.c_str(), "amdgpu", "gfx1151",
                                 &impl->executable));
  check(hrx_executable_lookup_export_by_name(impl->executable, symbol.c_str(),
                                             &impl->ordinal));
}
void Kernel::launch(unsigned gx, unsigned gy, unsigned bx, const void *args,
                    size_t size, unsigned gz, unsigned by, unsigned bz) const {
  if (!impl || !gx || !gy || !gz || !bx || !by || !bz ||
      uint64_t(bx) * by * bz > 1024 || size > 256)
    throw std::invalid_argument("invalid GPU launch");
  auto &r = runtime();
  std::lock_guard lock(r.mutex);
  hrx_dispatch_config_t config{{gx, gy, gz}, {bx, by, bz}, 32};
  check(hrx_stream_dispatch(r.stream, impl->executable, impl->ordinal, &config,
                            args, size, nullptr, 0,
                            HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS));
  check(hrx_stream_execution_barrier(r.stream));
}
} // namespace gpu
