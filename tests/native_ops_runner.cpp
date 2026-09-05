// Test-only access to individual embedded Loom kernels through the HRX adapter.
#include "../host/native_kernels.h"
#include <cstdio>
#include <nlohmann/json.hpp>
extern "C" void *test_alloc(size_t size) { return gpu::allocate(size); }
extern "C" void test_free(void *p) { gpu::release(p); }
extern "C" void test_copy(void *dst, const void *src, size_t size) {
  gpu::copy(dst, src, size);
}
extern "C" int test_run(const char *name, const char *json, const void *args,
                        size_t size, unsigned gx, unsigned gy,
                        unsigned threads) {
  try {
    gpu::Args a;
    if (size > sizeof(a.bytes))
      throw std::invalid_argument("argument size");
    std::memcpy(a.bytes, args, size);
    a.size = size;
    auto config = nlohmann::json::parse(json).get<krea_native::Config>();
    krea_native::native_launch(name, config, a, gx, gy, threads);
    gpu::synchronize();
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
