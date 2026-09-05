#pragma once
#include "native_ops.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace krea_native {
// Opt-in synchronized wall-clock stage timing; disabled during normal
// inference.
struct Profile {
  bool enabled;
  const char *scope;
  std::chrono::steady_clock::time_point last;
  explicit Profile(const char *name) : scope(name) {
    const char *value = std::getenv("KREA2_NATIVE_PROFILE");
    enabled = value && std::strcmp(value, "1") == 0;
    if (enabled) {
      gpu::synchronize();
      last = std::chrono::steady_clock::now();
    }
  }
  void mark(const char *stage) {
    if (!enabled)
      return;
    gpu::synchronize();
    auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "native profile %s / %s: %.3f ms\n", scope, stage,
                 std::chrono::duration<double, std::milli>(now - last).count());
    last = now;
  }
};
} // namespace krea_native
