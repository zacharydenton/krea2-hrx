// Observe actual HRX allocation ownership across failed native constructors.
#include "../host/krea2.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <hrx_runtime.h>
#include <iostream>

static int outstanding = 0, allocations = 0;
extern "C" hrx_status_t __real_hrx_buffer_allocate(hrx_stream_t, size_t,
                                                   hrx_memory_type_t,
                                                   hrx_buffer_usage_t,
                                                   hrx_buffer_t *);
extern "C" void __real_hrx_buffer_release(hrx_buffer_t);
extern "C" hrx_status_t __wrap_hrx_buffer_allocate(hrx_stream_t stream,
                                                   size_t bytes,
                                                   hrx_memory_type_t type,
                                                   hrx_buffer_usage_t usage,
                                                   hrx_buffer_t *out) {
  auto status = __real_hrx_buffer_allocate(stream, bytes, type, usage, out);
  if (hrx_status_is_ok(status)) {
    ++outstanding;
    ++allocations;
  }
  return status;
}
extern "C" void __wrap_hrx_buffer_release(hrx_buffer_t p) {
  if (p)
    --outstanding;
  __real_hrx_buffer_release(p);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const auto dir = std::filesystem::path(argv[1]) / "session-fixture";
  std::filesystem::create_directory(dir);
  std::ofstream(dir / "launch.txt") << "3 16 128 1 64 8 6144 16512\n";
  std::ofstream(dir / "manifest.txt") << "";
  std::ofstream(dir / "weights.bin") << "incomplete weights";
  for (int i = 0; i < 3; ++i) {
    krea2_session *session = nullptr;
    char error[4096];
    assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                        sizeof(error)) == KREA2_ERROR);
    assert(!session);
    assert(std::string(error).find("missing tensor") != std::string::npos);
    assert(outstanding == 0);
  }
  assert(allocations == 3);
  // Invalid metadata must be rejected before allocating weights.
  std::ofstream(dir / "launch.txt") << "3 16 128 0 64 8 6144 16512\n";
  krea2_session *session = nullptr;
  char error[4096];
  assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                      sizeof(error)) == KREA2_INVALID_ARGUMENT);
  assert(allocations == 3 && outstanding == 0);
  std::ofstream(dir / "launch.txt") << "2 16 1 64 8\n";
  assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                      sizeof(error)) == KREA2_INVALID_ARGUMENT);
  assert(allocations == 3 && outstanding == 0);
  std::ofstream(dir / "launch.txt") << "3 16 128 1 64 4 6144 16512\n";
  assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                      sizeof(error)) == KREA2_INVALID_ARGUMENT);
  assert(allocations == 3 && outstanding == 0);
  // A dense down-projection pitch or a 256-row tile below 4096 tokens is a
  // bundle built by a different rule.
  for (const char *metadata : {"3 16 128 1 64 8 6144 16384\n", "3 16 256 4 64 8 6144 16512\n"}) {
    std::ofstream(dir / "launch.txt") << metadata;
    assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                        sizeof(error)) == KREA2_INVALID_ARGUMENT);
    assert(allocations == 3 && outstanding == 0);
  }
  std::cout << "PASS native constructor cleanup and launch metadata\n";
}
