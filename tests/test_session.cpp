// Observe actual HRX allocation ownership across failed native constructors.
#include "../host/krea2.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <hrx_runtime.h>
#include <iostream>
#include <vector>

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
  const char *valid = "4 16 256 4 64 8 6144 16448 4 8\n";
  std::ofstream(dir / "launch.txt") << valid;
  // A checkpoint with one block's wq only: every other tensor is missing.
  const auto checkpoint = dir / "incomplete.safetensors";
  {
    const std::string header =
        R"({"blocks.0.attn.wq.weight":{"dtype":"I8","shape":[16,6144],"data_offsets":[0,98304]}})";
    std::ofstream f(checkpoint, std::ios::binary);
    uint64_t length = header.size();
    f.write((const char *)&length, 8);
    f << header;
    std::vector<char> zeros(98304);
    f.write(zeros.data(), zeros.size());
  }
  for (int i = 0; i < 3; ++i) {
    krea2_session *session = nullptr;
    char error[4096];
    assert(krea2_create(checkpoint.c_str(), dir.c_str(), 16, 1, &session,
                        error, sizeof(error)) == KREA2_ERROR);
    assert(!session);
    assert(std::string(error).find("missing tensor") != std::string::npos);
    assert(outstanding == 0);
  }
  // An incomplete checkpoint is rejected before any device allocation.
  assert(allocations == 0);
  // Invalid metadata must be rejected before reading weights: a raster group
  // of 0, an old version, the wrong wave count, a dense down-projection pitch,
  // a 128-row tile for int8 (the int8 family has only the 256-row one), a bad
  // attention width, and a missing field.
  krea2_session *session = nullptr;
  char error[4096];
  for (const char *metadata :
       {"3 16 256 4 64 8 6144 16448 4 8\n",
        "4 16 256 0 64 8 6144 16448 4 8\n", "2 16 1 64 8\n",
        "4 16 256 4 64 4 6144 16448 4 8\n", "4 16 256 4 64 8 6144 16384 4 8\n",
        "4 16 128 1 64 8 6144 16448 4 8\n", "4 16 256 4 64 8 6144 16448 6 8\n",
        "4 16 256 4 64 8 6144 16448 4\n"}) {
    std::ofstream(dir / "launch.txt") << metadata;
    assert(krea2_create(checkpoint.c_str(), dir.c_str(), 16, 1, &session,
                        error, sizeof(error)) == KREA2_INVALID_ARGUMENT);
    assert(allocations == 0 && outstanding == 0);
  }
  // A path that is not a checkpoint.
  std::ofstream(dir / "launch.txt") << valid;
  assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error,
                      sizeof(error)) == KREA2_ERROR);
  assert(std::string(error).find(".safetensors") != std::string::npos);
  assert(allocations == 0 && outstanding == 0);
  std::cout << "PASS native constructor cleanup and launch metadata\n";
}
