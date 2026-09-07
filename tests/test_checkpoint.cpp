// CPU-only checkpoint regression tests. The transfer functions below use host
// memory; this executable never links HRX or opens a GPU device.
#include "../host/native_ops.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace gpu {
void *allocate(size_t bytes) {
  auto p = std::malloc(bytes ? bytes : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void release(void *p) noexcept { std::free(p); }
void copy(void *dst, const void *src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}
} // namespace gpu

using namespace krea_native;
namespace fs = std::filesystem;

static void write_file(const fs::path &path, const std::string &header,
                       const std::vector<float> &data = {}) {
  std::ofstream f(path, std::ios::binary);
  uint64_t length = header.size();
  f.write(reinterpret_cast<const char *>(&length), sizeof(length));
  f << header;
  f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}
static size_t descriptors() {
  return std::distance(fs::directory_iterator("/proc/self/fd"), fs::directory_iterator());
}
static size_t mappings(const fs::path &path) {
  std::ifstream f("/proc/self/maps");
  size_t count = 0;
  for (std::string line; std::getline(f, line);)
    count += line.find(path.string()) != std::string::npos;
  return count;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const auto path = fs::absolute(fs::path(argv[1]) / "checkpoint.safetensors");
  const std::vector<std::string> invalid = {
      "{", // invalid JSON
      R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      R"({"w":{"dtype":"F32","data_offsets":[0,0]}})"};
  for (const auto &header : invalid) {
    write_file(path, header);
    const auto before = descriptors();
    for (int i = 0; i < 3; ++i) {
      bool failed = false;
      try { SafeTensors file(path); }
      catch (const std::exception &) { failed = true; }
      assert(failed);
      assert(descriptors() == before);
      assert(mappings(path) == 0);
    }
  }
  // A header length beyond EOF must also release both acquired resources.
  {
    std::ofstream f(path, std::ios::binary);
    uint64_t length = 1024;
    f.write(reinterpret_cast<const char *>(&length), sizeof(length));
  }
  const auto before = descriptors();
  try { SafeTensors file(path); assert(false); }
  catch (const std::exception &) {}
  assert(descriptors() == before && mappings(path) == 0);

  // A scalar-shaped FP8 scale still needs four readable bytes.
  write_file(path, R"({"linear.weight":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]},"linear.weight_scale":{"dtype":"F32","shape":[],"data_offsets":[4,4]}})", {0});
  {
    SafeTensors file(path);
    bool failed = false;
    try { Weights weights(file, [](const std::string &name) { return name; }); }
    catch (const std::runtime_error &e) {
      failed = std::string(e.what()).find("float8 scale") != std::string::npos;
    }
    assert(failed);
  }

  // Two convolution channels, three temporal taps, two spatial elements.
  // Both storage views must contain the last tap of each channel.
  write_file(path, R"({"conv":{"dtype":"F32","shape":[2,1,3,1,2],"data_offsets":[0,48]},"norm":{"dtype":"F32","shape":[2],"data_offsets":[48,56]}})",
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, .125f, .25f});
  {
    SafeTensors file(path);
    Weights weights(file, [](const std::string &name) { return name; });
    const auto &conv = weights["conv"];
    assert((conv.shape == std::vector<int>{2, 1, 1, 2}));
    const std::vector<float> want{5, 6, 11, 12};
    const auto rounded = conv.t.download();
    for (size_t i = 0; i < want.size(); ++i) {
      assert(conv.f32[i] == want[i]);
      assert(float(rounded[i]) == want[i]);
    }
    const auto &norm = weights["norm"];
    assert(norm.f32[0] == .125f && float(norm.t.download()[0]) == .125f);
  }
  assert(mappings(path) == 0);
  fs::remove(path);
  std::cout << "PASS checkpoint cleanup, FP8 scale validation and float32 temporal weights\n";
}
