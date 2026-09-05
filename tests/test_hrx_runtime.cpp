// Exercise real dispatch and inspect late-loaded dependencies in a C++ process.
#include "../host/native_kernels.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
int main() {
  try {
    std::vector<uint16_t> input(1009, 0x3f80), output(input.size());
    void *a = gpu::allocate(input.size() * 2);
    void *b = gpu::allocate(input.size() * 2);
    gpu::copy(a, input.data(), input.size() * 2);
    gpu::Args args;
    args.i32(input.size()).ptr(a).ptr(b);
    krea_native::native_launch("unary_one", {}, args, 4);
    gpu::copy(output.data(), b, output.size() * 2);
    for (auto value : output)
      assert(value == 0x4000);
    // Interior spans must be checked before submitting a copy.
    bool rejected = false;
    try {
      gpu::copy(output.data(), static_cast<char *>(b) + 2, output.size() * 2);
    } catch (const std::out_of_range &) {
      rejected = true;
    }
    assert(rejected);
    gpu::release(a);
    gpu::release(b);
    std::ifstream maps("/proc/self/maps");
    assert(maps);
    bool hrx = false, hsa = false;
    std::string line;
    while (std::getline(maps, line)) {
      hrx |= line.find("libhrx.so") != std::string::npos;
      hsa |= line.find("libhsa-runtime64") != std::string::npos;
      for (const char *forbidden :
           {"libamdhip64", "libhipblas", "librocblas", "libtorch", "libpython",
            "libpcre2", "libicu", "libcrypto"})
        assert(line.find(forbidden) == std::string::npos);
    }
    assert(hrx && hsa);
    std::cout << "PASS HRX dispatch, span validation and loaded-library audit: "
                 "no HIP, BLAS or framework runtime\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
