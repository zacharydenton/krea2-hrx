// Interleaved BF16 GEMMs on identical resident buffers. Timing includes HRX
// submission and synchronization; correctness is checked before timing.
#include "../host/gpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct Buffer {
  void *p;
  explicit Buffer(size_t bytes) : p(gpu::allocate(bytes)) {}
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { gpu::release(p); }
};

int main(int argc, char **argv) {
  try {
    if (argc != 10)
      throw std::invalid_argument("usage: gemm-bench BASELINE CANDIDATE SYMBOL "
                                  "M N K TILE_M TILE_N ROUNDS");
    int m = std::stoi(argv[4]), n = std::stoi(argv[5]), k = std::stoi(argv[6]),
        tile = std::stoi(argv[7]), columns = std::stoi(argv[8]),
        rounds = std::stoi(argv[9]);
    if (m < 1 || n < 1 || k < 1 || rounds < 10 || rounds > 10000 ||
        (tile != 64 && tile != 128) || (columns != 64 && columns != 128) ||
        size_t(m) * n > 268435456 || size_t(m) * k > 268435456 ||
        size_t(n) * k > 268435456)
      throw std::invalid_argument("invalid benchmark dimensions");
    gpu::Kernel kernels[2] = {{argv[1], "krea2_gemm_bf16_bf16_nt"},
                              {argv[2], argv[3]}};
    Buffer a(size_t(m) * k * 2), b(size_t(n) * k * 2);
    std::mt19937 random(42);
    auto initialize = [&](Buffer &buffer, size_t count) {
      std::vector<uint16_t> values(count);
      for (auto &v : values) {
        float f = (int(random() % 8192) - 4096) / 1024.f;
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        v = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
      }
      gpu::copy(buffer.p, values.data(), count * 2);
    };
    initialize(a, size_t(m) * k);
    initialize(b, size_t(n) * k);
    size_t bytes = size_t(m) * n * 2;
    Buffer output[2] = {Buffer(bytes), Buffer(bytes)};
    gpu::Args args[2];
    for (int i = 0; i < 2; ++i)
      args[i].i32(m).f32(1).ptr(a.p).ptr(b.p).ptr(output[i].p);
    auto launch = [&](int i) {
      int rows = i ? tile : 64;
      int cols = i ? columns : 64;
      kernels[i].launch((n + cols - 1) / cols, (m + rows - 1) / rows, 256,
                        args[i]);
    };
    launch(0);
    launch(1);
    std::vector<uint16_t> reference(size_t(m) * n), candidate(reference.size());
    gpu::copy(reference.data(), output[0].p, bytes);
    gpu::copy(candidate.data(), output[1].p, bytes);
    if (reference != candidate)
      throw std::runtime_error("candidate differs from baseline");
    for (int i = 0; i < 5; ++i) {
      launch(0);
      launch(1);
    }
    gpu::synchronize();
    std::vector<double> times[2];
    for (int r = 0; r < rounds; ++r) {
      for (int j = 0; j < 2; ++j) {
        int i = j ^ (r % 2);
        auto start = std::chrono::steady_clock::now();
        launch(i);
        gpu::synchronize();
        times[i].push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count());
      }
    }
    std::vector<double> ratios;
    for (int r = 0; r < rounds; ++r)
      ratios.push_back(times[0][r] / times[1][r]);
    std::sort(ratios.begin(), ratios.end());
    std::printf("{\"exact\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"rounds\":%d,"
                "\"paired_median_speedup\":%.6f",
                m, n, k, rounds, ratios[rounds / 2]);
    for (int i = 0; i < 2; ++i) {
      std::sort(times[i].begin(), times[i].end());
      std::printf(",\"%s_ms\":{\"min\":%.6f,\"p10\":%.6f,\"median\":%.6f,"
                  "\"p90\":%.6f}",
                  i ? "candidate" : "baseline", times[i][0],
                  times[i][rounds / 10], times[i][rounds / 2],
                  times[i][rounds * 9 / 10]);
    }
    std::puts("}");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
