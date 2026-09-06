// Interleaved resident INT4 down projections. Each kernel sees the same inputs
// and number of residual updates; compare every output before and after timing.
#include "../host/gpu.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
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
    if (argc != 6)
      throw std::invalid_argument(
          "usage: down-bench BASELINE CANDIDATE M GROUP ROUNDS");
    int m = std::stoi(argv[3]), group = std::stoi(argv[4]),
        rounds = std::stoi(argv[5]);
    constexpr int n = 6144, k = 16384;
    if (m < 16 || m > 16896 || group < 2 || group > 4 || rounds < 10 ||
        rounds > 10000)
      throw std::invalid_argument("invalid benchmark dimensions");
    gpu::Kernel kernels[2] = {{argv[1], "krea2_gemm_i4_resid"},
                              {argv[2], "krea2_gemm_down_i4"}};
    Buffer a(size_t(m) * k / 2), w(size_t(n) * k / 2), ws(n * 4), as(m * 4),
        gate(n * 4);
    std::mt19937 random(42);
    auto packed = [&](Buffer &buffer, size_t count) {
      std::vector<uint8_t> values(count);
      for (auto &v : values)
        v = uint8_t(random());
      gpu::copy(buffer.p, values.data(), count);
    };
    auto scales = [&](Buffer &buffer, size_t count, bool signed_values) {
      std::vector<float> values(count);
      for (auto &v : values)
        v = (int(random() % 1000) - (signed_values ? 500 : 0)) * .00001f;
      gpu::copy(buffer.p, values.data(), count * 4);
    };
    packed(a, size_t(m) * k / 2);
    packed(w, size_t(n) * k / 2);
    scales(ws, n, false);
    scales(as, m, false);
    scales(gate, n, true);
    size_t bytes = size_t(m) * n * 2;
    Buffer output[2] = {Buffer(bytes), Buffer(bytes)};
    std::vector<uint16_t> initial(size_t(m) * n);
    for (auto &v : initial)
      v = uint16_t(0x3800 + random() % 2048 + (random() % 2) * 0x8000);
    gpu::Args args[2];
    for (int i = 0; i < 2; ++i) {
      gpu::copy(output[i].p, initial.data(), bytes);
      args[i]
          .i32(m)
          .ptr(a.p)
          .ptr(w.p)
          .ptr(ws.p)
          .ptr(as.p)
          .ptr(output[i].p)
          .ptr(gate.p);
    }
    auto launch = [&](int i) {
      int rows =
          i ? (m + 255) / 256 : ((m + 127) / 128 + group - 1) / group * group;
      kernels[i].launch(n / 128, rows, 256, args[i]);
    };
    auto compare = [&] {
      std::vector<uint16_t> reference(initial.size()),
          candidate(initial.size());
      gpu::copy(reference.data(), output[0].p, bytes);
      gpu::copy(candidate.data(), output[1].p, bytes);
      if (reference != candidate)
        throw std::runtime_error("candidate differs from baseline");
    };
    launch(0);
    launch(1);
    compare();
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
    compare();
    std::vector<double> ratios;
    for (int r = 0; r < rounds; ++r)
      ratios.push_back(times[0][r] / times[1][r]);
    std::sort(ratios.begin(), ratios.end());
    std::printf("{\"exact\":true,\"m\":%d,\"n\":%d,\"k\":%d,\"rounds\":%d,"
                "\"paired_median_speedup\":%.6f",
                m, n, k, rounds, ratios[rounds / 2]);
    for (int i = 0; i < 2; ++i) {
      std::sort(times[i].begin(), times[i].end());
      std::printf(
          ",\"%s_ms\":{\"min\":%.6f,\"p10\":%.6f,\"median\":%.6f,\"p90\":%.6f}",
          i ? "candidate" : "baseline", times[i][0], times[i][rounds / 10],
          times[i][rounds / 2], times[i][rounds * 9 / 10]);
    }
    std::puts("}");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
