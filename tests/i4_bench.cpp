// Paired, interleaved timing of two INT4 GEMM kernels on the same resident
// inputs, for any of the three epilogues (plain, resid, swiglu). Both kernels
// see identical operands and the same number of residual updates; outputs are
// compared bit for bit before and after timing whenever the operands agree
// (they differ only when the two sides are given different K, which is how the
// operand-pitch falsifier runs the same kernel at two row pitches).
//
//   i4-bench BASELINE.hsaco CANDIDATE.hsaco key=value...
//     mode=plain|resid|swiglu   symbol=... symbol_cand=...
//     m= n= k= [k_cand=] [k_stride= k_stride_cand=]   (k_stride: operand row pitch)
//     tile=128|256 tile_cand= group= group_cand= [grid_group= grid_group_cand=] rounds=
//
// The launch grid is n/128 x ceil(ceil(m/tile)/group)*group, the padded raster
// form; a kernel that shortens its own tail takes group=1.
#include "../host/gpu.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
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
    if (argc < 3)
      throw std::invalid_argument(
          "usage: i4-bench BASELINE CANDIDATE key=value...");
    std::map<std::string, std::string> options;
    for (int i = 3; i < argc; ++i) {
      std::string item = argv[i];
      auto eq = item.find('=');
      if (eq == std::string::npos || eq == 0)
        throw std::invalid_argument("expected key=value: " + item);
      options[item.substr(0, eq)] = item.substr(eq + 1);
    }
    auto get = [&](const char *key, const std::string &fallback) {
      auto it = options.find(key);
      return it == options.end() ? fallback : it->second;
    };
    auto geti = [&](const char *key, int fallback) {
      return std::stoi(get(key, std::to_string(fallback)));
    };
    std::string mode = get("mode", "plain");
    if (mode != "plain" && mode != "resid" && mode != "swiglu")
      throw std::invalid_argument("mode must be plain, resid or swiglu");
    int m = geti("m", 4115), n = geti("n", 6144), rounds = geti("rounds", 80);
    int k[2] = {geti("k", 6144), 0};
    k[1] = geti("k_cand", k[0]);
    int k_stride[2] = {geti("k_stride", k[0]), 0};
    k_stride[1] = geti("k_stride_cand", k[1]);
    int tile[2] = {geti("tile", 128), 0};
    tile[1] = geti("tile_cand", tile[0]);
    int group[2] = {geti("group", 4), 0};
    group[1] = geti("group_cand", group[0]);
    // grid rows are padded to a multiple of grid_group; a kernel that shortens
    // its own raster tail takes grid_group=1 with its full m_group config
    int grid_group[2] = {geti("grid_group", group[0]), 0};
    grid_group[1] = geti("grid_group_cand", group[1]);
    std::string symbol[2] = {get("symbol", "krea2_gemm_i4"), ""};
    symbol[1] = get("symbol_cand", symbol[0]);
    if (m < 1 || m > 16896 || n < 128 || n > 32768 || n % 128 ||
        rounds < 10 || rounds > 10000)
      throw std::invalid_argument("invalid benchmark dimensions");
    for (int i = 0; i < 2; ++i)
      if (k[i] < 128 || k[i] > 65536 || k[i] % 128 || k_stride[i] < k[i] ||
          k_stride[i] % 128 || (tile[i] != 128 && tile[i] != 256) ||
          group[i] < 1 || group[i] > 4 || grid_group[i] < 1 ||
          grid_group[i] > 4)
        throw std::invalid_argument("invalid kernel shape");
    bool same_operands = k[0] == k[1] && k_stride[0] == k_stride[1];
    gpu::Kernel kernels[2] = {{argv[1], symbol[0]}, {argv[2], symbol[1]}};

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
    // Operands per side: identical bytes when the pitches agree, else each
    // side gets its own rows (the same seed, so the K prefix matches).
    std::vector<std::unique_ptr<Buffer>> a, w;
    for (int i = 0; i < (same_operands ? 1 : 2); ++i) {
      a.emplace_back(new Buffer(size_t(m) * k_stride[i] / 2));
      w.emplace_back(new Buffer(size_t(n) * k_stride[i] / 2));
      random.seed(42);
      packed(*a.back(), size_t(m) * k_stride[i] / 2);
      packed(*w.back(), size_t(n) * k_stride[i] / 2);
    }
    Buffer ws(n * 4), as(m * 4), gate(n * 4);
    scales(ws, n, false);
    scales(as, m, false);
    scales(gate, n, true);
    int out_columns = mode == "swiglu" ? n / 2 : n;
    size_t bytes = size_t(m) * out_columns * 2;
    Buffer output[2] = {Buffer(bytes), Buffer(bytes)};
    std::vector<uint16_t> initial(size_t(m) * out_columns);
    for (auto &v : initial)
      v = uint16_t(0x3800 + random() % 2048 + (random() % 2) * 0x8000);
    gpu::Args args[2];
    for (int i = 0; i < 2; ++i) {
      gpu::copy(output[i].p, initial.data(), bytes);
      int side = same_operands ? 0 : i;
      args[i]
          .i32(m)
          .ptr(a[side]->p)
          .ptr(w[side]->p)
          .ptr(ws.p)
          .ptr(as.p)
          .ptr(output[i].p);
      if (mode == "resid")
        args[i].ptr(gate.p);
    }
    auto launch = [&](int i) {
      int tiles = (m + tile[i] - 1) / tile[i];
      int rows = (tiles + grid_group[i] - 1) / grid_group[i] * grid_group[i];
      kernels[i].launch(n / 128, rows, 256, args[i]);
    };
    auto compare = [&] {
      if (!same_operands)
        return;
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
    std::printf("{\"mode\":\"%s\",\"exact\":%s,\"m\":%d,\"n\":%d,\"rounds\":%d,"
                "\"paired_median_speedup\":%.6f",
                mode.c_str(), same_operands ? "true" : "null", m, n, rounds,
                ratios[rounds / 2]);
    for (int i = 0; i < 2; ++i) {
      std::sort(times[i].begin(), times[i].end());
      double median = times[i][rounds / 2];
      double tops = 2.0 * m * n * k[i] / (median * 1e-3) / 1e12;
      std::printf(",\"%s\":{\"k\":%d,\"k_stride\":%d,\"tile\":%d,\"group\":%d,"
                  "\"min_ms\":%.6f,\"p10_ms\":%.6f,\"median_ms\":%.6f,"
                  "\"p90_ms\":%.6f,\"median_tops\":%.3f}",
                  i ? "candidate" : "baseline", k[i], k_stride[i], tile[i],
                  group[i], times[i][0], times[i][rounds / 10], median,
                  times[i][rounds * 9 / 10], tops);
    }
    std::puts("}");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
