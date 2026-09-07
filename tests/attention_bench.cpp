// Paired fp16 attention on resident inputs. Python supplies an independent CPU
// oracle for selected query rows; compare every output with the baseline too.
#include "../host/gpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

namespace {
constexpr int heads = 48, kv_heads = 12, dim = 128;
struct Buffer {
  void *p;
  explicit Buffer(size_t bytes) : p(gpu::allocate(bytes)) {}
  Buffer(const Buffer &) = delete;
  ~Buffer() { gpu::release(p); }
};

template <class T>
std::vector<T> read(const std::string &path, size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != std::streamoff(count * sizeof(T)))
    throw std::runtime_error("wrong input size: " + path);
  file.seekg(0);
  std::vector<T> values(count);
  if (!file.read(reinterpret_cast<char *>(values.data()), count * sizeof(T)))
    throw std::runtime_error("cannot read " + path);
  return values;
}

struct Error {
  double squared = 0, reference_squared = 0, maximum = 0;
  bool close = true;
  void add(float actual, float expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected))
      throw std::runtime_error("nonfinite attention output");
    double delta = double(actual) - expected;
    squared += delta * delta;
    reference_squared += double(expected) * expected;
    maximum = std::max(maximum, std::abs(delta));
    close &= std::abs(delta) <= .002 + .002 * std::abs(expected);
  }
  double relative_rms() const {
    return std::sqrt(squared / std::max(reference_squared, 1e-30));
  }
  void check(const char *name) const {
    std::fprintf(stderr, "%s: max_abs=%.6g relative_rms=%.6g\n", name,
                 maximum, relative_rms());
    if (!close || relative_rms() > .002)
      throw std::runtime_error(std::string(name) + " failed accuracy check");
  }
};
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 11 && argc != 12)
      throw std::invalid_argument(
          "usage: attention-bench BASE_HSACO BASE_SYMBOL CAND_HSACO CAND_SYMBOL "
          "TOKENS CAPACITY QTILES ROUNDS ORACLE_ROWS INPUT_DIR [TRANSPOSE_HSACO]");
    int tokens = std::stoi(argv[5]), capacity = std::stoi(argv[6]),
        qtiles = std::stoi(argv[7]), rounds = std::stoi(argv[8]),
        oracle_rows = std::stoi(argv[9]);
    if (tokens < 16 || tokens > 65536 || capacity < tokens + 16 ||
        capacity > 65600 || capacity % 64 ||
        (qtiles != 1 && qtiles != 2 && qtiles != 4) ||
        rounds < 10 || rounds > 10000 || oracle_rows < 1 || oracle_rows > tokens)
      throw std::invalid_argument("invalid benchmark dimensions");
    std::string input_dir = argv[10];
    auto q = read<_Float16>(input_dir + "/q.bin", size_t(capacity) * heads * dim);
    auto k = read<_Float16>(input_dir + "/k.bin", size_t(capacity) * kv_heads * dim);
    auto v = read<_Float16>(input_dir + "/v.bin", k.size());
    auto rows = read<uint32_t>(input_dir + "/rows.bin", oracle_rows);
    auto want = read<float>(input_dir + "/want.bin", size_t(oracle_rows) * heads * dim);
    for (auto row : rows)
      if (row >= unsigned(tokens))
        throw std::invalid_argument("oracle row outside sequence");
    gpu::Kernel kernels[2] = {{argv[1], argv[2]}, {argv[3], argv[4]}};
    gpu::Kernel transpose;
    if (argc == 12)
      transpose = gpu::Kernel(argv[11], "krea2_sage_transpose");
    Buffer q_device(q.size() * 2), k_device(k.size() * 2), v_device(v.size() * 2);
    Buffer vt_device(argc == 12 ? v.size() * 2 : 2);
    if (argc == 12)
      gpu::zero(vt_device.p, v.size() * 2);
    gpu::copy(q_device.p, q.data(), q.size() * 2);
    gpu::copy(k_device.p, k.data(), k.size() * 2);
    gpu::copy(v_device.p, v.data(), v.size() * 2);
    size_t count = size_t(tokens) * heads * dim, bytes = count * 2;
    Buffer outputs[2] = {Buffer(bytes), Buffer(bytes)};
    gpu::Args args[2];
    // Poison the output to catch missing rows/channels in publication.
    std::vector<_Float16> poison(count, _Float16(std::numeric_limits<float>::quiet_NaN()));
    for (int i = 0; i < 2; ++i) {
      gpu::copy(outputs[i].p, poison.data(), bytes);
      // token_count is an 8-byte Loom index. KV heads belong to the grid,
      // not the launch arguments (putting them here corrupts its high word).
      args[i].add(int64_t(tokens)).ptr(q_device.p).ptr(k_device.p)
          .ptr(i && argc == 12 ? vt_device.p : v_device.p).ptr(outputs[i].p);
    }
    gpu::Args transpose_args;
    transpose_args.add(int64_t(tokens)).ptr(v_device.p).ptr(vt_device.p);
    auto launch = [&](int i) {
      if (i && argc == 12)
        transpose.launch((tokens + 31) / 32, kv_heads * dim / 32, 256, transpose_args);
      int tiles = i ? qtiles : 1;
      kernels[i].launch((tokens + 16 * tiles - 1) / (16 * tiles), kv_heads,
                        128 * tiles, args[i]);
    };
    auto check = [&] {
      std::vector<_Float16> actual[2] = {std::vector<_Float16>(count),
                                       std::vector<_Float16>(count)};
      for (int i = 0; i < 2; ++i) {
        gpu::copy(actual[i].data(), outputs[i].p, bytes);
        Error oracle;
        for (size_t r = 0; r < rows.size(); ++r)
          for (int c = 0; c < heads * dim; ++c)
            oracle.add(float(actual[i][size_t(rows[r]) * heads * dim + c]),
                       want[r * heads * dim + c]);
        oracle.check(i ? "candidate vs CPU oracle" : "baseline vs CPU oracle");
      }
      Error full;
      for (size_t c = 0; c < count; ++c)
        full.add(float(actual[1][c]), float(actual[0][c]));
      full.check("candidate vs baseline, all outputs");
    };
    launch(0);
    launch(1);
    gpu::synchronize();
    check();
    for (int i = 0; i < 5; ++i) {
      launch(0);
      launch(1);
    }
    gpu::synchronize();
    std::vector<double> times[2], ratios;
    for (int r = 0; r < rounds; ++r) {
      for (int j = 0; j < 2; ++j) {
        int i = j ^ (r % 2);
        auto start = std::chrono::steady_clock::now();
        launch(i);
        gpu::synchronize();
        times[i].push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start).count());
      }
      ratios.push_back(times[0].back() / times[1].back());
    }
    check();
    std::sort(ratios.begin(), ratios.end());
    std::printf("{\"correct\":true,\"tokens\":%d,\"rounds\":%d,"
                "\"paired_speedup\":{\"p10\":%.6f,\"median\":%.6f,\"p90\":%.6f}",
                tokens, rounds, ratios[rounds / 10], ratios[rounds / 2],
                ratios[rounds * 9 / 10]);
    for (int i = 0; i < 2; ++i) {
      std::sort(times[i].begin(), times[i].end());
      std::printf(",\"%s_ms\":{\"min\":%.6f,\"p10\":%.6f,\"median\":%.6f,\"p90\":%.6f}",
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
