// Native attention benchmark including preprocessing; no Python runtime
// linkage.
#include "../host/gpu.h"
#include "../host/sage.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t bytes) { p = gpu::allocate(bytes); }
  ~Buffer() { gpu::release(p); }
  void read(const std::filesystem::path &path, size_t bytes) {
    if (std::filesystem::file_size(path) != bytes)
      throw std::runtime_error("wrong input file size");
    std::vector<char> v(bytes);
    std::ifstream f(path, std::ios::binary);
    f.read(v.data(), bytes);
    if (!f)
      throw std::runtime_error("input read failed");
    gpu::copy(p, v.data(), bytes);
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 9)
      throw std::runtime_error("usage: sage-runner HSACO SYMBOL TOKENS HEADS "
                               "KV CAP DIR REPEAT");
    int tokens = std::stoi(argv[3]), heads = std::stoi(argv[4]),
        kv = std::stoi(argv[5]), cap = std::stoi(argv[6]),
        repeat = std::stoi(argv[8]);
    const std::string symbol = argv[2];
    // krea2_attention_sage_i{4,8}_fast[_prefetch]: the code width selects the
    // preparation, the suffix the wave count.
    int bits = symbol.rfind("krea2_attention_sage_i4_fast", 0) == 0   ? 4
               : symbol.rfind("krea2_attention_sage_i8_fast", 0) == 0 ? 8
                                                                      : 0;
    std::string tail = symbol.substr(std::min(symbol.size(), size_t(28)));
    int waves = tail.empty() ? 8 : tail == "_prefetch" ? 4 : 0;
    if (!bits || (waves != 4 && waves != 8) || tokens < 16 ||
        cap < tokens + 16 || cap % 32 || heads != 4 * kv || kv < 1 ||
        repeat < 1)
      throw std::runtime_error("invalid dimensions");
    std::filesystem::path dir(argv[7]);
    Buffer q(size_t(cap) * heads * 128 * 2), k(size_t(cap) * kv * 128 * 2),
        v(size_t(cap) * kv * 128 * 2), out(size_t(tokens) * heads * 128 * 2);
    q.read(dir / "q.bin", size_t(cap) * heads * 128 * 2);
    k.read(dir / "k.bin", size_t(cap) * kv * 128 * 2);
    v.read(dir / "v.bin", size_t(cap) * kv * 128 * 2);
    SagePreparation prep(tokens, cap, heads, kv, bits);
    gpu::Kernel kernel(argv[1], argv[2]);
    gpu::Args args;
    args.i32(tokens)
        .i32(kv)
        .ptr(prep.q4)
        .ptr(prep.k4)
        .ptr(prep.v_transposed)
        .ptr(prep.qscale)
        .ptr(prep.kscale)
        .ptr(prep.correction)
        .ptr(out.p);
    auto launch = [&] {
      int rows = 16 * (waves / 4);
      kernel.launch((tokens + rows - 1) / rows, kv, waves * 32, args);
    };
    prep.run(q.p, k.p, v.p);
    launch();
    gpu::synchronize();
    // Synchronized wall time includes HRX submission overhead.
    using Clock = std::chrono::steady_clock;
    double prep_ms = 0, attention_ms = 0;
    for (int i = 0; i < repeat; ++i) {
      auto a = Clock::now();
      prep.run(q.p, k.p, v.p);
      gpu::synchronize();
      auto b = Clock::now();
      launch();
      gpu::synchronize();
      auto c = Clock::now();
      prep_ms += std::chrono::duration<double, std::milli>(b - a).count();
      attention_ms += std::chrono::duration<double, std::milli>(c - b).count();
    }
    std::vector<uint16_t> result(size_t(tokens) * heads * 128);
    gpu::copy(result.data(), out.p, result.size() * 2);
    std::ofstream f(dir / "out.bin", std::ios::binary);
    f.write((char *)result.data(), result.size() * 2);
    if (!f)
      throw std::runtime_error("output write failed");
    std::cout << "{\"prepare_ms\":" << prep_ms / repeat
              << ",\"attention_ms\":" << attention_ms / repeat
              << ",\"total_ms\":" << (prep_ms + attention_ms) / repeat << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
