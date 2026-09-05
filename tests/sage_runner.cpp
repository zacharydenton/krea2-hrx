// Native attention benchmark including preprocessing; no Python runtime
// linkage.
#include "../host/sage.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes)); }
  ~Buffer() { (void)hipFree(p); }
  void read(const std::filesystem::path &path, size_t bytes) {
    if (std::filesystem::file_size(path) != bytes)
      throw std::runtime_error("wrong input file size");
    std::vector<char> v(bytes);
    std::ifstream f(path, std::ios::binary);
    f.read(v.data(), bytes);
    if (!f)
      throw std::runtime_error("input read failed");
    check(hipMemcpy(p, v.data(), bytes, hipMemcpyHostToDevice));
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
    int waves = symbol == "krea2_attention_sage_i4_fast" ? 8 :
                symbol == "krea2_attention_sage_i4_fast_prefetch" ? 4 : 0;
    if ((waves != 4 && waves != 8) || tokens < 16 || cap < tokens + 16 ||
        cap % 32 || heads != 4 * kv || kv < 1 || repeat < 1)
      throw std::runtime_error("invalid dimensions");
    std::filesystem::path dir(argv[7]);
    Buffer q(size_t(cap) * heads * 128 * 2), k(size_t(cap) * kv * 128 * 2),
        v(size_t(cap) * kv * 128 * 2), out(size_t(tokens) * heads * 128 * 2);
    q.read(dir / "q.bin", size_t(cap) * heads * 128 * 2);
    k.read(dir / "k.bin", size_t(cap) * kv * 128 * 2);
    v.read(dir / "v.bin", size_t(cap) * kv * 128 * 2);
    SagePreparation prep(tokens, cap, heads, kv);
    hipModule_t module;
    check(hipModuleLoad(&module, argv[1]));
    struct Module {
      hipModule_t m;
      ~Module() { (void)hipModuleUnload(m); }
    } module_owner{module};
    hipFunction_t function;
    check(hipModuleGetFunction(&function, module, argv[2]));
    alignas(16) char args[128];
    size_t bytes = 0;
    auto scalar = [&](int x) {
      memcpy(args + bytes, &x, 4);
      bytes += 4;
    };
    auto pointer = [&](void *p) {
      bytes = (bytes + 7) & ~size_t(7);
      memcpy(args + bytes, &p, 8);
      bytes += 8;
    };
    scalar(tokens);
    scalar(kv);
    pointer(prep.q4);
    pointer(prep.k4);
    pointer(prep.v_transposed);
    pointer(prep.qscale);
    pointer(prep.kscale);
    pointer(prep.correction);
    pointer(out.p);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &bytes,
                      HIP_LAUNCH_PARAM_END};
    auto launch = [&] {
      int rows = 16 * (waves / 4);
      check(hipModuleLaunchKernel(function, (tokens + rows - 1) / rows, kv, 1,
                                  waves * 32, 1, 1, 0, nullptr, nullptr,
                                  config));
    };
    prep.run(q.p, k.p, v.p);
    launch();
    check(hipDeviceSynchronize());
    hipEvent_t a, b, c;
    check(hipEventCreate(&a));
    check(hipEventCreate(&b));
    check(hipEventCreate(&c));
    float prep_ms = 0, attention_ms = 0;
    for (int i = 0; i < repeat; ++i) {
      check(hipEventRecord(a));
      prep.run(q.p, k.p, v.p);
      check(hipEventRecord(b));
      launch();
      check(hipEventRecord(c));
      check(hipEventSynchronize(c));
      float t;
      check(hipEventElapsedTime(&t, a, b));
      prep_ms += t;
      check(hipEventElapsedTime(&t, b, c));
      attention_ms += t;
    }
    check(hipEventDestroy(a));
    check(hipEventDestroy(b));
    check(hipEventDestroy(c));
    std::vector<uint16_t> result(size_t(tokens) * heads * 128);
    check(hipMemcpy(result.data(), out.p, result.size() * 2,
                    hipMemcpyDeviceToHost));
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
