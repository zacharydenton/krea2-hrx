// Small independent harness for the production scheduler operation; no models.
#include "../host/native_ops.h"
#include "../host/native_schedule.h"
#include <filesystem>
#include <iostream>
using namespace krea_native;
int main(int argc, char **argv) {
  try {
    if (argc != 2) return 2;
    std::filesystem::path root(argv[1]);
    constexpr int rows = 5050, width = 256;
    auto read = [&](const char *name) {
      std::ifstream f(root / name, std::ios::binary);
      std::vector<float> data(rows * width);
      if (!f.read((char *)data.data(), data.size() * 4))
        throw std::runtime_error("cannot read scheduler fixture");
      std::vector<B> bf(data.size());
      for (size_t i = 0; i < bf.size(); ++i) bf[i] = B(data[i]);
      return Tensor::upload(bf, rows, width);
    };
    auto samples = read("samples.bin"), velocity = read("velocity.bin");
    Ops ops;
    std::vector<float> sigmas(rows);
    int row = 0;
    for (int steps = 1; steps <= 100; ++steps)
      for (int step = 0; step < steps; ++step, ++row) {
        sigmas[row] = scheduler_sigma(step, steps);
        auto sample = samples.view(1, width, size_t(row) * width);
        ops.euler_step(sample, velocity.view(1, width, size_t(row) * width),
                       scheduler_sigma(step + 1, steps) - scheduler_sigma(step, steps));
      }
    std::ofstream(root / "sigmas.bin", std::ios::binary).write((char *)sigmas.data(), sigmas.size() * 4);
    auto bf = samples.download();
    std::vector<float> result(bf.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = float(bf[i]);
    std::ofstream f(root / "result.bin", std::ios::binary);
    if (!f.write((char *)result.data(), result.size() * 4))
      throw std::runtime_error("cannot write scheduler result");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
