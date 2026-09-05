// Internal component oracle harness; not part of the public inference API.
#include "../host/native_models.h"
#include <filesystem>
#include <iostream>
using namespace krea_native;
static Tensor read(const std::filesystem::path &p, int cols) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("missing fixture");
  size_t count = size_t(f.tellg()) / 4;
  f.seekg(0);
  std::vector<float> v(count);
  f.read((char *)v.data(), count * 4);
  std::vector<B> b(count);
  for (size_t i = 0; i < count; ++i)
    b[i] = B(v[i]);
  return Tensor::upload(b, count / cols, cols);
}
static void write(const std::filesystem::path &p, const Tensor &t) {
  auto b = t.download();
  std::vector<float> v(b.size());
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = float(b[i]);
  std::ofstream f(p, std::ios::binary);
  f.write((char *)v.data(), v.size() * 4);
}
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      return 2;
    Models model(argv[1]);
    std::filesystem::path dir = argv[2];
    write(dir / "condition.bin",
          model.text_fusion(read(dir / "text.bin", 2560)));
    auto [e, m] = model.time(.75f);
    write(dir / "temb.bin", e);
    write(dir / "mod.bin", m);
    auto modulation = model.modulation(m);
    std::vector<float> tables(Models::modulation_elements);
    gpu::copy(tables.data(), modulation.get(), tables.size() * 4);
    std::ofstream mod_file(dir / "block_mod.bin", std::ios::binary);
    mod_file.write((char *)tables.data(), tables.size() * 4);
    write(dir / "final.bin", model.final(read(dir / "hidden.bin", 6144), e));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
