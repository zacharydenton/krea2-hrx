// Constant logits have exactly known probabilities. Repeated resident
// dispatches expose LDS read/write races without a model, random data, Torch,
// or HIP.
#include "../host/native_ops.h"
#include <iostream>

using namespace krea_native;

void check(int tokens, bool causal) {
  const int rows = 8 * tokens;
  const size_t count = size_t(rows) * tokens;
  const int repeats = tokens == 256 ? 256 : 32;
  std::vector<float> scores(count, 20.f);
  std::vector<B> expected(count), actual(count);
  for (int row = 0; row < rows; ++row) {
    const int valid = causal ? row % tokens + 1 : tokens;
    for (int col = 0; col < tokens; ++col)
      expected[size_t(row) * tokens + col] = B(col < valid ? 1.f / valid : 0.f);
  }
  auto input = device_storage(count * sizeof(float));
  auto output = device_storage(count * sizeof(B));
  gpu::copy(input.get(), scores.data(), count * sizeof(float));
  gpu::Args args;
  args.i32(rows).ptr(input.get()).ptr(output.get());
  const std::string name = causal ? "softmax_causal" : "softmax";
  for (int run = 0; run < repeats; ++run) {
    native_launch(name, {{"xsize", count}, {"tokens", size_t(tokens)}}, args,
                  rows);
    gpu::copy(actual.data(), output.get(), count * sizeof(B));
    if (std::memcmp(actual.data(), expected.data(), count * sizeof(B))) {
      for (size_t i = 0; i < count; ++i)
        if (actual[i].bits != expected[i].bits)
          throw std::runtime_error(
              name + " tokens=" + std::to_string(tokens) + " repeat=" +
              std::to_string(run) + " row=" + std::to_string(i / tokens) +
              " col=" + std::to_string(i % tokens) +
              " expected=" + std::to_string(float(expected[i])) +
              " actual=" + std::to_string(float(actual[i])));
    }
  }
  std::cout << "PASS " << name << " tokens=" << tokens << " " << repeats
            << " exact repeats\n"
            << std::flush;
}

int main() {
  try {
    for (int tokens : {256, 33, 257, 1024})
      for (bool causal : {false, true})
        check(tokens, causal);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
