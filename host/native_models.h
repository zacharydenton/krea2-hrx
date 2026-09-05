#pragma once
#include "native_ops.h"
#include "native_tokenizer.h"
namespace krea_native {
struct Models {
  Ops ops;
  Weights text, transformer, vae;
  Tokenizer tokenizer;
  explicit Models(const std::string &root);
  static constexpr size_t modulation_elements = 28 * 6 * 6144;
  Tensor lin(const Tensor &x, const Weights &w, const std::string &p) {
    return ops.linear(x, w[p + ".weight"],
                      w.has(p + ".bias") ? w[p + ".bias"].t.ptr : nullptr);
  }
  Tensor encode(const std::vector<int32_t> &ids);
  Tensor text_fusion(const Tensor &taps);
  std::pair<Tensor, Tensor> time(float timestep);
  Tensor image_in(const Tensor &latents) {
    return lin(latents, transformer, "first");
  }
  Tensor final(const Tensor &x, const Tensor &temb);
  // Device float32 buffer, including bf16 rounding of each table addition.
  std::shared_ptr<float> modulation(const Tensor &mod);
  std::vector<uint8_t> decode(const Tensor &packed, int height, int width);

private:
  Tensor block_tables;
  Tensor fusion_block(const Tensor &x, const std::string &p, int batch,
                      int tokens);
  Tensor residual(const Tensor &x, const std::string &p, int h, int w);
  Tensor decode_tile(const Tensor &x, int h, int w);
};
} // namespace krea_native
