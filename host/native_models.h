#pragma once
#include "native_ops.h"
#include "native_tokenizer.h"
namespace krea_native {
// ComfyUI's models directory layout: <models>/diffusion_models/<checkpoint>
// beside <models>/text_encoders/qwen3vl_4b_{bf16,fp8_scaled}.safetensors and
// <models>/vae/qwen_image_vae.safetensors.
struct ComfyFiles {
  std::string checkpoint, text_encoder, vae;
  bool distilled = true; // Turbo (fixed shift, no guidance) or Raw
};
// text_encoder / vae empty: found beside the checkpoint (bf16 preferred);
// distilled -1: from the file name ("raw").
ComfyFiles resolve_comfy_files(const std::string &checkpoint,
                               const std::string &text_encoder,
                               const std::string &vae, int distilled);
struct Models {
  Ops ops;
  Weights text, transformer, vae;
  Tokenizer tokenizer;
  // ComfyUI's files as they are: the diffusion model checkpoint (its
  // non-block tensors and modulation tables), the text encoder (bf16 or
  // fp8_scaled) and the VAE; the tokenizer is embedded.
  Models(const std::string &checkpoint, const std::string &text_encoder,
         const std::string &vae_file);
  explicit Models(const struct ComfyFiles &files);
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
  void tables();
  Tensor fusion_block(const Tensor &x, const std::string &p, int batch,
                      int tokens);
  Tensor residual(const Tensor &x, const std::string &p, int h, int w);
  Tensor decode_tile(const Tensor &x, int h, int w);
};
} // namespace krea_native
