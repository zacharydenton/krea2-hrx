#include "native_models.h"
namespace krea_native {
static Tensor columns(const Tensor &x, int start, int n) {
  Tensor y(x.rows, n);
  gpu::Args a;
  a.i32(y.size()).ptr(x.ptr).ptr(y.ptr);
  native_launch("columns",
                {{"xsize", x.size()},
                 {"cols", size_t(n)},
                 {"width", size_t(x.cols)},
                 {"start1", size_t(start + 1)}},
                a, (y.size() + 255) / 256);
  return y;
}
Tensor Models::encode(const std::vector<int32_t> &ids) {
  if (ids.size() < 35 || ids.size() > 546)
    throw std::invalid_argument("text token count must be 35..546");
  if (text["embed_tokens.weight"].shape.size() != 2 ||
      text["embed_tokens.weight"].shape[1] != 2560)
    throw std::invalid_argument("embedding dimensions");
  int s = ids.size(), tokens = s - 34;
  Tensor x(s, 2560), taps(tokens * 12, 2560);
  auto idbuf = device_storage(ids.size() * 4);
  void *p = idbuf.get();
  gpu::copy(p, ids.data(), ids.size() * 4);
  for (int id : ids)
    if (id < 0 || id >= text["embed_tokens.weight"].shape[0])
      throw std::invalid_argument("token id out of range");
  gpu::Args embedding;
  embedding.i32(x.size())
      .ptr(text["embed_tokens.weight"].t.ptr)
      .ptr(p)
      .ptr(x.ptr);
  native_launch("embedding",
                {{"wsize", text["embed_tokens.weight"].t.size()},
                 {"rows", ids.size()},
                 {"cols", 2560}},
                embedding, (x.size() + 255) / 256);
  for (int i = 0; i < 35; ++i) {
    std::string layer = "layers." + std::to_string(i),
                att = layer + ".self_attn";
    auto norm = ops.norm(x, text[layer + ".input_layernorm.weight"].t, 1, 1e-6);
    auto q = lin(norm, text, att + ".q_proj"),
         k = lin(norm, text, att + ".k_proj"),
         v = lin(norm, text, att + ".v_proj");
    q = ops.norm(q.view(s * 32, 128), text[att + ".q_norm.weight"].t, 1, 1e-6)
            .view(s, 4096);
    k = ops.norm(k.view(s * 8, 128), text[att + ".k_norm.weight"].t, 1, 1e-6)
            .view(s, 1024);
    q = ops.rope(q, s, 32, 5000000);
    k = ops.rope(k, s, 8, 5000000);
    x = ops.binary(x,
                   lin(ops.attention(q, k, v, 1, s, 32, 8, 128, true), text,
                       att + ".o_proj"),
                   0);
    norm = ops.norm(x, text[layer + ".post_attention_layernorm.weight"].t, 1,
                    1e-6);
    auto g = ops.unary(lin(norm, text, layer + ".mlp.gate_proj"), 0),
         u = lin(norm, text, layer + ".mlp.up_proj");
    x = ops.binary(x, lin(ops.binary(g, u, 1), text, layer + ".mlp.down_proj"),
                   0);
    if (i % 3 == 1) {
      gpu::Args tap;
      tap.i32(tokens * 2560).ptr(x.ptr).ptr(taps.ptr);
      native_launch("tap",
                    {{"xsize", x.size()},
                     {"ysize", taps.size()},
                     {"tap1", size_t((i - 1) / 3 + 1)}},
                    tap, (size_t(tokens) * 2560 + 255) / 256);
    }
  };
  return taps;
}
Tensor Models::fusion_block(const Tensor &x, const std::string &p, int batch,
                            int tokens) {
  auto n = ops.norm(x, transformer[p + ".prenorm.scale"].t);
  auto q = lin(n, transformer, p + ".attn.wq"),
       k = lin(n, transformer, p + ".attn.wk"),
       v = lin(n, transformer, p + ".attn.wv");
  q = ops.norm(q.view(q.rows * 20, 128),
               transformer[p + ".attn.qknorm.qnorm.scale"].t)
          .view(x.rows, 2560);
  k = ops.norm(k.view(k.rows * 20, 128),
               transformer[p + ".attn.qknorm.knorm.scale"].t)
          .view(x.rows, 2560);
  auto attended = ops.attention(q, k, v, batch, tokens, 20, 20, 128);
  attended = ops.binary(attended,
                        ops.unary(lin(n, transformer, p + ".attn.gate"), 2), 1);
  auto y = ops.binary(x, lin(attended, transformer, p + ".attn.wo"), 0);
  n = ops.norm(y, transformer[p + ".postnorm.scale"].t);
  auto g = ops.unary(lin(n, transformer, p + ".mlp.gate"), 0),
       u = lin(n, transformer, p + ".mlp.up");
  return ops.binary(y, lin(ops.binary(g, u, 1), transformer, p + ".mlp.down"),
                    0);
}
Tensor Models::text_fusion(const Tensor &taps) {
  if (taps.rows % 12 || taps.cols != 2560 ||
      transformer["txtfusion.projector.weight"].t.size() != 12)
    throw std::invalid_argument("text fusion dimensions");
  int tokens = taps.rows / 12;
  auto x = taps;
  for (int i = 0; i < 2; ++i)
    x = fusion_block(x, "txtfusion.layerwise_blocks." + std::to_string(i),
                     tokens, 12);
  Tensor projected(tokens, 2560);
  gpu::Args fuse;
  fuse.i32(projected.size())
      .ptr(x.ptr)
      .ptr(transformer["txtfusion.projector.weight"].t.ptr)
      .ptr(projected.ptr);
  native_launch("fuse", {{"xsize", x.size()}, {"wsize", 12}}, fuse,
                (projected.size() + 255) / 256);
  x = projected;
  for (int i = 0; i < 2; ++i)
    x = fusion_block(x, "txtfusion.refiner_blocks." + std::to_string(i), 1,
                     tokens);
  x = ops.norm(x, transformer["txtmlp.0.scale"].t);
  return lin(ops.unary(lin(x, transformer, "txtmlp.1"), 1), transformer,
             "txtmlp.3");
}
std::pair<Tensor, Tensor> Models::time(float t) {
  t = float(B(t));
  std::vector<B> values(256);
  for (int i = 0; i < 128; ++i) {
    float a = t * 1000 * expf(-logf(10000.f) * i / 128);
    values[i] = B(cosf(a));
    values[i + 128] = B(sinf(a));
  }
  auto e = lin(
      ops.unary(lin(Tensor::upload(values, 1, 256), transformer, "tmlp.0"), 1),
      transformer, "tmlp.2");
  return {e, lin(ops.unary(e, 1), transformer, "tproj.1")};
}
Models::Models(const std::string &root)
    : text(root + "/text"), transformer(root + "/transformer"),
      vae(root + "/vae"), tokenizer(root + "/tokenizer.json"),
      block_tables(28 * 6, 6144) {
  for (int i = 0; i < 28; ++i) {
    const auto &table =
        transformer["blocks." + std::to_string(i) + ".mod.lin"].t;
    if (table.size() != 6 * 6144)
      throw std::invalid_argument("block modulation dimensions");
    gpu::copy(block_tables.ptr + size_t(i) * 6 * 6144, table.ptr,
              table.size() * sizeof(B));
  }
}
std::shared_ptr<float> Models::modulation(const Tensor &mod) {
  if (mod.size() != 6 * 6144)
    throw std::invalid_argument("modulation dimensions");
  auto storage = device_storage(modulation_elements * sizeof(float));
  auto result = std::shared_ptr<float>(storage, (float *)storage.get());
  gpu::Args args;
  args.i32(modulation_elements)
      .ptr(mod.ptr)
      .ptr(block_tables.ptr)
      .ptr(result.get());
  native_launch("modulation", {{"xsize", mod.size()}}, args,
                (modulation_elements + 255) / 256);
  return result;
}
Tensor Models::final(const Tensor &x, const Tensor &e) {
  if (e.size() != 6144 || x.cols != 6144)
    throw std::invalid_argument("final layer dimensions");
  auto table = transformer["last.modulation.lin"].t.view(2, 6144);
  Tensor expanded(2, 6144);
  gpu::copy(expanded.ptr, e.ptr, 6144 * 2);

  gpu::copy(expanded.ptr + 6144, e.ptr, 6144 * 2);
  auto m = ops.binary(expanded, table, 0);
  auto scale = m.view(1, 6144), shift = m.view(1, 6144, 6144);
  Tensor factor(1, 6144);
  gpu::Args add;
  add.i32(6144).ptr(scale.ptr).ptr(factor.ptr);
  native_launch("unary_one", {}, add, 24);
  auto norm = ops.norm(x, transformer["last.norm.scale"].t);
  return lin(ops.binary(ops.binary(norm, factor, 1), shift, 0), transformer,
             "last.linear");
}
Tensor Models::residual(const Tensor &x, const std::string &p, int h, int w) {
  Tensor skip = x;
  if (vae.has(p + ".conv_shortcut.weight"))
    skip = ops.conv(x, h, w, vae[p + ".conv_shortcut.weight"],
                    vae[p + ".conv_shortcut.bias"].t.ptr);
  auto y = ops.unary(ops.norm(x, vae[p + ".norm1.gamma"].t, 2), 0);
  y = ops.conv(y, h, w, vae[p + ".conv1.weight"], vae[p + ".conv1.bias"].t.ptr);
  y = ops.unary(ops.norm(y, vae[p + ".norm2.gamma"].t, 2), 0);
  return ops.binary(
      ops.conv(y, h, w, vae[p + ".conv2.weight"], vae[p + ".conv2.bias"].t.ptr),
      skip, 0);
}
Tensor Models::decode_tile(const Tensor &input, int h, int w) {
  auto conv = [&](const Tensor &x, const std::string &p) {
    return ops.conv(x, h, w, vae[p + ".weight"], vae[p + ".bias"].t.ptr);
  };
  auto x = conv(input, "post_quant_conv");
  x = conv(x, "decoder.conv_in");
  x = residual(x, "decoder.mid_block.resnets.0", h, w);
  auto norm =
      ops.norm(x, vae["decoder.mid_block.attentions.0.norm.gamma"].t, 2);
  auto qkv = conv(norm, "decoder.mid_block.attentions.0.to_qkv");
  int d = x.cols;
  auto att = ops.attention(columns(qkv, 0, d), columns(qkv, d, d),
                           columns(qkv, 2 * d, d), 1, h * w, 1, 1, d);
  x = ops.binary(x, conv(att, "decoder.mid_block.attentions.0.proj"), 0);
  x = residual(x, "decoder.mid_block.resnets.1", h, w);
  for (int i = 0; i < 4; ++i) {
    auto p = "decoder.up_blocks." + std::to_string(i);
    for (int j = 0; j < 3; ++j)
      x = residual(x, p + ".resnets." + std::to_string(j), h, w);
    if (i < 3) {
      x = ops.upsample(x, h, w);
      h *= 2;
      w *= 2;
      x = conv(x, p + ".upsamplers.0.resample.1");
    }
  }
  return conv(ops.unary(ops.norm(x, vae["decoder.norm_out.gamma"].t, 2), 0),
              "decoder.conv_out");
}
std::vector<uint8_t> Models::decode(const Tensor &packed, int height,
                                    int width) {
  int h = height / 8, w = width / 8;
  auto z = packed.download();
  const float mean[16] = {-.7571, -.7089, -.9113, .1075,  -.1745, .9653,
                          -.1517, 1.5508, .4134,  -.0715, .5517,  -.3632,
                          -.1922, -.9497, .2503,  -.2921};
  const float stddev[16] = {2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
                            2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
                            1.6382, 1.1253, 2.8251, 1.916};
  std::vector<B> latent(size_t(h) * w * 16);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 16; ++c) {
        size_t p = (size_t(y / 2) * (w / 2) + x / 2) * 64 + c * 4 +
                   (y % 2) * 2 + x % 2;
        latent[(size_t(y) * w + x) * 16 + c] =
            B(float(B(float(z[p]) / float(B(1 / float(B(stddev[c])))))) +
              float(B(mean[c])));
      }
  struct Tile {
    int h, w;
    std::vector<B> data;
  };
  const int stride = (h > 32 || w > 32) ? 24 : 32;
  const int output_stride = stride * 8;
  std::vector<std::vector<Tile>> tiles;
  for (int y = 0; y < h; y += stride) {
    std::vector<Tile> row;
    for (int x = 0; x < w; x += stride) {
      int th = std::min(32, h - y), tw = std::min(32, w - x);
      std::vector<B> input(size_t(th) * tw * 16);
      for (int j = 0; j < th; ++j)
        std::copy_n(latent.data() + ((size_t(y + j) * w + x) * 16), tw * 16,
                    input.data() + size_t(j) * tw * 16);
      auto output = decode_tile(Tensor::upload(input, th * tw, 16), th, tw);
      row.push_back({th * 8, tw * 8, output.download()});
    }
    tiles.push_back(std::move(row));
  }
  std::vector<uint8_t> rgb(size_t(height) * width * 3);
  for (size_t r = 0; r < tiles.size(); ++r)
    for (size_t c = 0; c < tiles[r].size(); ++c) {
      auto &t = tiles[r][c];
      if (r) {
        auto &a = tiles[r - 1][c];
        int blend = std::min({64, a.h, t.h});
        for (int y = 0; y < blend; ++y)
          for (int x = 0; x < t.w; ++x)
            for (int k = 0; k < 3; ++k) {
              float f = float(y) / blend;
              auto i = (size_t(y) * t.w + x) * 3 + k;
              t.data[i] = B(
                  float(B(
                      float(
                          a.data[(size_t(a.h - blend + y) * a.w + x) * 3 + k]) *
                      (1 - f))) +
                  float(B(float(t.data[i]) * f)));
            }
      }
      if (c) {
        auto &a = tiles[r][c - 1];
        int blend = std::min({64, a.w, t.w});
        for (int y = 0; y < t.h; ++y)
          for (int x = 0; x < blend; ++x)
            for (int k = 0; k < 3; ++k) {
              float f = float(x) / blend;
              auto i = (size_t(y) * t.w + x) * 3 + k;
              t.data[i] = B(
                  float(B(
                      float(
                          a.data[(size_t(y) * a.w + a.w - blend + x) * 3 + k]) *
                      (1 - f))) +
                  float(B(float(t.data[i]) * f)));
            }
      }
      for (int y = 0; y < std::min(output_stride, t.h) &&
                      int(r) * output_stride + y < height;
           ++y)
        for (int x = 0; x < std::min(output_stride, t.w) &&
                        int(c) * output_stride + x < width;
             ++x)
          for (int k = 0; k < 3; ++k) {
            float sample = float(t.data[(size_t(y) * t.w + x) * 3 + k]);
            if (!std::isfinite(sample))
              throw std::runtime_error("VAE produced a nonfinite sample");
            float v = std::clamp(sample * .5f + .5f, 0.f, 1.f);
            rgb[((r * output_stride + y) * width + c * output_stride + x) * 3 +
                k] = uint8_t(std::nearbyint(v * 255));
          }
    }
  return rgb;
}
} // namespace krea_native
