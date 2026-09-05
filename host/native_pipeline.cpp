#include "krea2.h"
#include "krea2_device.h"
#include "krea2_pipeline.h"
#include "native_compile.h"
#include "native_models.h"
#include "native_profile.h"
#include "native_schedule.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>

using namespace krea_native;
struct krea2_pipeline {
  std::string bundle, compiler;
  Models models;
  std::mutex mutex;
  std::shared_ptr<BufferPool> pool = std::make_shared<BufferPool>();
  std::vector<float> rope_cos, rope_sin;
  int rope_width = 0, rope_height = 0, rope_text_tokens = 0;
  std::shared_ptr<krea2_weights> block_weights;
  krea2_session *blocks = nullptr;
  int block_tokens = 0;
  krea2_pipeline(std::string b, std::string c)
      : bundle(std::move(b)), compiler(std::move(c)), models(bundle) {}
  ~krea2_pipeline() { krea2_destroy(blocks); }
  Tensor forward(const Tensor &latents, const Tensor &text, float t, int width,
                 int height) {
    Profile timing("forward");
    int image_tokens = width / 16 * (height / 16),
        tokens = text.rows + image_tokens;
    auto [temb, mod] = models.time(t);
    auto image = models.image_in(latents);
    Tensor x(tokens, 6144);

    gpu::copy(x.ptr, text.ptr, text.size() * 2);
    gpu::copy(x.ptr + text.size(), image.ptr, image.size() * 2);
    timing.mark("embeddings");
    if (tokens != block_tokens) {
      krea2_destroy(blocks);
      blocks = nullptr;
      block_tokens = 0;
      auto kernels = prepare_kernels(bundle, compiler, tokens);
      if (!block_weights)
        block_weights = krea2_load_weights(bundle + "/blocks");
      blocks = krea2_create_shared(block_weights, kernels, tokens, 28);
      block_tokens = tokens;
    }
    timing.mark("prepare session");
    auto mods = models.modulation(mod);
    if (width != rope_width || height != rope_height ||
        text.rows != rope_text_tokens) {
      // Include geometry, not just token count: rectangular grids have
      // different phases.
      rope_width = rope_height = rope_text_tokens = 0;
      rope_cos.resize(size_t(tokens) * 128);
      rope_sin.resize(rope_cos.size());
      for (int s = 0; s < tokens; ++s) {
        int offset = 0;
        for (int axis = 0; axis < 3; ++axis) {
          int d = axis ? 48 : 32,
              pos = s < text.rows ? 0
                    : axis == 1   ? (s - text.rows) / (width / 16)
                    : axis == 2   ? (s - text.rows) % (width / 16)
                                  : 0;
          for (int k = 0; k < d; ++k) {
            double phase = pos * std::pow(1000., -2. * (k / 2) / d);
            rope_cos[size_t(s) * 128 + offset + k] = std::cos(phase);
            rope_sin[size_t(s) * 128 + offset + k] = std::sin(phase);
          }
          offset += d;
        }
      }
      rope_width = width;
      rope_height = height;
      rope_text_tokens = text.rows;
    }
    timing.mark("modulation and rope");
    krea2_run_device_bf16(blocks, (uint16_t *)x.ptr, x.size(), mods.get(),
                          Models::modulation_elements, rope_cos.data(),
                          rope_sin.data(), rope_cos.size());
    timing.mark("blocks");
    auto output = x.view(image_tokens, 6144, size_t(text.rows) * 6144);
    output = models.final(output, temb);
    timing.mark("final layer");
    return output;
  }
};
namespace {
void dimensions(int w, int h) {
  if (w < 64 || h < 64 || w > 2048 || h > 2048 || w % 16 || h % 16)
    throw std::invalid_argument(
        "dimensions must be multiples of 16 in 64..2048");
}
Tensor upload_float(const float *p, size_t n, int rows, int cols) {
  if (!p || n != size_t(rows) * cols)
    throw std::invalid_argument("wrong input buffer size");
  std::vector<B> b(n);
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i]))
      throw std::invalid_argument("nonfinite input");
    b[i] = B(p[i]);
    if (!std::isfinite(float(b[i])))
      throw std::invalid_argument("input exceeds bf16 range");
  }
  return Tensor::upload(b, rows, cols);
}
void output_float(const Tensor &t, float *p, size_t n) {
  if (!p || n < t.size())
    throw std::invalid_argument("output buffer too small");
  auto b = t.download();
  for (size_t i = 0; i < b.size(); ++i)
    p[i] = float(b[i]);
}
template <class F> int guard(krea2_pipeline *p, char *e, size_t n, F f) {
  if (e && n)
    e[0] = 0;
  try {
    if (!p)
      throw std::invalid_argument("null pipeline");
    std::lock_guard<std::mutex> lock(p->mutex);
    PoolScope buffers(p->pool);
    f();
    gpu::synchronize();
    return 0;
  } catch (const std::exception &ex) {
    if (e && n)
      snprintf(e, n, "%s", ex.what());
    return 1;
  } catch (...) {
    if (e && n)
      snprintf(e, n, "native inference failed");
    return 1;
  }
}
} // namespace
extern "C" uint32_t krea2_pipeline_abi_version() {
  return KREA2_PIPELINE_ABI_VERSION;
}
extern "C" int krea2_pipeline_create(const char *b, const char *c,
                                     krea2_pipeline **out, char *e, size_t n) {
  if (e && n)
    e[0] = 0;
  if (out)
    *out = nullptr;
  try {
    if (!out || !b)
      throw std::invalid_argument("bundle and output pointer are required");
    std::ifstream f(std::string(b) + "/native.json");
    if (!f)
      throw std::runtime_error("cannot read " + std::string(b) +
                               "/native.json");
    json j;
    f >> j;
    if (j.at("version") != 1 || j.at("model") != "krea2-turbo")
      throw std::invalid_argument("unsupported native bundle");
    const char *env = getenv("LOOM_COMPILE");
    native_compiler(c ? c : env ? env : "loom-compile");
    *out = new krea2_pipeline(b, c ? c : env ? env : "loom-compile");
    return 0;
  } catch (const std::exception &ex) {
    if (e && n)
      snprintf(e, n, "%s", ex.what());
    return 1;
  } catch (...) {
    if (e && n)
      snprintf(e, n, "native initialization failed");
    return 1;
  }
}
extern "C" void krea2_pipeline_destroy(krea2_pipeline *p) { delete p; }
extern "C" int krea2_tokenize(krea2_pipeline *p, const char *text, int32_t *out,
                              size_t cap, size_t *written, char *e, size_t n) {
  return guard(p, e, n, [&] {
    if (!text || !written)
      throw std::invalid_argument("text and count are required");
    auto ids = p->models.tokenizer.encode(text);
    *written = ids.size();
    if (!out)
      return;
    if (cap < ids.size())
      throw std::invalid_argument("token buffer too small");
    std::copy(ids.begin(), ids.end(), out);
  });
}
extern "C" int krea2_encode(krea2_pipeline *p, const char *prompt, float *out,
                            size_t cap, size_t *tokens, char *e, size_t n) {
  return guard(p, e, n, [&] {
    if (!prompt || !tokens)
      throw std::invalid_argument("prompt and count are required");
    auto ids = p->models.tokenizer.prompt(prompt);
    *tokens = ids.size() - 34;
    if (!out)
      return;
    if (cap < *tokens * 12 * 2560)
      throw std::invalid_argument("text output buffer too small");
    output_float(p->models.encode(ids), out, cap);
  });
}
extern "C" int krea2_transformer(krea2_pipeline *p, const float *text,
                                 size_t tn, int tokens, const float *latents,
                                 size_t ln, int w, int h, float t, float *out,
                                 size_t on, char *e, size_t n) {
  return guard(p, e, n, [&] {
    dimensions(w, h);
    if (tokens < 1 || tokens > 512 || !std::isfinite(t) || t < 0 || t > 1 ||
        !out || on != size_t(w / 16) * (h / 16) * 64)
      throw std::invalid_argument("invalid transformer arguments");
    auto taps = upload_float(text, tn, tokens * 12, 2560);
    auto l = upload_float(latents, ln, w / 16 * (h / 16), 64);
    output_float(p->forward(l, p->models.text_fusion(taps), t, w, h), out, on);
  });
}
extern "C" int krea2_decode(krea2_pipeline *p, const float *latents,
                            size_t count, int w, int h, uint8_t *rgb,
                            size_t cap, char *e, size_t n) {
  return guard(p, e, n, [&] {
    dimensions(w, h);
    if (!rgb || cap < size_t(w) * h * 3)
      throw std::invalid_argument("RGB output buffer too small");
    auto x = upload_float(latents, count, w / 16 * (h / 16), 64);
    auto result = p->models.decode(x, h, w);
    std::copy(result.begin(), result.end(), rgb);
  });
}
extern "C" int krea2_generate(krea2_pipeline *p, const char *prompt, int w,
                              int h, int steps, uint64_t seed,
                              const float *initial, size_t count, uint8_t *rgb,
                              size_t cap, char *e, size_t n) {
  return guard(p, e, n, [&] {
    dimensions(w, h);
    if (!prompt || steps < 1 || steps > 100 || !rgb || cap < size_t(w) * h * 3)
      throw std::invalid_argument("invalid generation arguments");
    size_t elements = size_t(w / 16) * (h / 16) * 64;
    std::vector<float> random;
    if (!initial) {
      if (count)
        throw std::invalid_argument("latent count without input buffer");
      std::mt19937_64 gen(seed);
      random.resize(elements);
      for (size_t i = 0; i < elements; i += 2) {
        double u = ((gen() >> 11) + 1.) / 9007199254740993.,
               v = (gen() >> 11) / 9007199254740992.;
        double radius = sqrt(-2 * log(u));
        random[i] = radius * cos(6.283185307179586 * v);
        if (i + 1 < elements)
          random[i + 1] = radius * sin(6.283185307179586 * v);
      }
      initial = random.data();
      count = elements;
    }
    Profile timing("generate");
    auto latents = upload_float(initial, count, w / 16 * (h / 16), 64);
    auto text = p->models.text_fusion(
        p->models.encode(p->models.tokenizer.prompt(prompt)));
    timing.mark("encode and text fusion");
    for (int step = 0; step < steps; ++step) {
      float sigma = scheduler_sigma(step, steps),
            next = scheduler_sigma(step + 1, steps);
      auto velocity = p->forward(latents, text, sigma, w, h);
      p->models.ops.euler_step(latents, velocity, next - sigma);
    }
    timing.mark("denoise");
    auto result = p->models.decode(latents, h, w);
    timing.mark("VAE decode");
    std::copy(result.begin(), result.end(), rgb);
  });
}
