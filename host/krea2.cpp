// The 28 Krea 2 transformer blocks as a resident Loom session: per block ten
// launches (prepare -> fused qkv|gate GEMM -> QK-norm+RoPE (+ contiguous q/k/v)
// -> attention -> gated prepare -> wo GEMM with gated residual -> prepare ->
// fused gate|up GEMM -> SwiGLU prepare -> down GEMM with gated residual), W4A4
// ConvRot throughout.
//
// Build: ./scripts/build_host.sh

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "gemm_shape.h"
#include "krea2.h"
#include "krea2_device.h"
#include "safetensors.h"
#include "native_kernels.h"
#include "sage.h"
#include <memory>

namespace {

constexpr int HIDDEN = 6144, KV_HEADS = 12, HEAD_DIM = 128, INTER = 16384;
constexpr int QKVG = HIDDEN + 2 * KV_HEADS * HEAD_DIM + HIDDEN; // 15360
constexpr int GATE_OFFSET = HIDDEN + 2 * KV_HEADS * HEAD_DIM;
constexpr int THREADS = 256;

// A tensor's home on the device, and where its bytes come from: a run of row
// segments of ComfyUI's checkpoint mapping (the fused qkv|gate and interleaved
// gate|up operands are assembled here) or a small host-built array (their
// per-row scales). GEMM weights ("<name>.q", int8 rows [N][K]) are re-pitched
// on upload to their padded row pitch; everything else keeps its layout.
struct Segment {
  const char *source;
  size_t rows;
};
struct Span {
  size_t bytes = 0; // as the session counts the tensor (its file bytes)
  size_t device_offset = 0, device_bytes = 0;
  size_t rows = 0, row_bytes = 0, device_row_bytes = 0; // .q tensors only
  int bits = 0;                                          // .q tensors only
  std::vector<Segment> segments; // checkpoint rows, in order
  std::vector<char> host;        // host-built bytes (else segments)
};

// ComfyUI's checkpoint as it is: per block the int8 ConvRot rows of wq, wk,
// wv and the attention gate become one qkv|gate operand, the MLP gate and up
// rows are interleaved in 16-row groups for the SwiGLU epilogue, wo and down
// stay as they are, the f32 per-row scales follow the same arrangement, and
// the RMSNorm scales are the checkpoint's f32 tensors.
std::map<std::string, Span>
checkpoint_spans(const krea_native::SafeTensors &file, int &bits) {
  using krea_native::SafeTensors;
  std::map<std::string, Span> spans;
  auto rows_of = [&](const SafeTensors::Entry &e) { return size_t(e.shape.at(0)); };
  auto row_bytes_of = [&](const SafeTensors::Entry &e) {
    return e.bytes / size_t(e.shape.at(0));
  };
  auto codes = [&](const std::string &name) -> const SafeTensors::Entry & {
    const auto &e = file.at(name + ".weight");
    if (e.dtype != "I8" || e.shape.size() != 2)
      throw std::runtime_error(name + ".weight is " + e.dtype +
                               ", not int8 ConvRot rows (" + file.path + ")");
    return e;
  };
  auto scales = [&](const std::string &name) -> const SafeTensors::Entry & {
    const auto &e = file.at(name + ".weight_scale");
    if (e.dtype != "F32")
      throw std::runtime_error(name + ".weight_scale is not float32");
    return e;
  };
  auto operand = [&](const std::string &out, const std::vector<std::string> &parts,
                     size_t group) {
    // group 0: concatenate the parts' rows; group g: interleave them g rows at a time
    Span q, s;
    q.bits = 8;
    std::vector<const SafeTensors::Entry *> entries;
    for (const auto &part : parts) {
      entries.push_back(&codes(part));
      if (entries.back()->shape.at(1) != entries.front()->shape.at(1))
        throw std::runtime_error("mismatched K in " + out);
      q.rows += rows_of(*entries.back());
    }
    q.row_bytes = row_bytes_of(*entries.front());
    q.bytes = q.rows * q.row_bytes;
    q.device_row_bytes =
        size_t(krea2_shape::gemm_pitch(int(q.row_bytes), 8)); // int8: K bytes
    q.device_bytes = q.rows * q.device_row_bytes;
    s.host.resize(q.rows * 4);
    s.bytes = s.device_bytes = s.host.size();
    if (group == 0) {
      size_t row = 0;
      for (size_t i = 0; i < entries.size(); ++i) {
        q.segments.push_back({file.data(*entries[i]), rows_of(*entries[i])});
        const auto &sc = scales(parts[i]);
        if (sc.bytes != rows_of(*entries[i]) * 4)
          throw std::runtime_error("scale count in " + parts[i]);
        std::memcpy(s.host.data() + row * 4, file.data(sc), sc.bytes);
        row += rows_of(*entries[i]);
      }
    } else {
      if (entries.size() != 2 || rows_of(*entries[0]) != rows_of(*entries[1]) ||
          rows_of(*entries[0]) % group)
        throw std::runtime_error("interleave shape in " + out);
      const auto &s0 = scales(parts[0]), &s1 = scales(parts[1]);
      size_t row = 0;
      for (size_t r = 0; r < rows_of(*entries[0]); r += group)
        for (int side = 0; side < 2; ++side) {
          const auto &e = *entries[side];
          q.segments.push_back({file.data(e) + r * q.row_bytes, group});
          std::memcpy(s.host.data() + row * 4,
                      file.data(side ? s1 : s0) + r * 4, group * 4);
          row += group;
        }
    }
    spans[out + ".q"] = std::move(q);
    spans[out + ".s"] = std::move(s);
  };
  auto vector = [&](const std::string &out, const std::string &name) {
    const auto &e = file.at(name);
    if (e.dtype != "F32")
      throw std::runtime_error(name + " is " + e.dtype + ", not float32");
    Span v;
    v.bytes = v.device_bytes = e.bytes;
    v.segments.push_back({file.data(e), 1});
    v.row_bytes = e.bytes; // one row of the whole tensor
    spans[out] = std::move(v);
  };
  int layers = 0;
  while (file.has("blocks." + std::to_string(layers) + ".attn.wq.weight"))
    ++layers;
  if (!layers)
    throw std::runtime_error("no transformer blocks in " + file.path);
  for (int i = 0; i < layers; ++i) {
    std::string p = "blocks." + std::to_string(i);
    operand(p + ".qkvg", {p + ".attn.wq", p + ".attn.wk", p + ".attn.wv", p + ".attn.gate"}, 0);
    operand(p + ".wo", {p + ".attn.wo"}, 0);
    operand(p + ".gu", {p + ".mlp.gate", p + ".mlp.up"}, 16);
    operand(p + ".down", {p + ".mlp.down"}, 0);
    vector(p + ".prenorm", p + ".prenorm.scale");
    vector(p + ".postnorm", p + ".postnorm.scale");
    vector(p + ".qnorm", p + ".attn.qknorm.qnorm.scale");
    vector(p + ".knorm", p + ".attn.qknorm.knorm.scale");
  }
  bits = 8;
  return spans;
}

bool is_safetensors(const std::string &path) {
  return path.size() > 12 &&
         path.compare(path.size() - 12, 12, ".safetensors") == 0;
}

} // namespace

struct krea2_weights {
  std::map<std::string, Span> spans;
  std::shared_ptr<void> storage;
  int bits = 4;
  // checkpoint: ComfyUI's int8 ConvRot diffusion model (.safetensors), read
  // as it is.
  explicit krea2_weights(const std::string &checkpoint) {
    if (!is_safetensors(checkpoint))
      throw std::runtime_error(checkpoint + " is not a ComfyUI checkpoint (.safetensors)");
    krea_native::SafeTensors file(checkpoint);
    spans = checkpoint_spans(file, bits);
    // Device layout: each tensor at a 256-byte boundary, GEMM weights with
    // their padded row pitch.
    size_t total = 0;
    for (auto &[name, span] : spans) {
      span.device_offset = total;
      total += (span.device_bytes + 255) / 256 * 256;
    }
    void *device = gpu::allocate(total);
    storage =
        std::shared_ptr<void>(device, [](void *p) { (void)gpu::release(p); });
    const size_t chunk_bytes = size_t(16) << 20;
    std::vector<char> staging;
    for (const auto &[name, span] : spans) {
      char *dst = (char *)device + span.device_offset;
      if (!span.host.empty()) {
        gpu::copy(dst, span.host.data(), span.host.size());
        continue;
      }
      // Checkpoint rows: assemble the operand's rows in order, at the device
      // pitch (the pad bytes stay zero: the kernels never read them,
      // tests/test_gemm_i4.py GEMM_KPAD), a staging chunk at a time.
      const size_t pitch = span.rows ? span.device_row_bytes : span.row_bytes;
      const size_t width = span.row_bytes;
      size_t rows_per_chunk = std::max<size_t>(1, chunk_bytes / pitch);
      staging.assign(rows_per_chunk * pitch, 0);
      size_t staged = 0, written = 0;
      auto flush = [&] {
        gpu::copy(dst + written * pitch, staging.data(), staged * pitch);
        written += staged;
        staged = 0;
      };
      for (const auto &segment : span.segments)
        for (size_t r = 0; r < segment.rows; ++r) {
          std::memcpy(staging.data() + staged * pitch,
                      segment.source + r * width, width);
          if (++staged == rows_per_chunk)
            flush();
        }
      if (staged)
        flush();
    }
  }
};

std::shared_ptr<krea2_weights>
krea2_load_weights(const std::string &directory) {
  return std::make_shared<krea2_weights>(directory);
}

int krea2_weights_bits(const krea2_weights &weights) { return weights.bits; }

namespace {

struct Kernel {
  gpu::Kernel kernel;
  void load(const std::string &path, const char *symbol) {
    kernel = gpu::Kernel(path, symbol);
  }
};

struct KernArgs {
  alignas(16) unsigned char bytes[160];
  size_t size = 0;
  void scalar_i32(int v) {
    size = (size + 3) & ~size_t(3);
    memcpy(bytes + size, &v, 4);
    size += 4;
  }
  void pointer(const void *p) {
    size = (size + 7) & ~size_t(7);
    memcpy(bytes + size, &p, 8);
    size += 8;
  }
};

class Session {
public:
  Session(const std::string &weights_dir, const std::string &kernels_dir,
          int tokens, int layers, std::shared_ptr<krea2_weights> weights = {})
      : tokens_(tokens), layers_(layers) {
    try {
      if (tokens < 16 || tokens > 16896)
        throw std::invalid_argument("tokens must be 16..16896");
      if (layers < 1 || layers > 28)
        throw std::invalid_argument("layers must be 1..28");
      capacity_ = std::max<size_t>(
          (tokens + 16 + 31) / 32 * 32,
          (tokens + 63) / 64 * 64); // tokens+16 headroom, whole 64-key blocks
      // "3 tokens gemm_rows m_group capacity attention_waves pitch(6144)
      // pitch(16384) attention_bits gemm_bits": every shape field must match
      // what this host derives for gemm_bits (the weights' width, 4 or 8);
      // attention_bits (4 or 8) is the builder's choice.
      std::ifstream metadata(kernels_dir + "/launch.txt");
      unsigned version = 0, compiled_tokens = 0, pitch_hidden = 0,
               pitch_inter = 0;
      size_t compiled_capacity = 0;
      if (!(metadata >> version >> compiled_tokens >> gemm_rows_ >> m_group_ >>
            compiled_capacity >> attention_waves_ >> pitch_hidden >>
            pitch_inter >> attention_bits_ >> gemm_bits_) ||
          version != 3 || compiled_tokens != unsigned(tokens) ||
          (attention_bits_ != 4 && attention_bits_ != 8) ||
          (gemm_bits_ != 4 && gemm_bits_ != 8) ||
          compiled_capacity != capacity_ ||
          gemm_rows_ !=
              unsigned(krea2_shape::gemm_rows(tokens, int(gemm_bits_))) ||
          m_group_ !=
              unsigned(krea2_shape::gemm_m_group(tokens, int(gemm_rows_))) ||
          attention_waves_ != (tokens < 8192 ? 8u : 4u) ||
          pitch_hidden !=
              unsigned(krea2_shape::gemm_pitch(HIDDEN, int(gemm_bits_))) ||
          pitch_inter !=
              unsigned(krea2_shape::gemm_pitch(INTER, int(gemm_bits_))))
        throw std::invalid_argument("invalid kernel launch metadata; rebuild "
                                    "with scripts/build_kernels.py");
      weights_ = weights ? std::move(weights) : krea2_load_weights(weights_dir);
      if (unsigned(weights_->bits) != gemm_bits_)
        throw std::invalid_argument(
            "kernel bundle built for int" + std::to_string(gemm_bits_) +
            " GEMM operands but the weights are int" +
            std::to_string(weights_->bits));
      const size_t bits = gemm_bits_;
      const auto &spans = weights_->spans;
      auto need = [&](const std::string &name, size_t bytes) {
        auto it = spans.find(name);
        if (it == spans.end())
          throw std::runtime_error("missing tensor " + name);
        if (it->second.bytes != bytes)
          throw std::runtime_error("tensor " + name + " has " +
                                   std::to_string(it->second.bytes) +
                                   " bytes, expected " + std::to_string(bytes));
        return (char *)weights_->storage.get() + it->second.device_offset;
      };
      for (int i = 0; i < layers; ++i) {
        std::string p = "blocks." + std::to_string(i);
        Block b;
        b.qkvg_q = need(p + ".qkvg.q", size_t(QKVG) * HIDDEN * bits / 8);
        b.qkvg_s = need(p + ".qkvg.s", size_t(QKVG) * 4);
        b.wo_q = need(p + ".wo.q", size_t(HIDDEN) * HIDDEN * bits / 8);
        b.wo_s = need(p + ".wo.s", size_t(HIDDEN) * 4);
        b.gu_q = need(p + ".gu.q", size_t(2 * INTER) * HIDDEN * bits / 8);
        b.gu_s = need(p + ".gu.s", size_t(2 * INTER) * 4);
        b.down_q = need(p + ".down.q", size_t(HIDDEN) * INTER * bits / 8);
        b.down_s = need(p + ".down.s", size_t(HIDDEN) * 4);
        b.prenorm = need(p + ".prenorm", HIDDEN * 4);
        b.postnorm = need(p + ".postnorm", HIDDEN * 4);
        b.qnorm = need(p + ".qnorm", HEAD_DIM * 4);
        b.knorm = need(p + ".knorm", HEAD_DIM * 4);
        blocks_.push_back(b);
      }
      auto load = [&](Kernel &k, const char *stem, const char *symbol) {
        k.load(kernels_dir + "/" + stem + ".hsaco", symbol);
      };
      const std::string ib = "i" + std::to_string(bits);
      load(k_prep_norm_, ("prepare_norm_" + ib).c_str(),
           ("krea2_prepare_norm_" + ib).c_str());
      load(k_prep_gated_, ("prepare_gated_" + ib).c_str(),
           ("krea2_prepare_gated_" + ib).c_str());
      load(k_prep_swiglu_, ("prepare_plain_" + ib).c_str(),
           ("krea2_prepare_plain_" + ib).c_str());
      const std::string tile = gemm_rows_ == 256 ? "_256" : "";
      load(k_gemm_qkvg_, "gemm_qkvg", ("krea2_gemm_" + ib + tile).c_str());
      load(k_gemm_gu_, "gemm_gu",
           ("krea2_gemm_" + ib + "_swiglu" + tile).c_str());
      load(k_gemm_wo_, "gemm_wo",
           ("krea2_gemm_" + ib + "_resid" + tile).c_str());
      load(k_gemm_down_, "gemm_down",
           ("krea2_gemm_" + ib + "_resid" + tile).c_str());
      load(k_rope_, "rope_qknorm", "krea2_rope_qknorm_f16");
      std::string attention = attention_bits_ == 4
                                  ? "krea2_attention_sage_i4_fast"
                                  : "krea2_attention_sage_i8_fast";
      if (attention_waves_ != 8)
        attention += "_prefetch";
      load(k_attention_, "attention", attention.c_str());
      sage_ = std::make_unique<SagePreparation>(tokens, int(capacity_), 48, 12,
                                                int(attention_bits_));
      const size_t T = capacity_;
      x_ = gpu::allocate(T * HIDDEN * 2);
      // the widest prepared operand: down's K = 16384 at its padded pitch
      a_q_ = gpu::allocate(
          T * size_t(krea2_shape::gemm_pitch(INTER, int(bits))) * bits / 8);
      a_s_ = gpu::allocate(T * 4);
      fused_ = gpu::allocate(T * QKVG * 2);
      q_ = gpu::allocate(T * HIDDEN * 2);
      k_ = gpu::allocate(size_t(KV_HEADS * HEAD_DIM) * T * 2);
      v_ = gpu::allocate(size_t(KV_HEADS * HEAD_DIM) * T * 2);
      gpu::zero(v_, size_t(KV_HEADS * HEAD_DIM) * T * 2);
      gpu::zero(q_, T * HIDDEN * 2);
      gpu::zero(k_, size_t(KV_HEADS * HEAD_DIM) * T * 2);
      attn_ = gpu::allocate(T * HIDDEN * 2);
      gu_ = gpu::allocate(T * INTER *
                          2); // silu(gate) * up, fused into the GEMM epilogue
      mods_ = gpu::allocate(size_t(layers) * 6 * HIDDEN * 4);
      cos_ = gpu::allocate(T * HEAD_DIM * 4);
      sin_ = gpu::allocate(T * HEAD_DIM * 4);
      gpu::zero(fused_, T * QKVG * 2); // headroom rows stay zero
      gpu::zero(x_, T * HIDDEN * 2);
    } catch (...) {
      release();
      throw;
    }
  }
  ~Session() { release(); }

private:
  void release() noexcept {
    for (void *p :
         {x_, a_q_, a_s_, fused_, q_, k_, v_, attn_, gu_, mods_, cos_, sin_})
      if (p)
        (void)gpu::release(p);
  }

public:
  void run(uint16_t *x, size_t x_elements, const float *mods,
           size_t mods_elements, const float *cos, const float *sin,
           size_t rope_elements, int first_block = 0, int block_count = -1,
           bool device_bf16 = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (first_block < 0 || first_block >= layers_)
      throw std::invalid_argument(
          "block range must be within the loaded layers");
    if (block_count == -1)
      block_count = layers_ - first_block;
    if (block_count < 1 || block_count > layers_ - first_block)
      throw std::invalid_argument(
          "block range must be within the loaded layers");
    const size_t T = tokens_;
    if (x_elements != T * HIDDEN)
      throw std::invalid_argument("x has " + std::to_string(x_elements) +
                                  " elements, expected " +
                                  std::to_string(T * HIDDEN));
    if (mods_elements != size_t(layers_) * 6 * HIDDEN)
      throw std::invalid_argument("mods has the wrong element count");
    if (rope_elements != T * HEAD_DIM)
      throw std::invalid_argument("cos/sin have the wrong element count");
    if (device_bf16) {
      gpu::Args args;
      args.i32(x_elements).ptr(x).ptr(x_);
      krea_native::native_launch("cast_bf16_f16", {}, args,
                                 (x_elements + 255) / 256);
    } else {
      gpu::copy(x_, x, T * HIDDEN * 2);
    }
    const float *device_mods = mods;
    if (!device_bf16) {
      gpu::copy(mods_, mods, mods_elements * 4);
      device_mods = (const float *)mods_;
    }
    gpu::copy(cos_, cos, rope_elements * 4);
    gpu::copy(sin_, sin, rope_elements * 4);
    for (int i = first_block; i < first_block + block_count; ++i)
      block(i, device_mods);
    if (device_bf16) {
      gpu::Args args;
      args.i32(x_elements).ptr(x_).ptr(x);
      krea_native::native_launch("cast_f16_bf16", {}, args,
                                 (x_elements + 255) / 256);
      gpu::synchronize();
    } else {
      gpu::synchronize();
      gpu::copy(x, x_, T * HIDDEN * 2);
    }
    if (profile) {
      double total = 0;
      for (auto &e : stage_us)
        total += e.second;
      std::vector<std::pair<double, std::string>> rows;
      for (auto &e : stage_us)
        rows.push_back({e.second, e.first});
      std::sort(rows.rbegin(), rows.rend());
      fprintf(stderr, "stage profile over %d block(s), %zu tokens:\n",
              block_count, T);
      for (auto &r : rows)
        fprintf(stderr, "  %-24s %9.3f ms  %5.1f%%\n", r.second.c_str(),
                r.first / 1000.0, 100.0 * r.first / total);
      fprintf(stderr, "  %-24s %9.3f ms\n", "total", total / 1000.0);
      stage_us.clear();
    }
  }

  bool profile = false;
  std::map<std::string, double> stage_us;

private:
  struct Block {
    char *qkvg_q, *qkvg_s, *wo_q, *wo_s, *gu_q, *gu_s, *down_q, *down_s,
        *prenorm, *postnorm, *qnorm, *knorm;
  };

  void launch(Kernel &k, const char *stage, unsigned gx, unsigned gy,
              unsigned bx, KernArgs &args) {
    std::chrono::steady_clock::time_point t0;
    if (profile) {
      gpu::synchronize();
      t0 = std::chrono::steady_clock::now();
    }
    k.kernel.launch(gx, gy, bx, args.bytes, args.size);
    if (profile) {
      gpu::synchronize();
      stage_us[stage] += std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    }
  }
  unsigned gemm_grid_y(size_t m) const {
    return unsigned(krea2_shape::gemm_grid_rows(int(m), int(gemm_rows_),
                                                int(m_group_)));
  }

  void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n,
            void *out, const float *gate) {
    KernArgs a;
    a.scalar_i32(int(tokens_));
    a.pointer(a_q_);
    a.pointer(w_q);
    a.pointer(w_s);
    a.pointer(a_s_);
    a.pointer(out);
    if (gate)
      a.pointer(gate);
    launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
  }

  void block(int i, const float *mods) {
    const Block &b = blocks_[i];
    const float *mod = mods + size_t(i) * 6 * HIDDEN;
    const float *prescale = mod, *preshift = mod + HIDDEN,
                *pregate = mod + 2 * HIDDEN;
    const float *postscale = mod + 3 * HIDDEN, *postshift = mod + 4 * HIDDEN,
                *postgate = mod + 5 * HIDDEN;
    const unsigned T = unsigned(tokens_);
    {
      KernArgs a;
      a.scalar_i32(T);
      a.pointer(x_);
      a.pointer(b.prenorm);
      a.pointer(prescale);
      a.pointer(preshift);
      a.pointer(a_q_);
      a.pointer(a_s_);
      launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a);
    }
    gemm(k_gemm_qkvg_, "gemm qkv|gate", b.qkvg_q, b.qkvg_s, QKVG, fused_,
         nullptr);
    {
      KernArgs a;
      a.scalar_i32(T);
      a.pointer(fused_);
      a.pointer(b.qnorm);
      a.pointer(b.knorm);
      a.pointer(cos_);
      a.pointer(sin_);
      a.pointer(q_);
      a.pointer(k_);
      a.pointer(v_);
      launch(k_rope_, "qk norm + rope", T, 1, THREADS, a);
    }
    {
      auto start = std::chrono::steady_clock::now();
      if (profile) {
        gpu::synchronize();
        start = std::chrono::steady_clock::now();
      }
      sage_->run(q_, k_, v_);
      if (profile) {
        gpu::synchronize();
        stage_us["SA2 preprocessing"] +=
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start)
                .count();
      }
      KernArgs a;
      a.scalar_i32(T);
      a.scalar_i32(KV_HEADS);
      a.pointer(sage_->q4);
      a.pointer(sage_->k4);
      a.pointer(sage_->v_transposed);
      a.pointer(sage_->qscale);
      a.pointer(sage_->kscale);
      a.pointer(sage_->correction);
      a.pointer(attn_);
      unsigned rows = 16 * (attention_waves_ / 4);
      launch(k_attention_, "SA2 attention", unsigned((T + rows - 1) / rows),
             KV_HEADS, 32 * attention_waves_, a);
    }
    {
      KernArgs a;
      a.scalar_i32(T);
      a.pointer(attn_);
      a.pointer((char *)fused_ + GATE_OFFSET * 2);
      a.pointer(a_q_);
      a.pointer(a_s_);
      launch(k_prep_gated_, "prepare gated", T, 1, THREADS, a);
    }
    gemm(k_gemm_wo_, "gemm wo + residual", b.wo_q, b.wo_s, HIDDEN, x_, pregate);
    {
      KernArgs a;
      a.scalar_i32(T);
      a.pointer(x_);
      a.pointer(b.postnorm);
      a.pointer(postscale);
      a.pointer(postshift);
      a.pointer(a_q_);
      a.pointer(a_s_);
      launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a);
    }
    gemm(k_gemm_gu_, "gemm gate|up + swiglu", b.gu_q, b.gu_s, 2 * INTER, gu_,
         nullptr);
    {
      KernArgs a;
      a.scalar_i32(T);
      a.pointer(gu_);
      a.pointer(a_q_);
      a.pointer(a_s_);
      launch(k_prep_swiglu_, "prepare down input", T, 1, THREADS, a);
    }
    gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_,
         postgate);
  }

  int tokens_, layers_;
  size_t capacity_ = 0;
  unsigned gemm_rows_ = 0, m_group_ = 0, attention_waves_ = 4,
           attention_bits_ = 4, gemm_bits_ = 4;
  std::mutex mutex_;
  std::vector<Block> blocks_;
  std::unique_ptr<SagePreparation> sage_;
  std::shared_ptr<krea2_weights> weights_;
  void *x_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr,
       *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *attn_ = nullptr,
       *gu_ = nullptr, *mods_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;
  Kernel k_prep_norm_, k_prep_gated_, k_prep_swiglu_, k_gemm_qkvg_, k_gemm_gu_,
      k_gemm_wo_, k_gemm_down_, k_rope_, k_attention_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
  if (error && capacity)
    std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

} // namespace

struct krea2_session {
  Session value;
  krea2_session(const std::string &w, const std::string &k, int t, int l,
                std::shared_ptr<krea2_weights> weights = {})
      : value(w, k, t, l, std::move(weights)) {}
};
krea2_session *
krea2_create_shared(const std::shared_ptr<krea2_weights> &weights,
                    const std::string &kernels, int tokens, int layers) {
  if (!weights)
    throw std::invalid_argument("missing shared block weights");
  return new krea2_session("", kernels, tokens, layers, weights);
}

void krea2_run_device_bf16(krea2_session *s, uint16_t *x, size_t elements,
                           const float *mods, size_t mods_elements,
                           const float *cos, const float *sin,
                           size_t rope_elements) {
  if (!s || !x || !mods || !cos || !sin)
    throw std::invalid_argument("null device bridge argument");
  s->value.run(x, elements, mods, mods_elements, cos, sin, rope_elements, 0, -1,
               true);
}

extern "C" uint32_t krea2_abi_version(void) { return KREA2_ABI_VERSION; }
extern "C" void krea2_destroy(krea2_session *s) { delete s; }
extern "C" int krea2_profile(krea2_session *s, int enable) {
  if (!s)
    return KREA2_INVALID_ARGUMENT;
  s->value.profile = enable != 0;
  return KREA2_OK;
}

extern "C" int krea2_create(const char *weights_dir, const char *kernels_dir,
                            int tokens, int layers, krea2_session **out,
                            char *error, size_t cap) {
  if (error && cap)
    error[0] = 0;
  if (!out) {
    write_error(error, cap, "out_session must not be null");
    return KREA2_INVALID_ARGUMENT;
  }
  *out = nullptr;
  try {
    if (!weights_dir || !kernels_dir)
      throw std::invalid_argument("weights_dir and kernels_dir are required");
    *out = new krea2_session(weights_dir, kernels_dir, tokens, layers);
    return KREA2_OK;
  } catch (const std::invalid_argument &e) {
    write_error(error, cap, e.what());
    return KREA2_INVALID_ARGUMENT;
  } catch (const std::exception &e) {
    write_error(error, cap, e.what());
    return KREA2_ERROR;
  } catch (...) {
    write_error(error, cap, "unknown C++ exception");
    return KREA2_ERROR;
  }
}

extern "C" int krea2_run(krea2_session *s, uint16_t *x, size_t x_elements,
                         const float *mods, size_t mods_elements,
                         const float *cos, const float *sin,
                         size_t rope_elements, char *error, size_t cap) {
  return krea2_run_range(s, 0, -1, x, x_elements, mods, mods_elements, cos, sin,
                         rope_elements, error, cap);
}

extern "C" int krea2_run_range(krea2_session *s, int first_block,
                               int block_count, uint16_t *x, size_t x_elements,
                               const float *mods, size_t mods_elements,
                               const float *cos, const float *sin,
                               size_t rope_elements, char *error, size_t cap) {
  if (error && cap)
    error[0] = 0;
  try {
    if (!s || !x || !mods || !cos || !sin)
      throw std::invalid_argument("null argument");
    s->value.run(x, x_elements, mods, mods_elements, cos, sin, rope_elements,
                 first_block, block_count);
    return KREA2_OK;
  } catch (const std::invalid_argument &e) {
    write_error(error, cap, e.what());
    return KREA2_INVALID_ARGUMENT;
  } catch (const std::exception &e) {
    write_error(error, cap, e.what());
    return KREA2_ERROR;
  } catch (...) {
    write_error(error, cap, "unknown C++ exception");
    return KREA2_ERROR;
  }
}
