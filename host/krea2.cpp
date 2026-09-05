// The 28 Krea 2 transformer blocks as a resident Loom session: per block ten launches
// (prepare -> fused qkv|gate GEMM -> QK-norm+RoPE (+ contiguous q/k/v) -> attention ->
// gated prepare -> wo GEMM with gated residual -> prepare -> fused gate|up GEMM ->
// SwiGLU prepare -> down GEMM with gated residual), W4A4 ConvRot throughout.
//
// Build: ./scripts/build_host.sh
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "krea2.h"
#include "sage.h"
#include "krea2_device.h"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <memory>

namespace {

constexpr int HIDDEN = 6144, KV_HEADS = 12, HEAD_DIM = 128, INTER = 16384;
constexpr int QKVG = HIDDEN + 2 * KV_HEADS * HEAD_DIM + HIDDEN;   // 15360
constexpr int GATE_OFFSET = HIDDEN + 2 * KV_HEADS * HEAD_DIM;
constexpr int THREADS = 256;

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

struct Span { size_t offset, bytes; };

__global__ void bf16_to_f16(const hip_bfloat16 *input, __half *output, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(float(input[i]));
}
__global__ void f16_to_bf16(const __half *input, hip_bfloat16 *output, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = hip_bfloat16(__half2float(input[i]));
}

std::map<std::string, Span> read_manifest(const std::string &path) {
    std::map<std::string, Span> spans;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name, dtype, shape; size_t offset, bytes;
        if (ls >> name >> offset >> bytes >> dtype >> shape) spans[name] = {offset, bytes};
    }
    return spans;
}

} // namespace

struct krea2_weights {
    std::map<std::string, Span> spans;
    std::shared_ptr<void> storage;
    explicit krea2_weights(const std::string &directory) {
        spans = read_manifest(directory + "/manifest.txt");
        const auto path = directory + "/weights.bin";
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot read " + path);
        struct File { int fd; ~File() { close(fd); } } file{fd};
        struct stat info;
        if (fstat(fd, &info) || info.st_size <= 0)
            throw std::runtime_error("invalid weight file: " + path);
        size_t bytes = size_t(info.st_size);
        for (const auto &[name, span] : spans)
            if (span.offset > bytes || span.bytes > bytes - span.offset)
                throw std::runtime_error("manifest span '" + name + "' runs past " + path);
        void *mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + path);
        struct Mapping { void *p; size_t bytes; ~Mapping() { munmap(p, bytes); } } mapping{mapped, bytes};
        void *device = nullptr;
        HIP_CHECK(hipMalloc(&device, bytes));
        storage = std::shared_ptr<void>(device, [](void *p) { (void)hipFree(p); });
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)device, mapped, bytes));
    }
};

std::shared_ptr<krea2_weights> krea2_load_weights(const std::string &directory) {
    return std::make_shared<krea2_weights>(directory);
}

namespace {

struct Kernel {
    hipModule_t module = nullptr; hipFunction_t function = nullptr;
    void load(const std::string &path, const char *symbol) {
        HIP_CHECK(hipModuleLoad(&module, path.c_str()));
        HIP_CHECK(hipModuleGetFunction(&function, module, symbol));
    }
};

struct KernArgs {
    alignas(16) unsigned char bytes[160]; size_t size = 0;
    void scalar_i32(int v) { size = (size + 3) & ~size_t(3); memcpy(bytes + size, &v, 4); size += 4; }
    void pointer(const void *p) { size = (size + 7) & ~size_t(7); memcpy(bytes + size, &p, 8); size += 8; }
};

class Session {
public:
    Session(const std::string &weights_dir, const std::string &kernels_dir, int tokens, int layers, std::shared_ptr<krea2_weights> weights = {})
        : tokens_(tokens), layers_(layers) {
        try {
        if (tokens < 16 || tokens > 16896) throw std::invalid_argument("tokens must be 16..16896");
        if (layers < 1 || layers > 28) throw std::invalid_argument("layers must be 1..28");
        HIP_CHECK(hipInit(0));
        capacity_ = std::max<size_t>((tokens + 16 + 31) / 32 * 32, (tokens + 63) / 64 * 64);   // tokens+16 headroom, whole 64-key blocks
        std::ifstream metadata(kernels_dir + "/launch.txt");
        unsigned version = 0, compiled_tokens = 0; size_t compiled_capacity = 0;
        if (!(metadata >> version >> compiled_tokens >> m_group_ >> compiled_capacity >> attention_waves_) ||
            version != 2 || compiled_tokens != unsigned(tokens) || compiled_capacity != capacity_ ||
            m_group_ < 2 || m_group_ > 4 || attention_waves_ != (tokens < 8192 ? 8u : 4u))
            throw std::invalid_argument("invalid kernel launch metadata; rebuild with scripts/build_kernels.py");
        weights_ = weights ? std::move(weights) : krea2_load_weights(weights_dir);
        const auto &spans = weights_->spans;
        auto need = [&](const std::string &name, size_t bytes) {
            auto it = spans.find(name);
            if (it == spans.end()) throw std::runtime_error("missing tensor " + name);
            if (it->second.bytes != bytes)
                throw std::runtime_error("tensor " + name + " has " + std::to_string(it->second.bytes) + " bytes, expected " + std::to_string(bytes));
            return (char *)weights_->storage.get() + it->second.offset;
        };
        for (int i = 0; i < layers; ++i) {
            std::string p = "blocks." + std::to_string(i);
            Block b;
            b.qkvg_q = need(p + ".qkvg.q", size_t(QKVG) * HIDDEN / 2); b.qkvg_s = need(p + ".qkvg.s", size_t(QKVG) * 4);
            b.wo_q = need(p + ".wo.q", size_t(HIDDEN) * HIDDEN / 2);    b.wo_s = need(p + ".wo.s", size_t(HIDDEN) * 4);
            b.gu_q = need(p + ".gu.q", size_t(2 * INTER) * HIDDEN / 2); b.gu_s = need(p + ".gu.s", size_t(2 * INTER) * 4);
            b.down_q = need(p + ".down.q", size_t(HIDDEN) * INTER / 2); b.down_s = need(p + ".down.s", size_t(HIDDEN) * 4);
            b.prenorm = need(p + ".prenorm", HIDDEN * 4); b.postnorm = need(p + ".postnorm", HIDDEN * 4);
            b.qnorm = need(p + ".qnorm", HEAD_DIM * 4); b.knorm = need(p + ".knorm", HEAD_DIM * 4);
            blocks_.push_back(b);
        }
        auto load = [&](Kernel &k, const char *stem, const char *symbol) { k.load(kernels_dir + "/" + stem + ".hsaco", symbol); };
        load(k_prep_norm_, "prepare_norm_i4", "krea2_prepare_norm_i4");
        load(k_prep_gated_, "prepare_gated_i4", "krea2_prepare_gated_i4");
        load(k_prep_swiglu_, "prepare_plain_i4", "krea2_prepare_plain_i4");
        load(k_gemm_qkvg_, "gemm_qkvg", "krea2_gemm_i4");
        load(k_gemm_gu_, "gemm_gu", "krea2_gemm_i4_swiglu");
        load(k_gemm_wo_, "gemm_wo", "krea2_gemm_i4_resid");
        load(k_gemm_down_, "gemm_down", "krea2_gemm_i4_resid");
        load(k_rope_, "rope_qknorm", "krea2_rope_qknorm_f16");
        load(k_attention_, "attention", attention_waves_ == 8 ?
             "krea2_attention_sage_i4_fast" : "krea2_attention_sage_i4_fast_prefetch");
        sage_ = std::make_unique<SagePreparation>(tokens, int(capacity_));
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&x_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&a_q_, T * INTER / 2));            // the widest prepared operand (down's K = 16384)
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * QKVG * 2));
        HIP_CHECK(hipMalloc(&q_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&k_, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMalloc(&v_, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMemset(v_, 0, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMemset(q_, 0, T * HIDDEN * 2));
        HIP_CHECK(hipMemset(k_, 0, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMalloc(&attn_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&gu_, T * INTER * 2));   // silu(gate) * up, fused into the GEMM epilogue
        HIP_CHECK(hipMalloc(&mods_, size_t(layers) * 6 * HIDDEN * 4));
        HIP_CHECK(hipMalloc(&cos_, T * HEAD_DIM * 4));
        HIP_CHECK(hipMalloc(&sin_, T * HEAD_DIM * 4));
        HIP_CHECK(hipMemset(fused_, 0, T * QKVG * 2));        // headroom rows stay zero
        HIP_CHECK(hipMemset(x_, 0, T * HIDDEN * 2));
        } catch (...) {
            release();
            throw;
        }
    }
    ~Session() { release(); }

private:
    void release() noexcept {
        for (void *p : {x_, a_q_, a_s_, fused_, q_, k_, v_, attn_, gu_, mods_, cos_, sin_}) if (p) (void)hipFree(p);
        for (Kernel *k : {&k_prep_norm_, &k_prep_gated_, &k_prep_swiglu_, &k_gemm_qkvg_, &k_gemm_gu_, &k_gemm_wo_, &k_gemm_down_, &k_rope_, &k_attention_})
            if (k->module) (void)hipModuleUnload(k->module);
    }

public:
    void run(uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements, const float *cos, const float *sin, size_t rope_elements,
             int first_block = 0, int block_count = -1, bool device_bf16 = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (first_block < 0 || first_block >= layers_)
            throw std::invalid_argument("block range must be within the loaded layers");
        if (block_count == -1) block_count = layers_ - first_block;
        if (block_count < 1 || block_count > layers_ - first_block)
            throw std::invalid_argument("block range must be within the loaded layers");
        const size_t T = tokens_;
        if (x_elements != T * HIDDEN) throw std::invalid_argument("x has " + std::to_string(x_elements) + " elements, expected " + std::to_string(T * HIDDEN));
        if (mods_elements != size_t(layers_) * 6 * HIDDEN) throw std::invalid_argument("mods has the wrong element count");
        if (rope_elements != T * HEAD_DIM) throw std::invalid_argument("cos/sin have the wrong element count");
        if (device_bf16) {
            bf16_to_f16<<<(x_elements + 255) / 256, 256>>>((const hip_bfloat16 *)x, (__half *)x_, x_elements);
            HIP_CHECK(hipGetLastError());
        } else {
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)x_, x, T * HIDDEN * 2));
        }
        const float *device_mods = mods;
        if (!device_bf16) {
            HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)mods_, mods, mods_elements * 4));
            device_mods = (const float *)mods_;
        }
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cos_, cos, rope_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)sin_, sin, rope_elements * 4));
        for (int i = first_block; i < first_block + block_count; ++i) block(i, device_mods);
        if (device_bf16) {
            f16_to_bf16<<<(x_elements + 255) / 256, 256>>>((const __half *)x_, (hip_bfloat16 *)x, x_elements);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        } else {
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpyDtoH(x, (hipDeviceptr_t)x_, T * HIDDEN * 2));
        }
        if (profile) {
            double total = 0; for (auto &e : stage_us) total += e.second;
            std::vector<std::pair<double, std::string>> rows;
            for (auto &e : stage_us) rows.push_back({e.second, e.first});
            std::sort(rows.rbegin(), rows.rend());
            fprintf(stderr, "stage profile over %d block(s), %zu tokens:\n", block_count, T);
            for (auto &r : rows) fprintf(stderr, "  %-24s %9.3f ms  %5.1f%%\n", r.second.c_str(), r.first / 1000.0, 100.0 * r.first / total);
            fprintf(stderr, "  %-24s %9.3f ms\n", "total", total / 1000.0);
            stage_us.clear();
        }
    }

    bool profile = false;
    std::map<std::string, double> stage_us;

private:
    struct Block { char *qkvg_q, *qkvg_s, *wo_q, *wo_s, *gu_q, *gu_s, *down_q, *down_s, *prenorm, *postnorm, *qnorm, *knorm; };

    void launch(Kernel &k, const char *stage, unsigned gx, unsigned gy, unsigned bx, KernArgs &args) {
        std::chrono::steady_clock::time_point t0;
        if (profile) { HIP_CHECK(hipDeviceSynchronize()); t0 = std::chrono::steady_clock::now(); }
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.bytes, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args.size, HIP_LAUNCH_PARAM_END};
        HIP_CHECK(hipModuleLaunchKernel(k.function, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config));
        if (profile) {
            HIP_CHECK(hipDeviceSynchronize());
            stage_us[stage] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    unsigned gemm_grid_y(size_t m) const { return unsigned(((m + 127) / 128 + m_group_ - 1) / m_group_ * m_group_); }

    void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n, void *out, const float *gate) {
        KernArgs a;
        a.scalar_i32(int(tokens_));
        a.pointer(a_q_); a.pointer(w_q); a.pointer(w_s); a.pointer(a_s_); a.pointer(out);
        if (gate) a.pointer(gate);
        launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
    }

    void block(int i, const float *mods) {
        const Block &b = blocks_[i];
        const float *mod = mods + size_t(i) * 6 * HIDDEN;
        const float *prescale = mod, *preshift = mod + HIDDEN, *pregate = mod + 2 * HIDDEN;
        const float *postscale = mod + 3 * HIDDEN, *postshift = mod + 4 * HIDDEN, *postgate = mod + 5 * HIDDEN;
        const unsigned T = unsigned(tokens_);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.prenorm); a.pointer(prescale); a.pointer(preshift); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a); }
        gemm(k_gemm_qkvg_, "gemm qkv|gate", b.qkvg_q, b.qkvg_s, QKVG, fused_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(fused_); a.pointer(b.qnorm); a.pointer(b.knorm); a.pointer(cos_); a.pointer(sin_); a.pointer(q_); a.pointer(k_); a.pointer(v_);
          launch(k_rope_, "qk norm + rope", T, 1, THREADS, a); }
        {
          auto start = std::chrono::steady_clock::now();
          if (profile) { HIP_CHECK(hipDeviceSynchronize()); start = std::chrono::steady_clock::now(); }
          sage_->run(q_, k_, v_);
          if (profile) {
            HIP_CHECK(hipDeviceSynchronize());
            stage_us["SA2 preprocessing"] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
          }
          KernArgs a; a.scalar_i32(T); a.scalar_i32(KV_HEADS);
          a.pointer(sage_->q4); a.pointer(sage_->k4); a.pointer(sage_->v_transposed);
          a.pointer(sage_->qscale); a.pointer(sage_->kscale); a.pointer(sage_->correction); a.pointer(attn_);
          unsigned rows = 16 * (attention_waves_ / 4);
          launch(k_attention_, "SA2 attention", unsigned((T + rows - 1) / rows), KV_HEADS, 32 * attention_waves_, a);
        }
        { KernArgs a; a.scalar_i32(T); a.pointer(attn_); a.pointer((char *)fused_ + GATE_OFFSET * 2); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_gated_, "prepare gated", T, 1, THREADS, a); }
        gemm(k_gemm_wo_, "gemm wo + residual", b.wo_q, b.wo_s, HIDDEN, x_, pregate);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.postnorm); a.pointer(postscale); a.pointer(postshift); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a); }
        gemm(k_gemm_gu_, "gemm gate|up + swiglu", b.gu_q, b.gu_s, 2 * INTER, gu_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(gu_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_swiglu_, "prepare down input", T, 1, THREADS, a); }
        gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_, postgate);
    }

    int tokens_, layers_;
    size_t capacity_ = 0;
    unsigned m_group_ = 0, attention_waves_ = 4;
    std::mutex mutex_;
    std::vector<Block> blocks_;
    std::unique_ptr<SagePreparation> sage_;
    std::shared_ptr<krea2_weights> weights_;
    void *x_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr,
         *attn_ = nullptr, *gu_ = nullptr, *mods_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;
    Kernel k_prep_norm_, k_prep_gated_, k_prep_swiglu_, k_gemm_qkvg_, k_gemm_gu_, k_gemm_wo_, k_gemm_down_, k_rope_, k_attention_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

}  // namespace

struct krea2_session {
    Session value;
    krea2_session(const std::string &w, const std::string &k, int t, int l,
                  std::shared_ptr<krea2_weights> weights = {}) : value(w, k, t, l, std::move(weights)) {}
};
krea2_session *krea2_create_shared(const std::shared_ptr<krea2_weights> &weights,
                                    const std::string &kernels, int tokens, int layers) {
    if (!weights) throw std::invalid_argument("missing shared block weights");
    return new krea2_session("", kernels, tokens, layers, weights);
}

void krea2_run_device_bf16(krea2_session *s, uint16_t *x, size_t elements,
                           const float *mods, size_t mods_elements,
                           const float *cos, const float *sin, size_t rope_elements) {
    if (!s || !x || !mods || !cos || !sin) throw std::invalid_argument("null device bridge argument");
    s->value.run(x, elements, mods, mods_elements, cos, sin, rope_elements, 0, -1, true);
}

extern "C" uint32_t krea2_abi_version(void) { return KREA2_ABI_VERSION; }
extern "C" void krea2_destroy(krea2_session *s) { delete s; }
extern "C" int krea2_profile(krea2_session *s, int enable) { if (!s) return KREA2_INVALID_ARGUMENT; s->value.profile = enable != 0; return KREA2_OK; }

extern "C" int krea2_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers, krea2_session **out, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    if (!out) { write_error(error, cap, "out_session must not be null"); return KREA2_INVALID_ARGUMENT; }
    *out = nullptr;
    try {
        if (!weights_dir || !kernels_dir) throw std::invalid_argument("weights_dir and kernels_dir are required");
        *out = new krea2_session(weights_dir, kernels_dir, tokens, layers);
        return KREA2_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return KREA2_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return KREA2_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return KREA2_ERROR; }
}

extern "C" int krea2_run(krea2_session *s, uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements,
                         const float *cos, const float *sin, size_t rope_elements, char *error, size_t cap) {
    return krea2_run_range(s, 0, -1, x, x_elements, mods, mods_elements, cos, sin, rope_elements, error, cap);
}

extern "C" int krea2_run_range(krea2_session *s, int first_block, int block_count,
                               uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements,
                               const float *cos, const float *sin, size_t rope_elements, char *error, size_t cap) {
    if (error && cap) error[0] = 0;
    try {
        if (!s || !x || !mods || !cos || !sin) throw std::invalid_argument("null argument");
        s->value.run(x, x_elements, mods, mods_elements, cos, sin, rope_elements, first_block, block_count);
        return KREA2_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return KREA2_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return KREA2_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return KREA2_ERROR; }
}
