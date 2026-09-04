// The 28 Krea 2 transformer blocks as a resident Loom session: per block ten launches
// (prepare -> fused qkv|gate GEMM -> QK-norm+RoPE -> V transpose -> attention ->
// gated prepare -> wo GEMM with gated residual -> prepare -> fused gate|up GEMM ->
// SwiGLU prepare -> down GEMM with gated residual), W4A4 ConvRot throughout.
//
// Build: ./scripts/build_host.sh
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "krea2.h"

namespace {

constexpr int HIDDEN = 6144, HEADS = 48, KV_HEADS = 12, HEAD_DIM = 128, INTER = 16384;
constexpr int QKVG = HIDDEN + 2 * KV_HEADS * HEAD_DIM + HIDDEN;   // 15360
constexpr int K_OFFSET = HIDDEN, V_OFFSET = HIDDEN + KV_HEADS * HEAD_DIM, GATE_OFFSET = V_OFFSET + KV_HEADS * HEAD_DIM;
constexpr int THREADS = 256;

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(e_)); } while (0)

struct Span { size_t offset, bytes; };

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

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<char> buffer(in.tellg());
    in.seekg(0);
    in.read(buffer.data(), buffer.size());
    return buffer;
}

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
    Session(const std::string &weights_dir, const std::string &kernels_dir, int tokens, int layers)
        : tokens_(tokens), layers_(layers) {
        if (tokens < 16 || tokens > 65536) throw std::invalid_argument("tokens must be 16..65536");
        if (layers < 1 || layers > 28) throw std::invalid_argument("layers must be 1..28");
        HIP_CHECK(hipInit(0));
        capacity_ = (tokens + 16 + 31) / 32 * 32;                  // >= tokens + 16 for attention, a multiple of 32 for the transpose
        auto spans = read_manifest(weights_dir + "/manifest.txt");
        auto blob = read_file(weights_dir + "/weights.bin");
        for (const auto &e : spans)
            if (e.second.offset + e.second.bytes > blob.size())
                throw std::runtime_error("manifest span '" + e.first + "' runs past weights.bin");
        HIP_CHECK(hipMalloc(&weights_, blob.size()));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)weights_, blob.data(), blob.size()));
        auto need = [&](const std::string &name, size_t bytes) {
            auto it = spans.find(name);
            if (it == spans.end()) throw std::runtime_error("missing tensor " + name);
            if (it->second.bytes != bytes)
                throw std::runtime_error("tensor " + name + " has " + std::to_string(it->second.bytes) + " bytes, expected " + std::to_string(bytes));
            return (char *)weights_ + it->second.offset;
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
        load(k_prep_swiglu_, "prepare_swiglu_i4", "krea2_prepare_swiglu_i4");
        load(k_gemm_qkvg_, "gemm_qkvg", "krea2_gemm_i4");
        load(k_gemm_gu_, "gemm_gu", "krea2_gemm_i4");
        load(k_gemm_wo_, "gemm_wo", "krea2_gemm_i4_resid");
        load(k_gemm_down_, "gemm_down", "krea2_gemm_i4_resid");
        load(k_rope_, "rope_qknorm", "krea2_rope_qknorm_f16");
        load(k_transpose_, "transpose_v", "krea2_transpose_f16");
        load(k_attention_, "attention", "krea2_attention_gqa_f16_wmma");
        const size_t T = capacity_;
        HIP_CHECK(hipMalloc(&x_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&a_q_, T * INTER / 2));            // the widest prepared operand (down's K = 16384)
        HIP_CHECK(hipMalloc(&a_s_, T * 4));
        HIP_CHECK(hipMalloc(&fused_, T * QKVG * 2));
        HIP_CHECK(hipMalloc(&vt_, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMalloc(&attn_, T * HIDDEN * 2));
        HIP_CHECK(hipMalloc(&gu_, T * 2 * INTER * 2));
        HIP_CHECK(hipMalloc(&mods_, size_t(layers) * 6 * HIDDEN * 4));
        HIP_CHECK(hipMalloc(&cos_, T * HEAD_DIM * 4));
        HIP_CHECK(hipMalloc(&sin_, T * HEAD_DIM * 4));
        HIP_CHECK(hipMemset(fused_, 0, T * QKVG * 2));        // headroom rows stay zero
        HIP_CHECK(hipMemset(vt_, 0, size_t(KV_HEADS * HEAD_DIM) * T * 2));
        HIP_CHECK(hipMemset(x_, 0, T * HIDDEN * 2));
    }
    ~Session() {
        for (void *p : {x_, a_q_, a_s_, fused_, vt_, attn_, gu_, mods_, cos_, sin_, weights_}) if (p) (void)hipFree(p);
        for (Kernel *k : {&k_prep_norm_, &k_prep_gated_, &k_prep_swiglu_, &k_gemm_qkvg_, &k_gemm_gu_, &k_gemm_wo_, &k_gemm_down_, &k_rope_, &k_transpose_, &k_attention_})
            if (k->module) (void)hipModuleUnload(k->module);
    }

    void run(uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements, const float *cos, const float *sin, size_t rope_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t T = tokens_;
        if (x_elements != T * HIDDEN) throw std::invalid_argument("x has " + std::to_string(x_elements) + " elements, expected " + std::to_string(T * HIDDEN));
        if (mods_elements != size_t(layers_) * 6 * HIDDEN) throw std::invalid_argument("mods has the wrong element count");
        if (rope_elements != T * HEAD_DIM) throw std::invalid_argument("cos/sin have the wrong element count");
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)x_, x, T * HIDDEN * 2));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)mods_, mods, mods_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)cos_, cos, rope_elements * 4));
        HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)sin_, sin, rope_elements * 4));
        for (int i = 0; i < layers_; ++i) block(i);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpyDtoH(x, (hipDeviceptr_t)x_, T * HIDDEN * 2));
        if (profile) {
            double total = 0; for (auto &e : stage_us) total += e.second;
            std::vector<std::pair<double, std::string>> rows;
            for (auto &e : stage_us) rows.push_back({e.second, e.first});
            std::sort(rows.rbegin(), rows.rend());
            fprintf(stderr, "stage profile over %d block(s), %zu tokens:\n", layers_, T);
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
    static unsigned gemm_grid_y(size_t m) { return unsigned((m + 127) / 128 + 3) / 4 * 4; }   // grouped raster: whole groups of 4

    void gemm(Kernel &k, const char *stage, char *w_q, char *w_s, int n, void *out, const float *gate) {
        KernArgs a;
        a.scalar_i32(int(tokens_));
        a.pointer(a_q_); a.pointer(w_q); a.pointer(w_s); a.pointer(a_s_); a.pointer(out);
        if (gate) a.pointer(gate);
        launch(k, stage, unsigned(n / 128), gemm_grid_y(tokens_), THREADS, a);
    }

    void block(int i) {
        const Block &b = blocks_[i];
        const float *mod = (const float *)mods_ + size_t(i) * 6 * HIDDEN;
        const float *prescale = mod, *preshift = mod + HIDDEN, *pregate = mod + 2 * HIDDEN;
        const float *postscale = mod + 3 * HIDDEN, *postshift = mod + 4 * HIDDEN, *postgate = mod + 5 * HIDDEN;
        const unsigned T = unsigned(tokens_);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.prenorm); a.pointer(prescale); a.pointer(preshift); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a); }
        gemm(k_gemm_qkvg_, "gemm qkv|gate", b.qkvg_q, b.qkvg_s, QKVG, fused_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(fused_); a.pointer(b.qnorm); a.pointer(b.knorm); a.pointer(cos_); a.pointer(sin_);
          launch(k_rope_, "qk norm + rope", T, 1, THREADS, a); }
        { KernArgs a; a.scalar_i32(T); a.pointer((char *)fused_ + V_OFFSET * 2); a.pointer(vt_);
          launch(k_transpose_, "v transpose", unsigned(KV_HEADS * HEAD_DIM / 32), unsigned(capacity_ / 32), THREADS, a); }
        { KernArgs a; a.scalar_i32(T); a.scalar_i32(HEADS); a.pointer(fused_); a.pointer((char *)fused_ + K_OFFSET * 2); a.pointer(vt_); a.pointer(attn_);
          launch(k_attention_, "attention", unsigned((T + 15) / 16), HEADS, 32, a); }
        { KernArgs a; a.scalar_i32(T); a.pointer(attn_); a.pointer((char *)fused_ + GATE_OFFSET * 2); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_gated_, "prepare gated", T, 1, THREADS, a); }
        gemm(k_gemm_wo_, "gemm wo + residual", b.wo_q, b.wo_s, HIDDEN, x_, pregate);
        { KernArgs a; a.scalar_i32(T); a.pointer(x_); a.pointer(b.postnorm); a.pointer(postscale); a.pointer(postshift); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_norm_, "prepare norm", T, 1, THREADS, a); }
        gemm(k_gemm_gu_, "gemm gate|up", b.gu_q, b.gu_s, 2 * INTER, gu_, nullptr);
        { KernArgs a; a.scalar_i32(T); a.pointer(gu_); a.pointer(a_q_); a.pointer(a_s_);
          launch(k_prep_swiglu_, "prepare swiglu", T, 1, THREADS, a); }
        gemm(k_gemm_down_, "gemm down + residual", b.down_q, b.down_s, HIDDEN, x_, postgate);
    }

    int tokens_, layers_;
    size_t capacity_ = 0;
    std::mutex mutex_;
    std::vector<Block> blocks_;
    void *weights_ = nullptr, *x_ = nullptr, *a_q_ = nullptr, *a_s_ = nullptr, *fused_ = nullptr, *vt_ = nullptr,
         *attn_ = nullptr, *gu_ = nullptr, *mods_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;
    Kernel k_prep_norm_, k_prep_gated_, k_prep_swiglu_, k_gemm_qkvg_, k_gemm_gu_, k_gemm_wo_, k_gemm_down_, k_rope_, k_transpose_, k_attention_;
};

void write_error(char *error, size_t capacity, const char *message) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", message ? message : "unknown error");
}

}  // namespace

struct krea2_session { Session value; krea2_session(const char *w, const char *k, int t, int l) : value(w, k, t, l) {} };

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
    if (error && cap) error[0] = 0;
    try {
        if (!s || !x || !mods || !cos || !sin) throw std::invalid_argument("null argument");
        s->value.run(x, x_elements, mods, mods_elements, cos, sin, rope_elements);
        return KREA2_OK;
    } catch (const std::invalid_argument &e) { write_error(error, cap, e.what()); return KREA2_INVALID_ARGUMENT; }
      catch (const std::exception &e) { write_error(error, cap, e.what()); return KREA2_ERROR; }
      catch (...) { write_error(error, cap, "unknown C++ exception"); return KREA2_ERROR; }
}
