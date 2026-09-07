#include "native_compile.h"
#include "block_sources.h"
#include "compiler_spawn.h"
#include "gemm_shape.h"
#include "sha256.h"
#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
namespace krea_native {
static std::string digest(const std::string &bytes) { return sha256(bytes); }
static std::string contents(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  return std::string((std::istreambuf_iterator<char>(input)), {});
}
std::string user_cache_directory() {
  const char *cache = std::getenv("XDG_CACHE_HOME"), *home = std::getenv("HOME");
  std::filesystem::path root = cache  ? cache
                               : home ? std::string(home) + "/.cache"
                                      : "/tmp";
  root /= "krea2-loom";
  std::filesystem::create_directories(root);
  return root.string();
}
std::string prepare_kernels(const std::string &cache_parent,
                            const std::string &compiler, int tokens,
                            int bits) {
  if (tokens < 16 || tokens > 16896)
    throw std::invalid_argument("tokens must be 16..16896");
  if (bits != 4 && bits != 8)
    throw std::invalid_argument("GEMM operand width must be 4 or 8");
  const int attention_waves = tokens < 8192 ? 8 : 4;
  // KREA2_ATTN_QK: 16 (default) f16 QK and PV, ComfyUI's SDPA class; 4 / 8 the
  // smoothed int4 / int8 QK kernels (faster, further from ComfyUI).
  const char *qk = std::getenv("KREA2_ATTN_QK");
  const int attention_bits = qk && *qk ? std::atoi(qk) : 16;
  if (attention_bits != 4 && attention_bits != 8 && attention_bits != 16)
    throw std::invalid_argument("KREA2_ATTN_QK must be 4, 8 or 16");
  namespace fs = std::filesystem;
  fs::path parent = cache_parent;
  auto source_text = [&](const std::string &name) {
    auto it = block_sources().find(name);
    if (it == block_sources().end())
      throw std::runtime_error("no embedded kernel source: " + name);
    return it->second;
  };
  int capacity = std::max((tokens + 47) / 32 * 32, (tokens + 63) / 64 * 64);
  // The GEMM tile, raster group and operand pitches come from gemm_shape.h,
  // which scripts/build_kernels.py mirrors; the session checks every field.
  const int rows = krea2_shape::gemm_rows(tokens, bits);
  const int m_group = krea2_shape::gemm_m_group(tokens, rows);
  const std::string pitch_hidden =
                        std::to_string(krea2_shape::gemm_pitch(6144, bits)),
                    pitch_inter =
                        std::to_string(krea2_shape::gemm_pitch(16384, bits)),
                    ib = "i" + std::to_string(bits);
  auto group = std::to_string(m_group);
  std::string metadata = "3 " + std::to_string(tokens) + " " +
                         std::to_string(rows) + " " + group + " " +
                         std::to_string(capacity) + " " +
                         std::to_string(attention_waves) + " " +
                         pitch_hidden + " " + pitch_inter + " " +
                         std::to_string(attention_bits) + " " +
                         std::to_string(bits) + "\n";
  struct Job {
    std::string source, symbol, stem;
    std::map<std::string, std::string> cfg;
  };
  std::vector<Job> jobs;
  auto add = [&](std::string src, std::string stem,
                 std::map<std::string, std::string> cfg) {
    jobs.push_back({src, "krea2_" + src, stem, cfg});
  };
  add("prepare_norm_" + ib, "prepare_norm_" + ib,
      {{"width", "6144"}, {"out_stride", pitch_hidden}, {"eps", "1e-5"}});
  add("prepare_gated_" + ib, "prepare_gated_" + ib,
      {{"width", "6144"},
       {"out_stride", pitch_hidden},
       {"gate_stride", "15360"}});
  add("prepare_plain_" + ib, "prepare_plain_" + ib,
      {{"width", "16384"}, {"out_stride", pitch_inter}});
  const std::string gemm = rows == 256 ? "_256" : "";
  add("gemm_" + ib + gemm, "gemm_qkvg",
      {{"k_size", "6144"},
       {"k_stride", pitch_hidden},
       {"n_size", "15360"},
       {"m_group", group}});
  add("gemm_" + ib + "_swiglu" + gemm, "gemm_gu",
      {{"k_size", "6144"},
       {"k_stride", pitch_hidden},
       {"n_size", "32768"},
       {"m_group", group}});
  add("gemm_" + ib + "_resid" + gemm, "gemm_wo",
      {{"k_size", "6144"},
       {"k_stride", pitch_hidden},
       {"n_size", "6144"},
       {"m_group", group}});
  add("gemm_" + ib + "_resid" + gemm, "gemm_down",
      {{"k_size", "16384"},
       {"k_stride", pitch_inter},
       {"n_size", "6144"},
       {"m_group", group}});
  add("rope_qknorm_f16", "rope_qknorm",
      {{"row_stride", "15360"},
       {"q_heads", "48"},
       {"kv_heads", "12"},
       {"k_offset", "6144"},
       {"eps", "1e-5"}});
  add(attention_bits == 16
          ? std::string("attention_gqa_lds_f16_wmma")
          : std::string(attention_bits == 4 ? "attention_sage_i4_fast"
                                            : "attention_sage_i8_fast") +
                (attention_waves == 8 ? "" : "_prefetch"),
      "attention",
      {{"q_stride", "6144"},
       {"kv_stride", "1536"},
       {"tokens", std::to_string(tokens)},
       {"token_capacity", std::to_string(capacity)},
       {"scale", "0.08838834764831845"},
       {"out_stride", "6144"}});
  // Bundles are deployable without the build-machine compiler. Source/config
  // fingerprints select immutable artifacts; compiler provenance is recorded.
  std::string signature =
      "native-kernels-v4:gfx1151:sage-prep-v3:64:vt\n" + metadata;
  for (const auto &j : jobs) {
    signature += source_text(j.source);
    signature += j.source + j.symbol + j.stem;
    for (const auto &[k, v] : j.cfg)
      signature += k + "=" + v + "\n";
  }
  auto out = parent / ("T" + std::to_string(tokens) + "-" + digest(signature));
  auto verify = [&] {
    auto hashes = nlohmann::json::parse(contents(out / "manifest.json"));
    if (contents(out / "launch.txt") != metadata)
      throw std::runtime_error("invalid native launch metadata");
    for (const auto &j : jobs) {
      std::string name = j.stem + ".hsaco";
      if (digest(contents(out / name)) != hashes.at(name).get<std::string>())
        throw std::runtime_error("corrupt native kernel: " + name);
    }
  };
  if (fs::exists(out)) {
    verify();
    return out.string();
  }
  fs::create_directories(parent);
  int fd = open((parent / ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0)
    throw std::runtime_error("cannot lock kernel cache");
  struct Lock {
    int fd;
    ~Lock() { close(fd); }
  } lock{fd};
  if (flock(fd, LOCK_EX))
    throw std::runtime_error("kernel cache lock failed");
  if (fs::exists(out)) {
    verify();
    return out.string();
  }
  auto staging = parent / (".prepare-" + std::to_string(getpid()));
  // A dead process may have left a staging directory with this recycled PID.
  fs::remove_all(staging);
  fs::create_directory(staging);
  struct Cleanup {
    fs::path p;
    ~Cleanup() {
      std::error_code e;
      fs::remove_all(p, e);
    }
  } cleanup{staging};
  for (const auto &j : jobs) {
    fs::path source_path = staging / (j.source + ".loom");
    std::ofstream(source_path) << source_text(j.source);
    std::vector<std::string> args = {
        compiler,
        source_path.string(),
        "--backend=amdgpu-hal",
        "--target=gfx1151",
        "--root=@" + j.symbol,
        "--output=" + (staging / (j.stem + ".hsaco")).string()};
    for (const auto &[k, v] : j.cfg)
      args.push_back("--config=krea2." + j.source + "." + k + "=" + v);
    run_compiler(args, staging / (j.stem + ".log"), j.source);
    fs::remove(staging / (j.stem + ".log"));
    fs::remove(source_path);
  }
  std::ofstream(staging / "launch.txt") << metadata;
  std::ofstream(staging / "signature", std::ios::binary) << signature;
  std::ofstream(staging / "compiler.txt") << compiler << "\n";
  nlohmann::json hashes;
  for (const auto &j : jobs) {
    std::string name = j.stem + ".hsaco";
    hashes[name] = digest(contents(staging / name));
  }
  std::ofstream(staging / "manifest.json") << hashes.dump();
  fs::rename(staging, out);
  return out.string();
}
} // namespace krea_native
