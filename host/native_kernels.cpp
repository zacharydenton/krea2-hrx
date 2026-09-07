#include "native_kernels.h"
#include "sha256.h"
#include "compiler_spawn.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sys/file.h>
#include <unistd.h>
namespace krea_native {
#include "native_sources.h"
namespace {
std::mutex mutex;
std::string compiler;
struct KernelProfile {
  std::map<std::string, std::pair<size_t, double>> totals;
  ~KernelProfile() {
    for (const auto &[key, value] : totals)
      std::fprintf(stderr, "native kernel profile: %zu calls %.3f ms %s\n",
                   value.first, value.second, key.c_str());
  }
};
std::map<std::string, gpu::Kernel> &kernel_cache() {
  gpu::initialize();
  static std::map<std::string, gpu::Kernel> cache;
  return cache;
}
std::string read(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}
} // namespace
void native_compiler(const std::string &path) {
  std::lock_guard lock(mutex);
  compiler = path;
}
void native_launch(const std::string &name, const Config &input_config,
                   const gpu::Args &args, unsigned gx, unsigned gy,
                   unsigned threads) {
  std::lock_guard lock(mutex);
  Config config = input_config;
  if (name != "sage_transpose") {
    config["grid_x"] = gx;
    config["grid_y"] = gy;
  }
  std::string signature = name + "\n";
  for (const auto &[key, value] : config)
    signature += key + "=" + std::to_string(value) + "\n";
  auto &loaded = kernel_cache();
  auto it = loaded.find(signature);
  if (it == loaded.end()) {
    const auto &source = native_sources.at(name);
    const auto key = sha256(source + signature);
    const char *cache = std::getenv("XDG_CACHE_HOME"),
               *home = std::getenv("HOME");
    std::filesystem::path root = cache  ? cache
                                 : home ? std::string(home) + "/.cache"
                                        : "/tmp";
    root /= "krea2-loom";
    root /= "native-gfx1151-v1";
    std::filesystem::create_directories(root);
    auto path = root / (key + ".hsaco"), hashfile = root / (key + ".sha256");
    // A populated cache is usable without write access: verify before locking.
    auto cached = [&] {
      return std::filesystem::exists(path) && std::filesystem::exists(hashfile);
    };
    if (!cached()) {
      int fd =
          open((root / ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
      if (fd < 0)
        throw std::runtime_error("cannot open auxiliary kernel cache lock " +
                                 (root / ".lock").string() + ": " +
                                 std::strerror(errno));
      struct Lock {
        int fd;
        ~Lock() { close(fd); }
      } lockfile{fd};
      if (flock(fd, LOCK_EX))
        throw std::runtime_error("cannot lock auxiliary kernel cache");
      if (!cached()) {
        auto src = root / (key + ".loom"), tmp = root / (key + ".tmp"),
             log = root / (key + ".log");
        struct Scratch {
          std::filesystem::path src, tmp, log;
          ~Scratch() {
            std::error_code e;
            std::filesystem::remove(src, e);
            std::filesystem::remove(tmp, e);
            std::filesystem::remove(log, e);
          }
        } scratch{src, tmp, log};
        std::ofstream(src) << source;
        const char *env = std::getenv("LOOM_COMPILE");
        std::string exe =
            compiler.empty() ? (env ? env : "loom-compile") : compiler;
        std::vector<std::string> cmd = {exe,
                                        src.string(),
                                        "--backend=amdgpu-hal",
                                        "--target=gfx1151",
                                        "--root=@krea2_" + name,
                                        "--output=" + tmp.string()};
        for (const auto &[key, value] : config)
          cmd.push_back("--config=krea2." + name + "." + key + "=" +
                        std::to_string(value));
        run_compiler(cmd, log, name);
        std::ofstream(hashfile) << sha256(read(tmp));
        std::filesystem::rename(tmp, path);
      }
    }
    if (read(hashfile) != sha256(read(path)))
      throw std::runtime_error("corrupt auxiliary kernel: " + path.string());
    it = loaded.emplace(signature, gpu::Kernel(path.string(), "krea2_" + name))
             .first;
  }
  const char *profile = std::getenv("KREA2_NATIVE_KERNEL_PROFILE");
  if (profile && std::strcmp(profile, "1") == 0) {
    gpu::synchronize();
    auto start = std::chrono::steady_clock::now();
    it->second.launch(gx, gy, threads, args);
    gpu::synchronize();
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
    static KernelProfile timing;
    std::string key = name;
    for (const auto &[k, v] : input_config)
      key += " " + k + "=" + std::to_string(v);
    auto &total = timing.totals[key];
    ++total.first;
    total.second += ms;
  } else {
    it->second.launch(gx, gy, threads, args);
  }
}
} // namespace krea_native
