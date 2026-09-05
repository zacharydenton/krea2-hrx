#include "native_kernels.h"
#include "sha256.h"
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace krea_native {
#include "native_sources.h"
namespace {
std::mutex mutex;
std::string compiler;
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
    int fd = open((root / ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
      throw std::runtime_error("cannot open auxiliary kernel cache lock");
    struct Lock {
      int fd;
      ~Lock() { close(fd); }
    } lockfile{fd};
    if (flock(fd, LOCK_EX))
      throw std::runtime_error("cannot lock auxiliary kernel cache");
    if (!std::filesystem::exists(path) || !std::filesystem::exists(hashfile)) {
      auto src = root / (key + ".loom"), tmp = root / (key + ".tmp");
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
      std::vector<char *> argv;
      for (auto &s : cmd)
        argv.push_back(s.data());
      argv.push_back(nullptr);
      pid_t pid;
      int error = posix_spawnp(&pid, exe.c_str(), nullptr, nullptr, argv.data(),
                               environ);
      if (error)
        throw std::runtime_error("cannot start Loom compiler: " + exe);
      int status;
      while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR)
          throw std::runtime_error("cannot wait for Loom compiler");
      }
      if (!WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("Loom compilation failed: " + name);
      std::ofstream(hashfile) << sha256(read(tmp));
      std::filesystem::rename(tmp, path);
      std::filesystem::remove(src);
    }
    if (read(hashfile) != sha256(read(path)))
      throw std::runtime_error("corrupt auxiliary kernel: " + path.string());
    it = loaded.emplace(signature, gpu::Kernel(path.string(), "krea2_" + name))
             .first;
  }
  it->second.launch(gx, gy, threads, args);
}
} // namespace krea_native
