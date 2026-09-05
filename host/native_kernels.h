#pragma once
#include "gpu.h"
#include <map>
#include <string>
namespace krea_native {
using Config = std::map<std::string, size_t>;
void native_launch(const std::string &name, const Config &config,
                   const gpu::Args &args, unsigned gx, unsigned gy = 1,
                   unsigned threads = 256);
void native_compiler(const std::string &path);
} // namespace krea_native
