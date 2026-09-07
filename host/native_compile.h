#pragma once
#include <string>
namespace krea_native {
// Compile (or find) the block kernel bundle for a token count under
// cache_parent. sources_dir holds the .loom sources (a bundle's sources/), or
// is empty for the sources embedded in this library. bits: the block weights'
// GEMM operand width (4 or 8, krea2_weights_bits).
std::string prepare_kernels(const std::string &cache_parent,
                            const std::string &sources_dir,
                            const std::string &compiler, int tokens,
                            int bits = 4);
// $XDG_CACHE_HOME/krea2-loom (or ~/.cache/krea2-loom), created.
std::string user_cache_directory();
}
