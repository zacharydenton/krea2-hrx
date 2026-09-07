#pragma once
#include <string>
namespace krea_native {
// Compile (or find) the block kernel bundle for a token count under
// cache_parent, from the kernel sources embedded in this library. bits: the
// block weights' GEMM operand width (8 for the int8 ConvRot checkpoints).
std::string prepare_kernels(const std::string &cache_parent,
                            const std::string &compiler, int tokens,
                            int bits);
// $XDG_CACHE_HOME/krea2-loom (or ~/.cache/krea2-loom), created.
std::string user_cache_directory();
}
