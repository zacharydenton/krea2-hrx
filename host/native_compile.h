#pragma once
#include <string>
namespace krea_native {
// bits: the block weights' GEMM operand width (4 or 8, krea2_weights_bits).
std::string prepare_kernels(const std::string &bundle,
                            const std::string &compiler, int tokens,
                            int bits = 4);
}
