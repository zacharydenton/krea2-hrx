#pragma once
#include <string>
namespace krea_native {
std::string prepare_kernels(const std::string &bundle,
                            const std::string &compiler, int tokens);
}
