#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace krea_native {
class Tokenizer {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  explicit Tokenizer(const std::string &path);
  ~Tokenizer();
  std::vector<int32_t> encode(const std::string &text) const;
  std::vector<int32_t> prompt(const std::string &text) const;
};
} // namespace krea_native
