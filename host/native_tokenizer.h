#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace krea_native {
class Tokenizer {
  struct Impl;
  std::unique_ptr<Impl> impl;
  void load(const std::string &json_text);

public:
  explicit Tokenizer(const std::string &path);
  // From the JSON text itself (the embedded tokenizer).
  Tokenizer(const char *json_text, size_t size);
  ~Tokenizer();
  std::vector<int32_t> encode(const std::string &text) const;
  std::vector<int32_t> prompt(const std::string &text) const;
};
} // namespace krea_native
