#include "../host/native_tokenizer.h"
#include "../host/sha256.h"
#include "../host/unicode.h"
#include <cstdio>
extern "C" void *tok_create(const char *path) {
  try {
    return new krea_native::Tokenizer(path);
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    return nullptr;
  }
}
extern "C" void tok_free(void *p) {
  delete static_cast<krea_native::Tokenizer *>(p);
}
extern "C" int tok_encode(void *p, const char *text, int32_t *out) {
  try {
    auto v = static_cast<krea_native::Tokenizer *>(p)->encode(text);
    std::copy(v.begin(), v.end(), out);
    return v.size();
  } catch (...) {
    return -1;
  }
}
extern "C" int nfc(const char *text, char *out) {
  try {
    std::string s;
    for (auto c : krea_native::unicode::nfc(text))
      krea_native::unicode::append(s, c);
    std::memcpy(out, s.c_str(), s.size() + 1);
    return s.size();
  } catch (...) {
    return -1;
  }
}
extern "C" void hash(const char *text, size_t n, char *out) {
  auto s = krea_native::sha256(std::string(text, n));
  std::memcpy(out, s.c_str(), 65);
}
