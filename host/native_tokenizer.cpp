#define PCRE2_CODE_UNIT_WIDTH 8
#include "native_tokenizer.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <pcre2.h>
#include <stdexcept>
#include <unicode/normalizer2.h>
#include <unicode/ustring.h>
#include <unordered_map>

namespace krea_native {
using json = nlohmann::json;
struct Tokenizer::Impl {
  std::map<unsigned, unsigned> reverse;
  std::unordered_map<std::string, int32_t> vocab;
  std::map<std::pair<std::string, std::string>, int> merges;
  std::vector<std::pair<std::string, int32_t>> special;
  pcre2_code *regex = nullptr;
  Impl() {
    unsigned extra = 256;
    for (unsigned i = 0; i < 256; ++i)
      if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174)
        reverse[i] = i;
      else
        reverse[extra++] = i;
  }
  ~Impl() {
    if (regex)
      pcre2_code_free(regex);
  }
  std::string unbyte(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
      unsigned c = (unsigned char)s[i++];
      if (c >= 192) {
        int n = c >= 240 ? 3 : c >= 224 ? 2 : 1;
        c &= (1u << (6 - n)) - 1;
        while (n--) {
          if (i >= s.size())
            throw std::runtime_error("invalid tokenizer UTF-8");
          c = (c << 6) | ((unsigned char)s[i++] & 63);
        }
      }
      auto it = reverse.find(c);
      if (it == reverse.end())
        throw std::runtime_error("unsupported tokenizer alphabet");
      out.push_back(char(it->second));
    }
    return out;
  }
  void word(const std::string &s, std::vector<int32_t> &out) const {
    std::vector<std::string> pieces;
    for (char c : s)
      pieces.emplace_back(1, c);
    while (pieces.size() > 1) {
      int best = INT32_MAX;
      size_t at = 0;
      for (size_t i = 0; i + 1 < pieces.size(); ++i) {
        auto it = merges.find({pieces[i], pieces[i + 1]});
        if (it != merges.end() && it->second < best) {
          best = it->second;
          at = i;
        }
      }
      if (best == INT32_MAX)
        break;
      pieces[at] += pieces[at + 1];
      pieces.erase(pieces.begin() + at + 1);
    }
    for (const auto &p : pieces)
      out.push_back(vocab.at(p));
  }
  void ordinary(const std::string &input, std::vector<int32_t> &out) const {
    if (input.empty())
      return;
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, input.data(), input.size(), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
      throw std::invalid_argument("invalid UTF-8 prompt");
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(length + 1);
    u_strFromUTF8(utf16.data(), utf16.size(), &length, input.data(),
                  input.size(), &status);
    auto *normalizer = icu::Normalizer2::getNFCInstance(status);
    icu::UnicodeString normalized;
    if (U_SUCCESS(status))
      normalizer->normalize(icu::UnicodeString(utf16.data(), length),
                            normalized, status);
    if (U_FAILURE(status))
      throw std::invalid_argument("cannot normalize UTF-8 prompt");
    std::string s;
    normalized.toUTF8String(s);
    auto data =
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>(
            pcre2_match_data_create_from_pattern(regex, nullptr),
            pcre2_match_data_free);
    size_t pos = 0;
    while (pos < s.size()) {
      int rc = pcre2_match(regex, (PCRE2_SPTR)s.data(), s.size(), pos, 0,
                           data.get(), nullptr);
      if (rc < 0)
        throw std::invalid_argument(
            "prompt is not valid UTF-8 or tokenizer failed");
      auto match = pcre2_get_ovector_pointer(data.get());
      if (match[0] != pos || match[1] <= pos)
        throw std::runtime_error("tokenizer did not cover input");
      word(s.substr(pos, match[1] - pos), out);
      pos = match[1];
    }
  }
};
Tokenizer::Tokenizer(const std::string &path) : impl(std::make_unique<Impl>()) {
  std::ifstream f(path);
  json j;
  f >> j;
  if (j["normalizer"]["type"] != "NFC" || j["model"]["type"] != "BPE")
    throw std::runtime_error("unsupported tokenizer format");
  for (auto &[token, id] : j["model"]["vocab"].items())
    impl->vocab.emplace(impl->unbyte(token), id.get<int32_t>());
  int rank = 0;
  for (auto &merge : j["model"]["merges"]) {
    std::string a, b;
    if (merge.is_array()) {
      a = merge[0];
      b = merge[1];
    } else {
      std::string pair = merge;
      auto p = pair.find(' ');
      a = pair.substr(0, p);
      b = pair.substr(p + 1);
    }
    impl->merges.emplace(std::pair{impl->unbyte(a), impl->unbyte(b)}, rank++);
  }
  for (auto &token : j["added_tokens"])
    impl->special.emplace_back(token["content"], token["id"]);
  std::string pattern =
      j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"];
  int error;
  PCRE2_SIZE offset;
  impl->regex = pcre2_compile((PCRE2_SPTR)pattern.c_str(), pattern.size(),
                              PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
  if (!impl->regex)
    throw std::runtime_error("cannot compile tokenizer regex at " +
                             std::to_string(offset));
}
Tokenizer::~Tokenizer() = default;
std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
  if (text.size() > 65536)
    throw std::invalid_argument("prompt exceeds 65536 UTF-8 bytes");
  std::vector<int32_t> out;
  size_t start = 0;
  while (start < text.size()) {
    size_t next = std::string::npos;
    const std::pair<std::string, int32_t> *found = nullptr;
    for (const auto &sp : impl->special) {
      auto at = text.find(sp.first, start);
      if (at < next || (at == next && at != std::string::npos && found &&
                        sp.first.size() > found->first.size())) {
        next = at;
        found = &sp;
      }
    }
    if (next == std::string::npos) {
      impl->ordinary(text.substr(start), out);
      break;
    }
    impl->ordinary(text.substr(start, next - start), out);
    out.push_back(found->second);
    start = next + found->first.size();
  }
  return out;
}
std::vector<int32_t> Tokenizer::prompt(const std::string &text) const {
  const std::string prefix =
      "<|im_start|>system\nDescribe the image by detailing the color, shape, "
      "size, texture, quantity, text, spatial relationships of the objects and "
      "background:<|im_end|>\n<|im_start|>user\n";
  auto ids = encode(prefix + text);
  if (ids.size() > 541)
    ids.resize(541);
  auto suffix = encode("<|im_end|>\n<|im_start|>assistant\n");
  ids.insert(ids.end(), suffix.begin(), suffix.end());
  return ids;
}
} // namespace krea_native
