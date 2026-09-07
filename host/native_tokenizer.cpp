#include "native_tokenizer.h"
#include "unicode.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace krea_native {
using json = nlohmann::json;
struct Tokenizer::Impl {
  std::map<unsigned, unsigned> reverse;
  std::unordered_map<std::string, int32_t> vocab;
  std::map<std::pair<std::string, std::string>, int> merges;
  std::vector<std::pair<std::string, int32_t>> special;
  Impl() {
    unsigned extra = 256;
    for (unsigned i = 0; i < 256; ++i)
      if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174)
        reverse[i] = i;
      else
        reverse[extra++] = i;
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
    for (const auto &piece : unicode::words(input))
      word(piece, out);
  }
};
Tokenizer::Tokenizer(const std::string &path) : impl(std::make_unique<Impl>()) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot read tokenizer: " + path);
  load(std::string(std::istreambuf_iterator<char>(f), {}));
}
Tokenizer::Tokenizer(const char *text, size_t size)
    : impl(std::make_unique<Impl>()) {
  load(std::string(text, size));
}
void Tokenizer::load(const std::string &text) {
  json j = json::parse(text);
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
  const std::string supported =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  if (pattern != supported)
    throw std::runtime_error("unsupported tokenizer split pattern");
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
