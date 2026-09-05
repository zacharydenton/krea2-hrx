#pragma once
#include "unicode_data.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
namespace krea_native::unicode {
inline std::vector<uint32_t> decode(const std::string &s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) {
    uint32_t a = uint8_t(s[i++]), cp = a;
    unsigned extra = 0;
    if (a >= 0xc2 && a <= 0xdf) {
      extra = 1;
      cp = a & 31;
    } else if (a >= 0xe0 && a <= 0xef) {
      extra = 2;
      cp = a & 15;
    } else if (a >= 0xf0 && a <= 0xf4) {
      extra = 3;
      cp = a & 7;
    } else if (a >= 128)
      throw std::invalid_argument("invalid UTF-8 prompt");
    if (i + extra > s.size())
      throw std::invalid_argument("truncated UTF-8 prompt");
    for (unsigned j = 0; j < extra; ++j) {
      uint32_t b = uint8_t(s[i++]);
      if ((b & 0xc0) != 0x80)
        throw std::invalid_argument("invalid UTF-8 continuation");
      cp = (cp << 6) | (b & 63);
    }
    if ((extra == 1 && cp < 128) || (extra == 2 && cp < 2048) ||
        (extra == 3 && cp < 65536) || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      throw std::invalid_argument("invalid UTF-8 codepoint");
    out.push_back(cp);
  }
  return out;
}
inline void append(std::string &s, uint32_t c) {
  if (c < 128)
    s += char(c);
  else if (c < 2048) {
    s += char(0xc0 | (c >> 6));
    s += char(0x80 | (c & 63));
  } else if (c < 65536) {
    s += char(0xe0 | (c >> 12));
    s += char(0x80 | ((c >> 6) & 63));
    s += char(0x80 | (c & 63));
  } else {
    s += char(0xf0 | (c >> 18));
    s += char(0x80 | ((c >> 12) & 63));
    s += char(0x80 | ((c >> 6) & 63));
    s += char(0x80 | (c & 63));
  }
}
template <size_t N>
bool in(uint32_t c, const unicode_data::Range (&ranges)[N]) {
  auto i =
      std::upper_bound(std::begin(ranges), std::end(ranges), c,
                       [](uint32_t c, const auto &r) { return c < r.first; });
  return i != std::begin(ranges) && c <= (i - 1)->last;
}
inline bool letter(uint32_t c) { return in(c, unicode_data::letters); }
inline bool number(uint32_t c) { return in(c, unicode_data::numbers); }
inline bool space(uint32_t c) {
  return c == 32 || (c >= 9 && c <= 13) || c == 0x85 || c == 0xa0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
         c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
inline unsigned ccc(uint32_t c) {
  const auto &a = unicode_data::combining;
  auto p =
      std::lower_bound(std::begin(a), std::end(a), c,
                       [](const auto &r, uint32_t c) { return r.code < c; });
  return p != std::end(a) && p->code == c ? p->value : 0;
}
inline void decompose(uint32_t c, std::vector<uint32_t> &out) {
  if (c >= 0xac00 && c < 0xac00 + 11172) {
    unsigned s = c - 0xac00;
    out.push_back(0x1100 + s / 588);
    out.push_back(0x1161 + (s % 588) / 28);
    if (s % 28)
      out.push_back(0x11a7 + s % 28);
    return;
  }
  const auto &a = unicode_data::decompositions;
  auto p =
      std::lower_bound(std::begin(a), std::end(a), c,
                       [](const auto &r, uint32_t c) { return r.code < c; });
  if (p != std::end(a) && p->code == c) {
    decompose(p->a, out);
    if (p->b)
      decompose(p->b, out);
  } else
    out.push_back(c);
}
inline uint32_t compose(uint32_t a, uint32_t b) {
  if (a >= 0x1100 && a < 0x1100 + 19 && b >= 0x1161 && b < 0x1161 + 21)
    return 0xac00 + ((a - 0x1100) * 21 + b - 0x1161) * 28;
  if (a >= 0xac00 && a < 0xac00 + 11172 && (a - 0xac00) % 28 == 0 &&
      b > 0x11a7 && b < 0x11a7 + 28)
    return a + b - 0x11a7;
  uint64_t key = (uint64_t(a) << 21) | b;
  const auto &cs = unicode_data::compositions;
  auto p =
      std::lower_bound(std::begin(cs), std::end(cs), key,
                       [](const auto &r, uint64_t k) { return r.pair < k; });
  return p != std::end(cs) && p->pair == key ? p->code : 0;
}
inline std::vector<uint32_t> nfc(const std::string &s) {
  std::vector<uint32_t> d;
  for (uint32_t c : decode(s))
    decompose(c, d);
  for (size_t i = 1; i < d.size(); ++i) {
    unsigned cc = ccc(d[i]);
    if (!cc)
      continue;
    size_t j = i;
    while (j && ccc(d[j - 1]) > cc) {
      std::swap(d[j - 1], d[j]);
      --j;
    }
  }
  std::vector<uint32_t> out;
  size_t starter = 0;
  unsigned last = 0;
  for (uint32_t c : d) {
    unsigned cc = ccc(c);
    uint32_t combined = out.empty() ? 0 : compose(out[starter], c);
    if (combined && (last < cc || last == 0))
      out[starter] = combined;
    else {
      if (cc == 0)
        starter = out.size();
      out.push_back(c);
      last = cc;
    }
  }
  return out;
}
// Qwen's ordered split alternatives, evaluated on complete Unicode categories.
inline std::vector<std::string> words(const std::string &s) {
  auto c = nfc(s);
  std::vector<std::string> out;
  size_t p = 0, n = c.size();
  auto lower = [](uint32_t x) {
    return x >= 65 && x <= 90 ? x + 32 : x == 0x17f ? uint32_t('s') : x;
  };
  auto nl = [](uint32_t x) { return x == 10 || x == 13; };
  while (p < n) {
    size_t q = p;
    uint32_t x = c[p];
    if (x == '\'' && p + 1 < n) {
      auto a = lower(c[p + 1]), b = p + 2 < n ? lower(c[p + 2]) : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd')
        q = p + 2;
      else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
               (a == 'l' && b == 'l'))
        q = p + 3;
    }
    if (q == p) {
      if (letter(x) ||
          (!nl(x) && !number(x) && p + 1 < n && letter(c[p + 1]))) {
        q = letter(x) ? p : p + 1;
        while (q < n && letter(c[q]))
          ++q;
      } else if (number(x))
        q = p + 1;
      else if (!space(x) || (x == 32 && p + 1 < n && !space(c[p + 1]) &&
                             !letter(c[p + 1]) && !number(c[p + 1]))) {
        q = x == 32 ? p + 1 : p;
        while (q < n && !space(c[q]) && !letter(c[q]) && !number(c[q]))
          ++q;
        while (q < n && nl(c[q]))
          ++q;
      } else {
        size_t r = p;
        while (r < n && space(c[r]))
          ++r;
        size_t newline = p;
        bool found = false;
        for (size_t i = p; i < r; ++i)
          if (nl(c[i])) {
            newline = i;
            found = true;
          }
        q = found ? newline + 1 : (r < n && r - p > 1 ? r - 1 : r);
      }
    }
    if (q <= p)
      throw std::runtime_error("tokenizer made no progress");
    std::string word;
    for (size_t i = p; i < q; ++i)
      append(word, c[i]);
    out.push_back(std::move(word));
    p = q;
  }
  return out;
}
} // namespace krea_native::unicode
