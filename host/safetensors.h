#pragma once
// A read-only view of a safetensors file: the JSON header (8-byte little
// endian length, then UTF-8 JSON of name -> {dtype, shape, data_offsets}) and
// the tensor bytes behind it, memory-mapped. No copies until a caller asks.
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace krea_native {

struct SafeTensors {
  struct Entry {
    std::string dtype; // "BF16", "F32", "I8", "U8", "F8_E4M3", "F16", ...
    std::vector<int64_t> shape;
    size_t offset = 0, bytes = 0; // into data()
    size_t elements() const {
      size_t n = 1;
      for (auto d : shape)
        n *= size_t(d);
      return n;
    }
  };
  std::string path;
  std::map<std::string, Entry> entries;

  explicit SafeTensors(const std::string &file) : path(file) {
    try {
      fd_ = open(file.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd_ < 0)
        throw std::runtime_error("cannot open " + file);
      struct stat info;
      if (fstat(fd_, &info) || info.st_size < 8) {
        throw std::runtime_error("invalid safetensors file: " + file);
      }
      size_ = size_t(info.st_size);
      mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (mapped_ == MAP_FAILED) {
        throw std::runtime_error("cannot map " + file);
      }
      uint64_t header = 0;
      std::memcpy(&header, mapped_, 8);
      if (header > size_ - 8)
        throw std::runtime_error("corrupt safetensors header: " + file);
      data_ = static_cast<const char *>(mapped_) + 8 + header;
      const size_t data_bytes = size_ - 8 - header;
      auto j = nlohmann::json::parse(static_cast<const char *>(mapped_) + 8,
                                     static_cast<const char *>(mapped_) + 8 +
                                         header);
      for (auto &[name, value] : j.items()) {
        if (name == "__metadata__")
          continue;
        Entry e;
        e.dtype = value.at("dtype").get<std::string>();
        e.shape = value.at("shape").get<std::vector<int64_t>>();
        auto offsets = value.at("data_offsets").get<std::vector<size_t>>();
        if (offsets.size() != 2 || offsets[1] < offsets[0] ||
            offsets[1] > data_bytes)
          throw std::runtime_error("corrupt tensor span: " + name);
        e.offset = offsets[0];
        e.bytes = offsets[1] - offsets[0];
        entries.emplace(name, std::move(e));
      }
    } catch (...) {
      release();
      throw;
    }
  }
  ~SafeTensors() { release(); }
  SafeTensors(const SafeTensors &) = delete;
  SafeTensors &operator=(const SafeTensors &) = delete;

  bool has(const std::string &name) const { return entries.count(name); }
  const Entry &at(const std::string &name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      throw std::runtime_error("missing tensor " + name + " in " + path);
    return it->second;
  }
  const char *data(const Entry &e) const { return data_ + e.offset; }
  const char *data(const std::string &name) const { return data(at(name)); }

private:
  void release() noexcept {
    if (mapped_ != MAP_FAILED && mapped_)
      munmap(mapped_, size_);
    if (fd_ >= 0)
      close(fd_);
  }
  int fd_ = -1;
  void *mapped_ = nullptr;
  size_t size_ = 0;
  const char *data_ = nullptr;
};

inline float bf16_to_float(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

} // namespace krea_native
