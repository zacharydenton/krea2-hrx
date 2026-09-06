#pragma once
#include "gpu.h"
#include <cstdint>
#include <memory>
#include <vector>

// Loom preprocessing for the Loom SA2-style kernel. All work uses the ordered
// HRX stream. The Q/K nibble layout is head-major, [heads][capacity][64 bytes],
// with scales [heads][capacity] per token, so a key tile is one contiguous
// block. Correction is [query heads][ceil(tokens/64)][capacity] in float32.
class SagePreparation {
public:
  // bits: 4 (codes -7..7, 64 B per head row) or 8 (-127..127, 128 B per head
  // row); the attention kernel of the same width consumes the output.
  SagePreparation(int tokens, int capacity, int heads = 48, int kv_heads = 12,
                  int bits = 4);
  ~SagePreparation();
  SagePreparation(const SagePreparation &) = delete;
  SagePreparation &operator=(const SagePreparation &) = delete;
  void run(const void *q, const void *k, const void *v, bool profile = false);
  void *q4 = nullptr, *k4 = nullptr, *qscale = nullptr, *kscale = nullptr,
       *correction = nullptr, *v_transposed = nullptr;

private:
  int tokens_, capacity_, heads_, kv_heads_, tiles_, bits_;
  std::vector<void *> allocations_;
  void *kpartial_ = nullptr, *kmean_ = nullptr, *qmean_ = nullptr,
       *qmean_half_ = nullptr, *centered_k_ = nullptr;
  void allocate(void *&p, size_t bytes);
  void release() noexcept;
};
