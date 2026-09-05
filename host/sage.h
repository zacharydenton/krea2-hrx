#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <memory>
#include <vector>

// Native preprocessing for the Loom SA2-style kernel. All work uses stream 0.
// The Q/K nibble layout is [capacity][heads][64 bytes]. Scales are per token.
// Correction is [query heads][ceil(tokens/64)][capacity] in float32.
class SagePreparation {
public:
  SagePreparation(int tokens, int capacity, int heads = 48, int kv_heads = 12);
  ~SagePreparation();
  SagePreparation(const SagePreparation &) = delete;
  SagePreparation &operator=(const SagePreparation &) = delete;
  void run(const void *q, const void *k, const void *v, bool profile = false);
  void *q4 = nullptr, *k4 = nullptr, *qscale = nullptr, *kscale = nullptr,
       *correction = nullptr, *v_transposed = nullptr;

private:
  int tokens_, capacity_, heads_, kv_heads_, tiles_;
  hipblasHandle_t handle_ = nullptr;
  std::vector<void *> allocations_;
  void *kpartial_ = nullptr, *kmean_ = nullptr, *qmean_ = nullptr,
       *qmean_half_ = nullptr, *centered_k_ = nullptr;
  void allocate(void *&p, size_t bytes);
  void release() noexcept;
};
