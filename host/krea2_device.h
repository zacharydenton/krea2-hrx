#pragma once
#include "krea2.h"
#include <memory>
#include <string>

// Immutable device weights shared by shape-specific sessions in one pipeline.
struct krea2_weights;
std::shared_ptr<krea2_weights> krea2_load_weights(const std::string &directory);
// The GEMM operand width the weights were exported for: 4 (W4A4) or 8 (W8A8).
int krea2_weights_bits(const krea2_weights &weights);
krea2_session *krea2_create_shared(const std::shared_ptr<krea2_weights> &weights,
                                    const std::string &kernels, int tokens, int layers);

// Internal bridge for the native pipeline. x is a HRX device allocation of
// bf16 [tokens][6144], converted on-device at the block stack's fp16 boundary.
// Modulation is float32 device memory; rotary arrays are host pointers. Uses the ordered HRX stream,
// serializes through the block session, completes before returning, and throws
// on error. The public host-memory C ABI remains unchanged.
void krea2_run_device_bf16(krea2_session *session, uint16_t *x, size_t elements,
                           const float *mods, size_t mods_elements,
                           const float *cos, const float *sin,
                           size_t rope_elements);
