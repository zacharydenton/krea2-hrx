#pragma once
#include <cmath>
namespace krea_native {
// Turbo's fixed exponential shift; callers validate 1..100 inference steps.
inline float scheduler_sigma(int step, int steps) {
  if (step == steps) return 0;
  // Diffusers casts linspace to float32 before NumPy's shift operations.
  // Preserve these boundaries: a tiny sigma difference can cross a bf16 dt tie.
  float raw = float(1. - double(step) / steps);
  const float shift = float(std::exp(1.15));
  float inverse = 1.f / raw;
  float offset = inverse - 1.f;
  return shift / (shift + offset);
}
} // namespace krea_native
