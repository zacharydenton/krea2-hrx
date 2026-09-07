#pragma once
#include <cmath>
namespace krea_native {
// diffusers' exponential time shift of linspace(1, 1/steps) sigmas. Turbo (the
// distilled checkpoint) uses the fixed mu = 1.15; Raw computes mu from the image
// token count (dynamic_mu). Callers validate 1..100 inference steps.
inline double dynamic_mu(int image_tokens) {
  // Flux's calculate_shift with Krea 2's scheduler config: base 256 tokens at
  // 0.5, 6400 tokens at 1.15, linear between (1024^2 = 4096 tokens: 0.906).
  const double m = (1.15 - 0.5) / (6400 - 256), b = 0.5 - m * 256;
  return image_tokens * m + b;
}
inline float scheduler_sigma(int step, int steps, double mu = 1.15) {
  if (step == steps) return 0;
  // Diffusers casts linspace to float32 before NumPy's shift operations.
  // Preserve these boundaries: a tiny sigma difference can cross a bf16 dt tie.
  float raw = float(1. - double(step) / steps);
  // Keep mu in double through exp, as Python does; rounding it first can
  // move a later bf16 Euler delta across a rounding boundary.
  const float shift = float(std::exp(mu));
  float inverse = 1.f / raw;
  float offset = inverse - 1.f;
  return shift / (shift + offset);
}
} // namespace krea_native
