// Standalone Krea 2 Turbo inference. No Python or Torch runtime is required.
#ifndef KREA2_PIPELINE_H
#define KREA2_PIPELINE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define KREA2_PIPELINE_ABI_VERSION 1u
typedef struct krea2_pipeline krea2_pipeline;
uint32_t krea2_pipeline_abi_version(void);
// bundle is produced by tools/export_native.py. compiler names a native
// loom-compile executable, or NULL to use LOOM_COMPILE/PATH. Compilation only
// occurs when preparing an unseen sequence length; bundles may be precompiled.
int krea2_pipeline_create(const char *bundle, const char *compiler,
                          krea2_pipeline **out, char *error,
                          size_t error_capacity);
void krea2_pipeline_destroy(krea2_pipeline *pipeline);
// Tuned gfx1151 INT4-QK/fp16-PV attention is selected internally by sequence
// length. All calls on a session are serialized. Destruction must not race a
// call. Dimensions must be multiples of 16, between 64 and 2048. This is the
// distilled Turbo model (no classifier-free guidance). Output is contiguous
// RGB8 HWC. Steps must be in 1..100. Optional initial_latents are float32
// packed [H/16 * W/16][64]. Otherwise seed initializes the native RNG, which
// intentionally does not emulate Torch's RNG.
int krea2_generate(krea2_pipeline *pipeline, const char *prompt, int width,
                   int height, int steps, uint64_t seed,
                   const float *initial_latents, size_t latent_elements,
                   uint8_t *rgb, size_t rgb_bytes, char *error,
                   size_t error_capacity);
// Component entry points for foreign-language integrations and validation.
// Tokenize arbitrary UTF-8 text with the bundled tokenizer (no chat template).
int krea2_tokenize(krea2_pipeline *pipeline, const char *text, int32_t *ids,
                   size_t capacity, size_t *written, char *error,
                   size_t error_capacity);
// Encode a prompt with Krea's template. Output is float32 [tokens][12][2560].
// With output=NULL, reports tokens without running the encoder. Maximum 512.
int krea2_encode(krea2_pipeline *pipeline, const char *prompt, float *output,
                 size_t elements, size_t *tokens, char *error,
                 size_t error_capacity);
// Complete transformer: packed latents + tapped text states + normalized time
// -> float32 packed velocity. Text tokens have already had padding removed.
int krea2_transformer(krea2_pipeline *pipeline, const float *text,
                      size_t text_elements, int text_tokens,
                      const float *latents, size_t latent_elements, int width,
                      int height, float timestep, float *velocity,
                      size_t velocity_elements, char *error,
                      size_t error_capacity);
int krea2_decode(krea2_pipeline *pipeline, const float *latents,
                 size_t elements, int width, int height, uint8_t *rgb,
                 size_t rgb_bytes, char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
