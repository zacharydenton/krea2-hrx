// Standalone Krea 2 inference (the Turbo and Raw checkpoints). No Python or
// Torch runtime is required.
#ifndef KREA2_PIPELINE_H
#define KREA2_PIPELINE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define KREA2_PIPELINE_ABI_VERSION 2u
typedef struct krea2_pipeline krea2_pipeline;
uint32_t krea2_pipeline_abi_version(void);
// bundle is ComfyUI's diffusion model checkpoint (a .safetensors file such as
// diffusion_models/krea2_turbo_int8_convrot.safetensors, with the text encoder
// and VAE found beside it in ComfyUI's layout) or a directory produced by
// tools/export_native.py. compiler names a native loom-compile executable, or
// NULL to use LOOM_COMPILE/PATH. Compilation only occurs when preparing an
// unseen sequence length; compiled kernels are cached (in the bundle, or under
// $XDG_CACHE_HOME/krea2-loom for a checkpoint).
int krea2_pipeline_create(const char *bundle, const char *compiler,
                          krea2_pipeline **out, char *error,
                          size_t error_capacity);
// ComfyUI's files named explicitly: the int8 ConvRot diffusion model, the
// Qwen3-VL-4B text encoder (bf16 or fp8_scaled) and the Qwen-Image VAE.
// text_encoder or vae NULL: found beside the model as above. distilled: 1 for
// Turbo, 0 for Raw, -1 to infer from the file name ("raw").
int krea2_pipeline_create_files(const char *model, const char *text_encoder,
                                const char *vae, int distilled,
                                const char *compiler, krea2_pipeline **out,
                                char *error, size_t error_capacity);
void krea2_pipeline_destroy(krea2_pipeline *pipeline);
// Tuned gfx1151 INT4-QK/fp16-PV attention is selected internally by sequence
// length. All calls on a session are serialized. Destruction must not race a
// call. Dimensions must be multiples of 16, between 64 and 2048. Output is
// contiguous RGB8 HWC. Steps must be in 1..100. Optional initial_latents are
// float32 packed [H/16 * W/16][64]. Otherwise seed initializes the native RNG,
// which intentionally does not emulate Torch's RNG. No classifier-free
// guidance: the Turbo form (a Raw bundle runs unguided, which is not how Raw
// is meant to be sampled; use krea2_generate_guided).
int krea2_generate(krea2_pipeline *pipeline, const char *prompt, int width,
                   int height, int steps, uint64_t seed,
                   const float *initial_latents, size_t latent_elements,
                   uint8_t *rgb, size_t rgb_bytes, char *error,
                   size_t error_capacity);
// The same with classifier-free guidance in Krea's convention, velocity =
// cond + guidance * (cond - uncond), two transformer forwards per step; the
// bundle's checkpoint (native.json "model": krea2-turbo or krea2-raw) selects
// the timestep shift. negative_prompt may be NULL or empty (the unconditional
// branch then encodes the empty prompt, as diffusers does). guidance < 0 or
// steps <= 0 take the checkpoint's defaults: Turbo 0 and 8 (no guidance),
// Raw 3.5 and 52.
int krea2_generate_guided(krea2_pipeline *pipeline, const char *prompt,
                          const char *negative_prompt, float guidance,
                          int width, int height, int steps, uint64_t seed,
                          const float *initial_latents,
                          size_t latent_elements, uint8_t *rgb,
                          size_t rgb_bytes, char *error,
                          size_t error_capacity);
// The bundle's checkpoint: 1 for the distilled Turbo model, 0 for Raw.
int krea2_pipeline_distilled(const krea2_pipeline *pipeline);
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
