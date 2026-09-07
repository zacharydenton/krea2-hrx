// C ABI for the resident Krea 2 transformer-block session (the 28 DiT blocks in Loom).
#ifndef KREA2_LOOM_H
#define KREA2_LOOM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define KREA2_ABI_VERSION 2u
enum { KREA2_OK = 0, KREA2_ERROR = 1, KREA2_INVALID_ARGUMENT = 64 };
typedef struct krea2_session krea2_session;
uint32_t krea2_abi_version(void);
// kernels_dir holds the HSACOs and launch.txt produced by scripts/build_kernels.py
// for exactly `tokens` (16..16896), targeting gfx1151; weights_dir is ComfyUI's int8 ConvRot
// checkpoint (a .safetensors file, read as it is).
int krea2_create(const char *weights_dir, const char *kernels_dir, int tokens, int layers,
                 krea2_session **out_session, char *error, size_t error_capacity);
// x: f16 [tokens][6144] in and out (the residual stream after all blocks);
// mods: f32 [layers][6][6144] (prescale, preshift, pregate, postscale, postshift, postgate,
// already including each block's table); cos/sin: f32 [tokens][128].
int krea2_run(krea2_session *s, uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements,
              const float *cos, const float *sin, size_t rope_elements, char *error, size_t error_capacity);
// Run a contiguous subset of the loaded blocks, retaining the same full-session
// mods layout. block_count=-1 means all remaining blocks. Useful for checking
// individual blocks on identical input streams.
int krea2_run_range(krea2_session *s, int first_block, int block_count,
                    uint16_t *x, size_t x_elements, const float *mods, size_t mods_elements,
                    const float *cos, const float *sin, size_t rope_elements, char *error, size_t error_capacity);
int krea2_profile(krea2_session *s, int enable);
void krea2_destroy(krea2_session *s);
#ifdef __cplusplus
}
#endif
#endif
