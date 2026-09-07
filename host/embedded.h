#pragma once
#include <cstddef>
namespace krea_native {
// Qwen3-VL's tokenizer.json (assets/tokenizer.json), embedded so that a
// ComfyUI models directory, which carries no tokenizer file, is enough to run.
const char *embedded_tokenizer_json();
size_t embedded_tokenizer_size();
} // namespace krea_native
