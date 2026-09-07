#include "embedded.h"
// .incbin keeps the 7 MB asset out of the compiler's front end; the path is
// relative to the repository root, where scripts/build_*.sh run.
__asm__(".section .rodata\n"
        ".balign 16\n"
        ".global krea2_tokenizer_json\n"
        "krea2_tokenizer_json:\n"
        ".incbin \"assets/tokenizer.json\"\n"
        ".byte 0\n"
        ".global krea2_tokenizer_json_end\n"
        "krea2_tokenizer_json_end:\n"
        ".previous\n");
extern "C" const char krea2_tokenizer_json[];
extern "C" const char krea2_tokenizer_json_end[];
namespace krea_native {
const char *embedded_tokenizer_json() { return krea2_tokenizer_json; }
size_t embedded_tokenizer_size() {
  return size_t(krea2_tokenizer_json_end - krea2_tokenizer_json) - 1;
}
} // namespace krea_native
