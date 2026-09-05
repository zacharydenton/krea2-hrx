// Observe actual HIP allocation ownership across failed native constructors.
#include <hip/hip_runtime.h>
#include "../host/krea2.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

static int outstanding = 0, allocations = 0;
extern "C" hipError_t __real_hipMalloc(void **, size_t);
extern "C" hipError_t __real_hipFree(void *);
extern "C" hipError_t __wrap_hipMalloc(void **p, size_t bytes) {
    auto status = __real_hipMalloc(p, bytes);
    if (status == hipSuccess) { ++outstanding; ++allocations; }
    return status;
}
extern "C" hipError_t __wrap_hipFree(void *p) {
    auto status = __real_hipFree(p);
    if (status == hipSuccess && p) --outstanding;
    return status;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const auto dir = std::filesystem::path(argv[1]) / "session-fixture";
    std::filesystem::create_directory(dir);
    std::ofstream(dir / "launch.txt") << "2 16 2 64 8\n";
    std::ofstream(dir / "manifest.txt") << "";
    std::ofstream(dir / "weights.bin") << "incomplete weights";
    for (int i = 0; i < 3; ++i) {
        krea2_session *session = nullptr;
        char error[4096];
        assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error, sizeof(error)) == KREA2_ERROR);
        assert(!session);
        assert(std::string(error).find("missing tensor") != std::string::npos);
        assert(outstanding == 0);
    }
    assert(allocations == 3);
    // Invalid metadata must be rejected before allocating weights.
    std::ofstream(dir / "launch.txt") << "2 16 0 64 8\n";
    krea2_session *session = nullptr;
    char error[4096];
    assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error, sizeof(error)) == KREA2_INVALID_ARGUMENT);
    assert(allocations == 3 && outstanding == 0);
    std::ofstream(dir / "launch.txt") << "1 16 2 64\n";
    assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error, sizeof(error)) == KREA2_INVALID_ARGUMENT);
    assert(allocations == 3 && outstanding == 0);
    std::ofstream(dir / "launch.txt") << "2 16 2 64 4\n";
    assert(krea2_create(dir.c_str(), dir.c_str(), 16, 1, &session, error, sizeof(error)) == KREA2_INVALID_ARGUMENT);
    assert(allocations == 3 && outstanding == 0);
    std::cout << "PASS native constructor cleanup and launch metadata\n";
}
