# Source this. Paths to the Loom toolchain and the ROCm runtime that works on Arch.
HRX_BUILD="${HRX_BUILD:-$HOME/code/hrx-system/build-cuda}"
export LOOM_TOOLS="$HRX_BUILD/loom/src/loom/tools"
export LOOM_COMPILE="$LOOM_TOOLS/loom-compile/loom-compile"
export LOOM_FORMAT="$LOOM_TOOLS/loom-format/loom-format"
export LOOM_CHECK="$LOOM_TOOLS/loom-check/loom-check"
export IREE_TEST_LOOM="$LOOM_TOOLS/iree-test-loom/iree-test-loom"
export IREE_BENCHMARK_LOOM="$LOOM_TOOLS/iree-benchmark-loom/iree-benchmark-loom"
# Tests that combine Torch and HRX must load the same HSA provider at startup.
# Standalone C callers locate the packaged provider without an environment change.
KREA2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$KREA2_ROOT/build/runtime:$HOME/.local/rocm-hrx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LOOM_TARGET="${LOOM_TARGET:-gfx1151}"
