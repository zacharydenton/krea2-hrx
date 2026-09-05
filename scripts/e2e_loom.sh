#!/usr/bin/env bash
# The whole pipeline with the blocks in Loom, same seed as the bf16 baseline, then the
# tiled decode and PSNR against bf16 and against the W4A4 reference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
rm -f build/e2e.done
source .venv/bin/activate
export PYTHONUNBUFFERED=1
env -u LD_LIBRARY_PATH python3 tools/pipeline.py --backend loom --seed 0 --latents-out build/loom_seed0.pt --out build/unused.png > build/loom.log 2>&1
env -u LD_LIBRARY_PATH python3 tools/decode_latents.py build/bf16_seed0.pt build/loom_seed0.pt > build/decode_loom.log 2>&1
env -u LD_LIBRARY_PATH python3 tools/decode_latents.py build/w4a4_seed0.pt build/loom_seed0.pt > build/decode_loom_vs_w4a4.log 2>&1
echo done > build/e2e.done
