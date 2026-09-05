#!/usr/bin/env bash
# After the export: bf16 and W4A4 latents from the same seed, then tiled decode + PSNR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
rm -f build/quality.done
source .venv/bin/activate
export PYTHONUNBUFFERED=1
env -u LD_LIBRARY_PATH python3 tools/pipeline.py --seed 0 --latents-out build/bf16_seed0.pt --out build/unused.png > build/bf16.log 2>&1
env -u LD_LIBRARY_PATH python3 tools/pipeline.py --quant w4a4 --seed 0 --latents-out build/w4a4_seed0.pt --out build/unused.png > build/w4a4.log 2>&1
env -u LD_LIBRARY_PATH python3 tools/decode_latents.py build/bf16_seed0.pt build/w4a4_seed0.pt > build/decode.log 2>&1
echo done > build/quality.done
