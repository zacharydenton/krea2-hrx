#!/usr/bin/env bash
# Wait for the bf16 baseline to exit, then export the int4 weights and run the W4A4
# image with the same seed. A script file, so no shell's own command line matches the
# pattern it waits on.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
rm -f build/export.done
while pgrep -f "pipeline.py --seed 0 --out build/bf16" >/dev/null; do sleep 10; done
source .venv/bin/activate
export PYTHONUNBUFFERED=1
env -u LD_LIBRARY_PATH python3 tools/export_weights.py > build/export.log 2>&1
env -u LD_LIBRARY_PATH python3 tools/pipeline.py --quant w4a4 --seed 0 --out build/w4a4_seed0.png > build/w4a4.log 2>&1
