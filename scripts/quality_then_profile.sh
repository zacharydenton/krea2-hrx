#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
rm -f build/profile.done
scripts/quality_rest.sh
source .venv/bin/activate
env -u LD_LIBRARY_PATH PYTHONUNBUFFERED=1 python3 tests/test_blocks.py --curve 28 --profile > build/blocks28.log 2>&1
echo done > build/profile.done
