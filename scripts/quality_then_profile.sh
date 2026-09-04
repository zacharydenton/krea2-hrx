#!/usr/bin/env bash
cd ~/code/krea2-loom
scripts/quality_rest.sh
source .venv/bin/activate
env -u LD_LIBRARY_PATH PYTHONUNBUFFERED=1 python3 tests/test_blocks.py --curve 28 --profile > build_blocks28.log 2>&1
echo done > build_profile.done
