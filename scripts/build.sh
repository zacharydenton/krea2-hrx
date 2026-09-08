#!/usr/bin/env bash
# Build and stage the workspace binaries, libraries and C headers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v cargo > /dev/null; then
  printf 'cargo not found: install Rust\n' >&2
  exit 1
fi
./scripts/runtime.sh

cargo build --release --workspace
install -m 755 target/release/krea2 build/krea2
# One library, both C ABIs. A process that loads the block API and the pipeline
# API cannot end up with two HRX runtimes, because there is only one library to
# load; docs/hrx-runtime.md records why that matters.
install -m 644 target/release/libkrea2.so build/libkrea2.so
install -m 644 target/release/libnative_ops_test.so build/libnative_ops_test.so
for name in krea2-native-components krea2-scheduler-test krea2-schedule-grid \
    loomrun sage-runner gemm-bench attention-bench i4-bench; do
  install -m 755 "target/release/$name" "build/$name"
done
printf 'built build/krea2, build/libkrea2.so, build/include/krea2*.h and the test runners\n'
