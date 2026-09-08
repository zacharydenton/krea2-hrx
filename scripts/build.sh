#!/usr/bin/env bash
# Everything this repository builds. Rust and the Loom kernels; no C++.
#
# Two cargo passes, because the CLI links the runtime's C ABI the way any other
# consumer would: the library has to exist before the binary that links it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v cargo > /dev/null; then
  printf 'cargo not found: install Rust\n' >&2
  exit 1
fi
./scripts/runtime.sh

cargo build --release -p krea2-abi
# One cdylib, two names. docs/hrx-runtime.md records why: a process that loads
# both APIs must not end up with two HRX runtimes, and a symlink makes that
# impossible by construction -- the dynamic linker sees one soname.
install -m 644 target/release/libkrea2.so build/libkrea2_pipeline.so
ln -sfn libkrea2_pipeline.so build/libkrea2.so

cargo build --release --workspace
install -m 755 target/release/krea2 build/krea2
install -m 644 target/release/libnative_ops_test.so build/libnative_ops_test.so
for name in krea2-native-components krea2-scheduler-test krea2-schedule-grid \
    loomrun sage-runner gemm-bench attention-bench i4-bench; do
  install -m 755 "target/release/$name" "build/$name"
done
printf 'built build/krea2, build/libkrea2*.so, build/include/krea2*.h and the test runners\n'
