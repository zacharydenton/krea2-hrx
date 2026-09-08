#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/build_common.sh
link_shared build/libkrea2.so "${COMMON[@]}"
link_pipeline build/libkrea2_pipeline.so
for pair in 'tests/native_components.cpp krea2-native-components' \
    'tests/scheduler_runner.cpp krea2-scheduler-test' \
    'tests/sage_runner.cpp sage-runner'; do
  read -r source output <<< "$pair"
  "$CXX" "${CXXFLAGS[@]}" "$source" -Lbuild -lkrea2_pipeline -lkrea2 -Wl,-rpath,'$ORIGIN' -o "build/$output.tmp"
  mv -f "build/$output.tmp" "build/$output"
done
cc -O2 -Wall -Werror examples/generate.c -Lbuild -lkrea2_pipeline -Wl,-rpath,'$ORIGIN' -o build/krea2-c-example.tmp
mv -f build/krea2-c-example.tmp build/krea2-c-example
printf 'built HRX native pipeline and C example\n'
# The Rust artifacts. Skipped without cargo, which the C++ libraries above do
# not need.
if command -v cargo > /dev/null; then
  ./scripts/build_rust.sh
else
  printf 'cargo not found: skipping build/krea2\n' >&2
fi
