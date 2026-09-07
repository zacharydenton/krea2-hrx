#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/build_common.sh
link_shared build/libkrea2.so "${COMMON[@]}"
link_pipeline build/libkrea2_pipeline.so
for pair in 'host/native_cli.cpp krea2' \
    'tests/native_components.cpp krea2-native-components' \
    'tests/scheduler_runner.cpp krea2-scheduler-test' \
    'tests/sage_runner.cpp sage-runner'; do
  read -r source output <<< "$pair"
  "$CXX" "${CXXFLAGS[@]}" "$source" -Lbuild -lkrea2_pipeline -lkrea2 -Wl,-rpath,'$ORIGIN' -o "build/$output.tmp"
  mv -f "build/$output.tmp" "build/$output"
done
cc -O2 -Wall -Werror examples/generate.c -Lbuild -lkrea2_pipeline -Wl,-rpath,'$ORIGIN' -o build/krea2-c-example.tmp
mv -f build/krea2-c-example.tmp build/krea2-c-example
"$CXX" "${CXXFLAGS[@]}" -shared tests/tokenizer_runner.cpp host/native_tokenizer.cpp -o build/libtokenizer_test.so.tmp
mv -f build/libtokenizer_test.so.tmp build/libtokenizer_test.so
printf 'built HRX native pipeline and C example\n'
