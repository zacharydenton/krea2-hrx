#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/build_common.sh
link_shared build/libkrea2.so "${COMMON[@]}"
"$CXX" "${CXXFLAGS[@]}" host/loomrun.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN/../build' -o host/loomrun.tmp
mv -f host/loomrun.tmp host/loomrun
"$CXX" "${CXXFLAGS[@]}" tests/sage_runner.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o build/sage-runner.tmp
mv -f build/sage-runner.tmp build/sage-runner
"$CXX" "${CXXFLAGS[@]}" -shared tests/native_ops_runner.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o build/libnative_ops_test.so.tmp
mv -f build/libnative_ops_test.so.tmp build/libnative_ops_test.so
"$CXX" "${CXXFLAGS[@]}" tests/test_hrx_runtime.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o build/test-hrx-runtime.tmp
mv -f build/test-hrx-runtime.tmp build/test-hrx-runtime
"$CXX" "${CXXFLAGS[@]}" tests/gemm_bench.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o build/gemm-bench.tmp
mv -f build/gemm-bench.tmp build/gemm-bench
"$CXX" "${CXXFLAGS[@]}" tests/i4_bench.cpp -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o build/i4-bench.tmp
mv -f build/i4-bench.tmp build/i4-bench
printf 'built HRX block library, Loom runner and Sage runner\n'
