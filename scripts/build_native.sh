#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror -fPIC -shared host/native_ops.cpp host/native_models.cpp host/native_tokenizer.cpp host/native_compile.cpp host/native_pipeline.cpp host/krea2.cpp host/sage.cpp -lhipblas -lpcre2-8 -licuuc -lcrypto -o build/libkrea2_pipeline.so
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror host/native_cli.cpp -Lbuild -lkrea2_pipeline -Wl,-rpath,'$ORIGIN' -o build/krea2-generate
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror tests/native_components.cpp -Lbuild -lkrea2_pipeline -lhipblas -Wl,-rpath,'$ORIGIN' -o build/krea2-native-components
cc -O2 -Wall -Werror examples/generate.c -Lbuild -lkrea2_pipeline -Wl,-rpath,'$ORIGIN' -o build/krea2-c-example

/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror tests/sage_runner.cpp host/sage.cpp -lhipblas -o build/sage-runner
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror tests/scheduler_runner.cpp -Lbuild -lkrea2_pipeline -lhipblas -Wl,-rpath,'$ORIGIN' -o build/krea2-scheduler-test
