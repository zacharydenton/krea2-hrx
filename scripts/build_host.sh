#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror -fPIC -shared -o build/libkrea2.so host/krea2.cpp host/sage.cpp -lhipblas
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror -o host/loomrun host/loomrun.cpp
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror tests/sage_runner.cpp host/sage.cpp -lhipblas -o build/sage-runner
printf 'built build/libkrea2.so, host/loomrun and build/sage-runner\n'
