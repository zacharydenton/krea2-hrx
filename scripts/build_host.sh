#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
/opt/rocm/bin/hipcc -O2 -Wall -Werror -fPIC -shared -o build/libkrea2.so host/krea2.cpp
printf 'built build/libkrea2.so\n'
