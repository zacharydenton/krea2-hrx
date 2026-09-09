#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo build --release --workspace
mkdir -p build
install -m 755 target/release/krea2 build/krea2
install -m 644 target/release/libkrea2.so build/libkrea2.so
printf 'built build/krea2, build/libkrea2.so and build/include/krea2*.h\n'
