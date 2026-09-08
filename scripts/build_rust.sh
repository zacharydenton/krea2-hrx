#!/usr/bin/env bash
# The runtime, installed into build/ under the names the tests, tools and docs
# already use. One cdylib carries both C ABIs, so a process that loads them
# both cannot end up with two HRX runtimes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v cargo > /dev/null; then
  printf 'cargo not found: install Rust to build krea2\n' >&2
  exit 1
fi
cargo build --release --workspace
install -m 755 target/release/krea2 build/krea2
# Beside the C++ artifacts they are replacing while both exist: KREA2_LIB
# points krea2_loom.py at either library.
install -m 644 target/release/libkrea2.so build/libkrea2_rust.so
install -m 755 target/release/krea2-native-components build/krea2-native-components-rust
printf 'built build/krea2, build/libkrea2_rust.so and build/include/krea2*.h\n'
