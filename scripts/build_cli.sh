#!/usr/bin/env bash
# The Rust CLI (cli/), linked against the runtime's C ABI, landing as build/krea2.
# Needs cargo; the libraries themselves need only a C++17 compiler and HRX.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v cargo > /dev/null; then
  printf 'cargo not found: install Rust to build the CLI (the libraries are already built)\n' >&2
  exit 1
fi
if [ ! -f build/libkrea2_pipeline.so ]; then
  printf 'build/libkrea2_pipeline.so missing: run scripts/build_native.sh first\n' >&2
  exit 1
fi
cargo build --release --manifest-path cli/Cargo.toml
install -m 755 cli/target/release/krea2 build/krea2
printf 'built build/krea2\n'
