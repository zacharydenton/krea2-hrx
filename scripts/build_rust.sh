#!/usr/bin/env bash
# The Rust side of the runtime, installed into build/ under the names the
# tests, tools and docs already use. Needs cargo; the C++ libraries do not.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v cargo > /dev/null; then
  printf 'cargo not found: install Rust to build the CLI and the Rust test bridges\n' >&2
  exit 1
fi
cargo build --release --workspace
install -m 755 target/release/krea2 build/krea2
printf 'built build/krea2\n'
