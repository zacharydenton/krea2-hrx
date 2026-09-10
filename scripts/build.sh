#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo build --release
mkdir -p build
install -m 755 target/release/krea2 build/krea2
printf 'built build/krea2\n'
