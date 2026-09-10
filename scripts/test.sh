#!/usr/bin/env bash
# CPU checks by default; --gpu includes native numerical regressions on gfx1151.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:---cpu}" in --cpu) gpu=0;; --gpu|--quick) gpu=1;; *) echo 'usage: scripts/test.sh [--cpu|--gpu]' >&2; exit 2;; esac
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --no-deps --lib
# A statically linked crate never shows up in /proc/self/maps, so the dependency
# set is pinned instead: an upstream default cannot enlarge the binary unseen.
shipped=$(mktemp) && allowed=$(mktemp) && trap 'rm -f "$shipped" "$allowed"' EXIT
cargo tree -p krea2 --edges normal --prefix none |
  cut -d' ' -f1 | grep -v '^$' | LC_ALL=C sort -u > "$shipped"
grep -v '^#' docs/dependencies.txt | grep -v '^$' | LC_ALL=C sort -u > "$allowed"
diff -u "$allowed" "$shipped"
cargo test
if [ "$gpu" = 1 ]; then
  # Keep model fixtures, benchmarks and the exhaustive CPU sweep opt-in.
  cargo test --lib ops::tensor::tests:: -- --ignored --test-threads=1
  cargo test --test arithmetic --test dispatch --test quantized \
    --test softmax_repeat --test uploads -- --ignored --test-threads=1
fi
