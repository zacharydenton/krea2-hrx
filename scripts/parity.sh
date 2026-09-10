#!/usr/bin/env bash
# Accuracy against the unquantized BF16 model using saved, identical inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo test --release --test unquantized_parity -- --ignored --nocapture
