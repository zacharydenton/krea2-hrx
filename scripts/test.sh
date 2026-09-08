#!/usr/bin/env bash
# The one test command: generated kernels against their generators, every kernel test
# against the reference, the host build, and the native blocks against the fixture.
#   scripts/test.sh          everything (needs ComfyUI's checkpoint, build/fixture_step0.pt and the models)
#   scripts/test.sh --quick  host, API and kernel regressions
#   scripts/test.sh --native include full pipeline comparisons against Torch
#   scripts/test.sh --quality include eight-step latent/image quality vs an archived baseline
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
quick=0; native=0; quality=0
for arg in "$@"; do
  case "$arg" in
    --quick) quick=1 ;;
    --native) native=1 ;;
    --quality) quality=1 ;;
    *) echo "unknown test option: $arg" >&2; exit 2 ;;
  esac
done
tmpdir=$(mktemp -d); trap 'rm -rf "$tmpdir"' EXIT; export tmpdir
status=0
# Each step's wall time, so "the suite is slow" is answerable from the log.
timings=()
step() {
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  local start=$SECONDS result=0
  "$@" || result=1
  local elapsed=$((SECONDS - start))
  timings+=("$elapsed $name")
  if [ "$result" = 0 ]; then
    printf '  ok (%d s)\n' "$elapsed"
  else
    printf '  FAILED after %d s: %s\n' "$elapsed" "$name"
    status=1
  fi
  return $result
}
step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom kernels/native/*.loom'
step "generated kernels match their generators" bash -c '
  cp -r kernels "$tmpdir/kernels" && mkdir -p "$tmpdir/experiments" && cd "$tmpdir" && mkdir -p tools &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_prepare.py" > tools/gen_prepare.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#; s#OUT = Path(__file__).resolve().parent.parent / \"kernels\"#OUT = Path(\"$tmpdir\") / \"kernels\"#" "$OLDPWD/tools/gen_attention_lds.py" > tools/gen_attention_lds.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_native_ops.py" > tools/gen_native_ops.py &&
  python3 tools/gen_native_ops.py && diff -r kernels/native "$OLDPWD/kernels/native" &&
  cp "$OLDPWD/tools/gen_attention_query.py" tools/gen_attention_query.py && cp "$OLDPWD/tools/gen_attention_query32.py" tools/gen_attention_query32.py && python3 tools/gen_attention_query32.py >/dev/null &&
  cp "$OLDPWD/tools/gen_sage_attention.py" tools/gen_sage_attention.py && cp "$OLDPWD/tools/gen_gemm.py" tools/gen_gemm.py &&
  python3 tools/gen_prepare.py >/dev/null && python3 tools/gen_attention_lds.py >/dev/null && python3 tools/gen_sage_attention.py >/dev/null && python3 tools/gen_gemm.py >/dev/null &&
  for f in attention_query32 attention_gqa_lds_f16_wmma prepare_norm_i4 prepare_gated_i4 prepare_plain_i4 prepare_norm_i8 prepare_gated_i8 prepare_plain_i8 attention_sage_i4_fast attention_sage_i4_fast_prefetch attention_sage_i8_fast attention_sage_i8_fast_prefetch gemm_i4_256 gemm_i4_resid_256 gemm_i4_swiglu_256 gemm_i8_256 gemm_i8_resid_256 gemm_i8_swiglu_256; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done'
step "build"                      ./scripts/build.sh
step "Python runtime regressions" bash -c 'source .venv/bin/activate && python3 tests/test_runtime.py'
step "Rust formatting and lints" bash -c 'cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings'
# Covers the HRX dispatch and dependency audit, the checkpoint and constructor
# regressions, and the softmax shared-memory repeat, all of which were separate
# C++ programs.
step "Rust workspace tests" cargo test --quiet --workspace
step "CPU trajectory quality gate" .venv/bin/python tests/test_quality_gate.py
step "CPU Turbo and Raw scheduler regressions" .venv/bin/python tests/test_schedule.py
step "CPU fp16 attention lane model and benchmark oracle" python3 tests/test_attention_query_cpu.py
step "repeat-image failure capture" python3 tests/test_bench_native.py
step "auxiliary Loom kernel regressions" bash -c 'source .venv/bin/activate && python3 tests/test_native_ops.py'
step "reference vs diffusers (toy)" bash -c 'source .venv/bin/activate && python3 tests/test_ref_vs_diffusers.py'
step "prepare kernels"            bash -c 'python3 tests/test_prepare.py'
step "INT4 GEMM epilogues, both tiles, vs float64" bash -c '.venv/bin/python tests/test_gemm_i4.py && GEMM_KPAD=128 .venv/bin/python tests/test_gemm_i4.py'
step "INT8 GEMM epilogues vs float64"  bash -c 'GEMM_BITS=8 .venv/bin/python tests/test_gemm_i4.py && GEMM_BITS=8 GEMM_KPAD=64 .venv/bin/python tests/test_gemm_i4.py'
step "qk norm + rope"             bash -c 'python3 tests/test_rope_qknorm.py'
step "FP16 attention reference"   bash -c 'python3 tests/test_attention.py'
step "gfx1151 attention vs oracle" bash -c '.venv/bin/python tests/test_sage_attention.py'
if [ "$quick" = 0 ]; then
  step "native blocks vs reference (fixture)" bash -c 'source .venv/bin/activate && python3 tests/test_blocks.py --curve 1,28'
fi
if [ "$native" = 1 ]; then
  step "native scheduler and weight reuse regressions" bash -c '.venv/bin/python tests/test_native_regressions.py'
  step "native pipeline vs reference" bash -c '.venv/bin/python tests/test_native_pipeline.py'
fi
if [ "$quality" = 1 ]; then
  step "eight-step latent and image quality" .venv/bin/python tools/quality_vs_bf16.py regression \
    --baseline "${KREA2_QUALITY_BASELINE:-build/quality}" --work "$tmpdir/quality"
fi
printf '\ntotal %d s. Slowest steps:\n' "$SECONDS"
printf '%s\n' "${timings[@]}" | sort -rn | head -8 |
  while read -r seconds rest; do printf '  %5d s  %s\n' "$seconds" "$rest"; done
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
