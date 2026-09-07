#!/usr/bin/env bash
# The one test command: generated kernels against their generators, every kernel test
# against the reference, the host build, and the native blocks against the fixture.
#   scripts/test.sh          everything (needs ComfyUI's checkpoint, build/fixture_step0.pt and the models)
#   scripts/test.sh --quick  host, API and kernel regressions
#   scripts/test.sh --native include full native pipeline comparisons
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
quick=0; native=0
for arg in "$@"; do
  case "$arg" in
    --quick) quick=1 ;;
    --native) native=1 ;;
    *) echo "unknown test option: $arg" >&2; exit 2 ;;
  esac
done
tmpdir=$(mktemp -d); trap 'rm -rf "$tmpdir"' EXIT; export tmpdir
status=0
step() { local name="$1"; shift; printf '\n=== %s ===\n' "$name"; if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; return 1; fi; }
step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom kernels/native/*.loom experiments/attention_gqa_lds_f16_wmma.loom experiments/gemm_down_i4.loom'
step "generated kernels match their generators" bash -c '
  cp -r kernels "$tmpdir/kernels" && mkdir -p "$tmpdir/experiments" && cd "$tmpdir" && mkdir -p tools host &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_prepare.py" > tools/gen_prepare.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#; s#OUT = Path(__file__).resolve().parent.parent / \"kernels\"#OUT = Path(\"$tmpdir\") / \"kernels\"#" "$OLDPWD/tools/gen_attention_lds.py" > tools/gen_attention_lds.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_native_ops.py" > tools/gen_native_ops.py &&
  python3 tools/gen_native_ops.py && cmp -s host/native_sources.h "$OLDPWD/host/native_sources.h" && diff -r kernels/native "$OLDPWD/kernels/native" &&
  cp "$OLDPWD/tools/gen_sage_attention.py" tools/gen_sage_attention.py && cp "$OLDPWD/tools/gen_gemm.py" tools/gen_gemm.py &&
  python3 tools/gen_prepare.py >/dev/null && python3 tools/gen_attention_lds.py >/dev/null && python3 tools/gen_sage_attention.py >/dev/null && python3 tools/gen_gemm.py >/dev/null &&
  "$LOOM_FORMAT" --in-place experiments/attention_gqa_lds_f16_wmma.loom >/dev/null && cmp -s experiments/attention_gqa_lds_f16_wmma.loom "$OLDPWD/experiments/attention_gqa_lds_f16_wmma.loom" &&
  for f in prepare_norm_i4 prepare_gated_i4 prepare_plain_i4 prepare_norm_i8 prepare_gated_i8 prepare_plain_i8 attention_sage_i4_fast attention_sage_i4_fast_prefetch attention_sage_i8_fast attention_sage_i8_fast_prefetch gemm_i4_256 gemm_i4_resid_256 gemm_i4_swiglu_256 gemm_i8_256 gemm_i8_resid_256 gemm_i8_swiglu_256; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done'
step "build host"                 ./scripts/build_host.sh
step "HRX dispatch and dependency audit" env -u LD_LIBRARY_PATH build/test-hrx-runtime
step "Python runtime regressions" bash -c 'source .venv/bin/activate && python3 tests/test_runtime.py'
step "repeat-image failure capture" python3 tests/test_bench_native.py
step "softmax shared-memory reuse regression" bash -c 'source scripts/build_common.sh && "$CXX" "${CXXFLAGS[@]}" tests/test_softmax_repeat.cpp -Lbuild -lkrea2 -Wl,-rpath,"$PWD/build" -o "$tmpdir/test-softmax-repeat" && "$tmpdir/test-softmax-repeat"'
step "native constructor cleanup" bash -c 'source scripts/build_common.sh && "$CXX" "${CXXFLAGS[@]}" tests/test_session.cpp build/obj/{gpu,krea2,sage,native_kernels}.o "${HRXLIBS[@]}" -Wl,--wrap=hrx_buffer_allocate,--wrap=hrx_buffer_release -o "$tmpdir/test-session" && "$tmpdir/test-session" "$tmpdir"'
step "auxiliary Loom kernel regressions" bash -c 'source .venv/bin/activate && python3 tests/test_native_ops.py'
step "reference vs diffusers (toy)" bash -c 'source .venv/bin/activate && python3 tests/test_ref_vs_diffusers.py'
step "prepare kernels"            bash -c 'python3 tests/test_prepare.py'
step "INT4 GEMM epilogues, both tiles, vs float64" bash -c '.venv/bin/python tests/test_gemm_i4.py && GEMM_KPAD=128 .venv/bin/python tests/test_gemm_i4.py'
step "INT8 GEMM epilogues vs float64"  bash -c 'GEMM_BITS=8 .venv/bin/python tests/test_gemm_i4.py && GEMM_BITS=8 GEMM_KPAD=64 .venv/bin/python tests/test_gemm_i4.py'
step "wide INT4 down projection"  bash -c '.venv/bin/python tests/test_gemm_down.py'
step "qk norm + rope"             bash -c 'python3 tests/test_rope_qknorm.py'
step "FP16 attention reference"   bash -c 'python3 tests/test_attention.py'
step "gfx1151 attention vs oracle" bash -c '.venv/bin/python tests/test_sage_attention.py'
if [ "$quick" = 0 ]; then
  step "native blocks vs reference (fixture)" bash -c 'source .venv/bin/activate && python3 tests/test_blocks.py --curve 1,28'
fi
if [ "$native" = 1 ]; then
  if step "build native pipeline" ./scripts/build_native.sh; then
    step "Unicode normalization, tokenizer and SHA-256" bash -c '.venv/bin/python tests/test_unicode.py'
    step "native scheduler and weight reuse regressions" bash -c '.venv/bin/python tests/test_native_regressions.py'
    step "native pipeline vs reference" bash -c '.venv/bin/python tests/test_native_pipeline.py'
  fi
fi
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
