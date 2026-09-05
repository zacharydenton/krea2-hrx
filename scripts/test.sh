#!/usr/bin/env bash
# The one test command: generated kernels against their generators, every kernel test
# against the reference, the host build, and the native blocks against the fixture.
#   scripts/test.sh          everything (needs build/weights, build/fixture_step0.pt and the models)
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
step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom experiments/attention_gqa_lds_f16_wmma.loom'
step "generated kernels match their generators" bash -c '
  cp -r kernels "$tmpdir/kernels" && mkdir -p "$tmpdir/experiments" && cd "$tmpdir" && mkdir -p tools &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_prepare.py" > tools/gen_prepare.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#; s#OUT = Path(__file__).resolve().parent.parent / \"kernels\"#OUT = Path(\"$tmpdir\") / \"kernels\"#" "$OLDPWD/tools/gen_attention_lds.py" > tools/gen_attention_lds.py &&
  cp "$OLDPWD/tools/gen_sage_attention.py" tools/gen_sage_attention.py &&
  python3 tools/gen_prepare.py >/dev/null && python3 tools/gen_attention_lds.py >/dev/null && python3 tools/gen_sage_attention.py >/dev/null &&
  "$LOOM_FORMAT" --in-place experiments/attention_gqa_lds_f16_wmma.loom >/dev/null && cmp -s experiments/attention_gqa_lds_f16_wmma.loom "$OLDPWD/experiments/attention_gqa_lds_f16_wmma.loom" &&
  for f in prepare_norm_i4 prepare_gated_i4 prepare_plain_i4 attention_sage_i4_fast attention_sage_i4_fast_prefetch; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done'
step "build host"                 ./scripts/build_host.sh
step "Python runtime regressions" bash -c 'source .venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_runtime.py'
step "native constructor cleanup" bash -c '/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -Wall -Werror tests/test_session.cpp host/krea2.cpp host/sage.cpp -lhipblas -Wl,--wrap=hipMalloc,--wrap=hipFree -o "$tmpdir/test-session" && env -u LD_LIBRARY_PATH "$tmpdir/test-session" "$tmpdir"'
step "reference vs diffusers (toy)" bash -c 'source .venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_ref_vs_diffusers.py'
step "prepare kernels"            bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_prepare.py'
step "qk norm + rope"             bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_rope_qknorm.py'
step "FP16 attention reference"   bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_attention.py'
step "gfx1151 attention vs oracle" bash -c 'env -u LD_LIBRARY_PATH .venv/bin/python tests/test_sage_attention.py'
if [ "$quick" = 0 ]; then
  step "native blocks vs reference (fixture)" bash -c 'source .venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_blocks.py --curve 1,28'
fi
if [ "$native" = 1 ]; then
  if step "build native pipeline" ./scripts/build_native.sh; then
    step "native scheduler and weight reuse regressions" bash -c 'env -u LD_LIBRARY_PATH .venv/bin/python tests/test_native_regressions.py'
    step "native pipeline vs reference" bash -c 'env -u LD_LIBRARY_PATH .venv/bin/python tests/test_native_pipeline.py'
  fi
fi
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
