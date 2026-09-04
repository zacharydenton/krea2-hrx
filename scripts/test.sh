#!/usr/bin/env bash
# The one test command: generated kernels against their generators, every kernel test
# against the reference, the host build, and the native blocks against the fixture.
#   scripts/test.sh          everything (needs build/weights, build/fixture_step0.pt and the models)
#   scripts/test.sh --quick  kernels and generators only
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
quick=0; [ "${1:-}" = "--quick" ] && quick=1
tmpdir=$(mktemp -d); trap 'rm -rf "$tmpdir"' EXIT; export tmpdir
status=0
step() { local name="$1"; shift; printf '\n=== %s ===\n' "$name"; if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi; }
step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'
step "generated kernels match their generators" bash -c '
  cp -r kernels "$tmpdir/kernels" && cd "$tmpdir" && mkdir -p tools &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#" "$OLDPWD/tools/gen_prepare.py" > tools/gen_prepare.py &&
  sed "s#ROOT = Path(__file__).resolve().parent.parent#ROOT = Path(\"$tmpdir\")#; s#OUT = Path(__file__).resolve().parent.parent / \"kernels\"#OUT = Path(\"$tmpdir\") / \"kernels\"#" "$OLDPWD/tools/gen_attention_lds.py" > tools/gen_attention_lds.py &&
  python3 tools/gen_prepare.py >/dev/null && python3 tools/gen_attention_lds.py >/dev/null &&
  for f in prepare_norm_i4 prepare_gated_i4 prepare_swiglu_i4 attention_gqa_lds_f16_wmma; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done'
step "build host"                 ./scripts/build_host.sh
step "reference vs diffusers (toy)" bash -c 'source .venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_ref_vs_diffusers.py'
step "prepare kernels"            bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_prepare.py'
step "qk norm + rope"             bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_rope_qknorm.py'
step "attention"                  bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_attention.py'
if [ "$quick" = 0 ]; then
  step "native blocks vs reference (fixture)" bash -c 'source .venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_blocks.py --curve 1,28'
fi
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
