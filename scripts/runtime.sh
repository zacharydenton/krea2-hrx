#!/usr/bin/env bash
# Stage libhrx and its HSA provider into build/runtime, which is what every
# artifact's rpath points at. Copies are published by rename, so a running
# process keeps the mapping it already has.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HRX_SOURCE="${HRX_SOURCE:-$HOME/code/hrx-system}"
HRX_BUILD="${HRX_BUILD:-$HRX_SOURCE/build-cuda}"
HSA_PROVIDER="${HSA_PROVIDER:-$HOME/.local/rocm-hrx}"
mkdir -p build/runtime

copy_runtime() {
  local source="$1" target="build/runtime/$2"
  if ! cmp -s "$source" "$target"; then
    cp -L "$source" "$target.tmp"
    mv -f "$target.tmp" "$target"
  fi
}
copy_runtime "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" libhrx.so.0
ln -sfn libhrx.so.0 build/runtime/libhrx.so
for name in libhsa-runtime64.so.1 librocm_sysdeps_elf.so.1 \
    librocm_sysdeps_drm.so.2 librocm_sysdeps_drm_amdgpu.so.1 \
    librocm_sysdeps_numa.so.1 librocm_sysdeps_z.so.1 librocm_sysdeps_zstd.so.1 \
    librocm_sysdeps_liblzma.so.5 librocm_sysdeps_bz2.so; do
  copy_runtime "$HSA_PROVIDER/$name" "$name"
done
# Use the installed ROCm registration ABI, also used by the optional Torch
# oracle. The provider's older registration ABI cannot share a process with it.
registration="$HSA_PROVIDER/librocprofiler-register.so.0"
if [ ! -f "$registration" ]; then registration=/opt/rocm/lib/librocprofiler-register.so.0; fi
copy_runtime "$registration" librocprofiler-register.so.0
while read -r name arrow path rest; do
  if [[ "$name" =~ ^lib(fmt|glog|gflags)\. ]] && [ "$arrow" = '=>' ]; then
    copy_runtime "$path" "$name"
  fi
done < <(ldd "$registration")
