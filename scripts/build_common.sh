# Shared plain-C++ build support. Sourced by build_host.sh and build_native.sh.
HRX_SOURCE="${HRX_SOURCE:-$HOME/code/hrx-system}"
HRX_BUILD="${HRX_BUILD:-$HRX_SOURCE/build-cuda}"
HSA_PROVIDER="${HSA_PROVIDER:-$HOME/.local/rocm-hrx}"
CXX="${CXX:-c++}"
mkdir -p build/runtime build/obj

# Publish runtime copies with rename so an existing process keeps its mappings.
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

CXXFLAGS=(-std=c++17 -O2 -Wall -Werror -fPIC -I"$HRX_SOURCE/libhrx/include")
HRXLIBS=(-Lbuild/runtime -lhrx -Wl,-rpath,'$ORIGIN/runtime')
COMMON=(gpu native_kernels krea2 sage)
# Dependency files make regeneration of embedded Loom sources rebuild only the
# affected translation units. Command changes also invalidate the objects.
compile() {
  local name="$1" obj="build/obj/$1.o" dep="build/obj/$1.d"
  local signature="$CXX ${CXXFLAGS[*]}"
  if [ ! -f "$obj" ] || [ "$(cat "$obj.cmd" 2>/dev/null || true)" != "$signature" ] ||
      ! make -s -q -f "$dep" -f <(printf '%s:\n\tfalse\n' "$obj") "$obj" 2>/dev/null; then
    "$CXX" "${CXXFLAGS[@]}" -MMD -MP -MF "$dep" -c "host/$name.cpp" -o "$obj"
    printf '%s' "$signature" > "$obj.cmd"
  fi
}
link_shared() {
  local output="$1"; shift
  local objects=() name
  for name in "$@"; do compile "$name"; objects+=("build/obj/$name.o"); done
  "$CXX" -shared "${objects[@]}" "${HRXLIBS[@]}" -o "$output.tmp"
  mv -f "$output.tmp" "$output"
}
link_pipeline() {
  local output="$1" name
  local objects=()
  for name in native_ops native_models native_tokenizer native_compile native_pipeline; do
    compile "$name"
    objects+=("build/obj/$name.o")
  done
  "$CXX" -shared "${objects[@]}" -Lbuild -lkrea2 -Wl,-rpath,'$ORIGIN' -o "$output.tmp"
  mv -f "$output.tmp" "$output"
}
