#!/bin/sh
# Build the driver library (file name libcuda.so.1, a Mach-O dylib) and the two test harnesses into out/.
#
#   MACUDA=<path to a macuda checkout>   default: ../macuda, next to this folder
#   ZSTD=<zstd install prefix>           default: brew --prefix zstd (brew install zstd)
#
# macuda must already be built: run `sh setup.sh shim` in the macuda checkout first. Nothing here touches the GPU.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
MACUDA=${MACUDA:-$here/../macuda}
[ -d "$MACUDA/cuda-shim" ] || { echo "build.sh: $MACUDA is not a macuda checkout (no cuda-shim/); set MACUDA=" >&2; exit 1; }
M=$(cd "$MACUDA/cuda-shim" && pwd)
if [ -z "${ZSTD:-}" ]; then ZSTD=$(brew --prefix zstd 2>/dev/null || true); fi
[ -n "$ZSTD" ] && [ -f "$ZSTD/lib/libzstd.a" ] && [ -f "$ZSTD/include/zstd.h" ] ||
  { echo "build.sh: zstd not found (brew install zstd, or set ZSTD=<prefix> with include/zstd.h and lib/libzstd.a)" >&2; exit 1; }
for f in nv/libtinynv.a copy1d_cubin.o keepalive_cubin.o; do
  [ -f "$M/build/shim/$f" ] || { echo "build.sh: $M/build/shim/$f missing: run  sh setup.sh shim  in $MACUDA" >&2; exit 1; }
done
SDK=$(sh "$M/build/sdk.sh")
mkdir -p "$here/out"
cc -isysroot "$SDK" -std=c11 -O2 -g -Wall -Wextra -Wno-unused-parameter -fvisibility=hidden -fPIC \
  -I"$M/include" -I"$ZSTD/include" \
  -dynamiclib -Wl,-exported_symbol,_cu\* -install_name libcuda.so.1 -o "$here/out/libcuda.so.1" "$here/tfcuda.c" \
  "$M/build/shim/nv/libtinynv.a" "$M/build/shim/copy1d_cubin.o" "$M/build/shim/keepalive_cubin.o" "$ZSTD/lib/libzstd.a"
for t in l2_harness graph_harness; do
  cc -isysroot "$SDK" -std=c11 -O2 -g -Wall -Wextra -ffp-contract=off -o "$here/out/$t" "$here/tests/$t.c"
done
echo "exported cu* symbols: $(nm -gU "$here/out/libcuda.so.1" | grep -c ' T _cu')"
