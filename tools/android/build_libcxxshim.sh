#!/usr/bin/env bash
# Author: David Li, SISHK

# Build the arm64 compatibility shim used by tools/patch_android_apk.py.
#
# Requires the Android NDK. Flet installs NDK 28 under ~/Android/sdk/ndk; set
# NDK to override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NDK="${NDK:-$HOME/Android/sdk/ndk/28.2.13676358}"
CLANGXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++"
OUT="$ROOT/tools/android/arm64-v8a"

if [[ ! -x "$CLANGXX" ]]; then
  echo "clang++ not found at $CLANGXX (set NDK=...)" >&2
  exit 1
fi

mkdir -p "$OUT"
"$CLANGXX" \
  -shared -fPIC -O2 -std=c++17 \
  -Wl,-soname,libcxxshim.so \
  -o "$OUT/libcxxshim.so" \
  "$ROOT/tools/android/libcxxshim.cpp"

echo "built $OUT/libcxxshim.so"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm" -D "$OUT/libcxxshim.so" \
  | grep __hash_memory
