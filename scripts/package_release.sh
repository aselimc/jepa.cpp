#!/usr/bin/env bash
# Package a built tree into the release archive .github/workflows/release.yml publishes.
#
#   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
#         -DJEPA_NATIVE=OFF -DJEPA_BUILD_TESTS=OFF \
#         -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"
#   cmake --build build-release -j
#   scripts/package_release.sh --build-dir build-release --out-dir dist
#
# Produces dist/jepa-<version>-<platform>.tar.gz and dist/SHA256SUMS (the checksum of that
# archive). Inside the archive, SHA256SUMS lists every packaged file.
#
# --version defaults to the project version in CMakeLists.txt, which is also what the packaged
# `jepa-info --version` prints; the script refuses to package a tree where the two disagree, so a
# release cannot ship binaries built from a different version than the tag says.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-release"
OUT="$ROOT/dist"
VERSION=""
PLATFORM="linux-x86_64"

while [ $# -gt 0 ]; do
  case "$1" in
    --version)   VERSION="${2#v}"; shift 2 ;;
    --build-dir) BUILD="$2"; shift 2 ;;
    --out-dir)   OUT="$2"; shift 2 ;;
    --platform)  PLATFORM="$2"; shift 2 ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$VERSION" ]; then
  VERSION="$(sed -n 's/^project(.*VERSION \([0-9][0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt")"
fi
[ -n "$VERSION" ] || { echo "cannot determine the version" >&2; exit 1; }

TOOLS="jepa-info jepa-embed jepa-classify jepa-worldmodel jepa-quantize jepa-bench jepa-server"
for t in $TOOLS; do
  [ -x "$BUILD/$t" ] || { echo "missing $BUILD/$t — build first" >&2; exit 1; }
done

# The binaries have to agree with the version being packaged: jepa_version() comes from the CMake
# project version (CMakeLists.txt), so a mismatch means the build tree is stale or from another tag.
BUILT="$("$BUILD/jepa-info" --version)"
if [ "$BUILT" != "$VERSION" ]; then
  echo "version mismatch: packaging $VERSION but $BUILD/jepa-info reports $BUILT" >&2
  exit 1
fi

NAME="jepa-$VERSION-$PLATFORM"
STAGE="$OUT/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/include" "$STAGE/lib"

for t in $TOOLS; do
  cp "$BUILD/$t" "$STAGE/bin/$t"
done
# The header alone cannot be linked against: ship the static libraries it needs with it.
cp "$BUILD/libjepa.a" "$STAGE/lib/"
cp "$BUILD"/ggml/src/libggml*.a "$STAGE/lib/"
cp "$ROOT/include/jepa.h" "$STAGE/include/"
cp "$ROOT/LICENSE" "$ROOT/README.md" "$ROOT/CHANGELOG.md" "$STAGE/"

# Checksums of the packaged files, relative to the archive root.
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS )

mkdir -p "$OUT"
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
rm -rf "$STAGE"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > SHA256SUMS )

echo "packaged $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
cat "$OUT/SHA256SUMS"
