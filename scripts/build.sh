#!/usr/bin/env bash
# Build NCCL 2.30.7 with the switchless patches plus the 4-HCA patch, for GB10 (sm_121).
# Usage: scripts/build.sh <nccl-2.30.7-1.tar.gz> <switchless-nccl v0.0.1 bundle dir> [out dir]
# The v0.0.1 bundle is the extracted release asset
# nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz from alexellis/switchless-nccl;
# it carries the three base patches this patch builds on.
set -euo pipefail
SRC_TGZ=$1; BASE=$2; OUT=${3:-$HOME/nccl-switchless-4hca}
HERE=$(cd "$(dirname "$0")/.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

want_tree() { local got; got=$(git write-tree); [ "$got" = "$1" ] || { echo "FAIL: tree $got, expected $1 ($2)"; exit 1; }; echo "ok: $2"; }

(cd "$HERE/patches" && sha256sum -c SHA256SUMS)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
tar xzf "$SRC_TGZ" -C "$WORK" --strip-components=1
cd "$WORK"; git init -q; git add -A
want_tree 3e7de6f92f0190d1afe9f05642e634cbf43ae4c9 "NVIDIA NCCL v2.30.7-1 source"
git apply --index "$BASE/nccl-2.30.7-skip-tree-pat.patch"
git apply --index "$BASE/nccl-2.30.7-advertise-all-listener-gids.patch"
want_tree 9e80bc2489864b4e6c6e2184af8797b07baa68f1 "SparkRing two-patch tree"
git apply --index "$BASE/nccl-2.30.7-hardened-switchless.patch"
want_tree 560ba01b9becbc7d3daa1677f0216503fc3be631 "switchless-nccl v0.0.1 hardened tree"
git apply --index "$HERE/patches/nccl-2.30.7-switchless-4hca.patch"
want_tree f7d971c4aecd0d7f66e3a487a0b80050f6d21c19 "switchless-4hca tree"

make -j"$(nproc)" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
mkdir -p "$OUT"
cp -a build/lib/libnccl.so* "$OUT/"
cp "$HERE/patches/nccl-2.30.7-switchless-4hca.patch" "$OUT/"
(cd "$OUT" && sha256sum libnccl.so.2.30.7 > SHA256SUMS && cat SHA256SUMS)
echo "built: $OUT/libnccl.so.2"
