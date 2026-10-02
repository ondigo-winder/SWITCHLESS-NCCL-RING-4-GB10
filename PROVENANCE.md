# Provenance

## NVIDIA NCCL source

- repository: `https://github.com/NVIDIA/nccl.git`, release tag `v2.30.7-1`
- source archive used: `nccl-2.30.7-1.tar.gz` (GitHub "Source code (tar.gz)"),
  SHA-256 `292a7f7a27b6754acaf46b5506a60758ca7b18cc1dfbd3d1d4e1e229d0863b4e`
- unmodified Git tree: `3e7de6f92f0190d1afe9f05642e634cbf43ae4c9`
- licence: NVIDIA's NCCL licence (see `LICENSE.txt` in the NCCL source)

## Base patches (not included here)

This patch applies on top of the three patches shipped in release `v0.0.1` of
[alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)
(asset `nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz`,
SHA-256 `b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f`):

| Patch | SHA-256 | Tree after applying |
| --- | --- | --- |
| `nccl-2.30.7-skip-tree-pat.patch` (SparkRing) | `097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2` | |
| `nccl-2.30.7-advertise-all-listener-gids.patch` (SparkRing) | `dccfce86d14c15c39f0e0a742863960205a3d9823c464b31a7f7389354844178` | `9e80bc2489864b4e6c6e2184af8797b07baa68f1` |
| `nccl-2.30.7-hardened-switchless.patch` (Alex Ellis, OpenFaaS Ltd) | `e2dd39eaefc022f99d5a3d3195e81947da20a4c7acbf1d688b4df8f6691c210c` | `560ba01b9becbc7d3daa1677f0216503fc3be631` |

All three are Apache-2.0. See that project's `PROVENANCE.md` for their origin
and for NVIDIA's DGX Spark subnet-aware routing work they extend.

## This patch

- file: `patches/nccl-2.30.7-switchless-4hca.patch`
- SHA-256: see `patches/SHA256SUMS`
- tree after applying: `f7d971c4aecd0d7f66e3a487a0b80050f6d21c19`
- touches one file: `src/transport/net_ib/connect.cc`; every change is marked
  `Modified by switchless-4hca`
- licence: Apache-2.0 (`LICENSE`)

## Build target

- Linux, ARM64, CUDA 13.0 (`nvcc` from DGX OS 7.6.0 on the GB10 itself)
- GPU code generation: `sm_121` only (GB10 / DGX Spark)
- output: `libnccl.so.2.30.7`
