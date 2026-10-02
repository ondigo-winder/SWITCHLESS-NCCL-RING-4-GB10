# SWITCHLESS-NCCL-RING-4-GB10

**NCCL over a switchless 4-node NVIDIA DGX Spark / GB10 ring, using both PCIe
halves of every QSFP cable: ~193 Gb/s all-reduce bus bandwidth per cable instead
of ~112 Gb/s, and up to +34% vLLM prefill throughput.**

This is one small patch on top of the hardened switchless NCCL 2.30.7 from
[alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) v0.0.1.
With the old settings (two HCAs) it behaves exactly like v0.0.1.

> **Status: experimental.** Tested on one 4-node cluster: collective benchmarks,
> an 11-minute soak, a full reboot, and vLLM with tensor parallelism 4 on BF16,
> FP8 and NVFP4 models up to Qwen3.5-397B. See [results](results/2026-10-gb10-ring.md).
> Unofficial: not affiliated with or endorsed by NVIDIA, Dell or OpenFaaS Ltd.
> Use at your own risk.

## Results at a glance

4 × Dell Pro Max GB10, ring of four QSFP cables, NCCL all-reduce, 1 GiB:

| Library | HCAs per node | Bus bandwidth |
| --- | --- | --- |
| switchless-nccl v0.0.1 | 2 (one half per cable) | 111.6 Gb/s |
| this patch | 4 (both halves per cable) | **193.1 Gb/s** |

Raw RDMA over both halves of one cable tops out at ~196 Gb/s, so the collective
reaches ~98% of what the cable carries.

vLLM, tensor parallelism 4, prefill of 4,096-token prompts:

| Model | v0.0.1 | this patch | Gain |
| --- | --- | --- | --- |
| Qwen2.5-72B-Instruct, BF16 | 893 tok/s | 982 tok/s | +10% |
| Qwen2.5-72B-Instruct, FP8 | 1,482 tok/s | 1,801 tok/s | +22% |
| Llama-3.3-70B-Instruct, NVFP4 | 1,945 tok/s | 2,605 tok/s | +34% |
| Qwen3.5-397B-A17B, NVFP4 (MoE) | 2,066 tok/s | 2,601 tok/s | +26% |

The faster the model computes per token, the more it gains. Decode gains 2–8%
(latency-bound). One run per configuration; treat the percentages as indicative.

## Why

On a GB10 each QSFP port shows up as two RoCE devices, one per PCIe x4 Gen5
link (~128 Gb/s each). One cable therefore needs both devices to reach its full
rate: `rocep1s0f0` + `roceP2p1s0f0` for the f0 cable, `rocep1s0f1` +
`roceP2p1s0f1` for the f1 cable.

The hardened switchless patch deliberately accepts exactly two listener GIDs,
so it can use only one half of each cable. Adding the other two HCAs stops NCCL
with `SWITCHLESS/HARDENED: more than two eligible listener GIDs`.

## What the patch changes

All changes are in `src/transport/net_ib/connect.cc` and are marked
`Modified by switchless-4hca`.

1. **Four listener addresses instead of two.** Four full GIDs would make
   `ncclIbHandle` exactly 128 bytes, which is over NCCL's limit. In switchless
   mode the same 32 bytes therefore carry up to four IPv4 addresses (RoCEv2
   IPv4-mapped GIDs) plus a tag. A non-IPv4 GID, a duplicate, a count other than
   2 or 4, or a peer running a different build is a hard error.
2. **Balanced device choice.** Upstream subnet-aware routing takes the *first*
   local device on the peer's subnet, which would put every channel on one half.
   In switchless mode the choice is spread over all matching devices.
3. **Quieter log.** The routing decision is logged once per connection instead
   of on every progress call.

Pairing half 1 with half 1 and half 2 with half 2 needs no new code: both sides
already pick their device by subnet, so each half needs its own /24. NIC merging
stays forbidden (`NCCL_IB_MERGE_NICS=0`).

## Requirements

Tested with:

| Component | Version |
| --- | --- |
| Hardware | 4 × Dell Pro Max GB10 (DGX Spark class), ConnectX-7 |
| OS | DGX OS 7.6.0, kernel 7.0.0-1019-nvidia |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (`nvcc` on the host, for building) |
| NCCL base | NVIDIA NCCL v2.30.7-1 + switchless-nccl v0.0.1 patches |
| vLLM image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (PyTorch 2.13, CUDA 13) |

Fabric:

- Ring cabling, one QSFP cable per neighbour (node1–node2–node3–node4–node1).
- MTU 9000 on all four ring interfaces of every node.
- **Every PCIe half of every cable on its own point-to-point /24.** The layout
  used here:

| Cable | Half 1 (`enp1s0…`) | Half 2 (`enP2p1s0…`) |
| --- | --- | --- |
| node1 ↔ node2 (f0) | `10.10.12.0/24` | `10.10.112.0/24` |
| node2 ↔ node3 (f1) | `10.10.23.0/24` | `10.10.123.0/24` |
| node3 ↔ node4 (f0) | `10.10.34.0/24` | `10.10.134.0/24` |
| node4 ↔ node1 (f1) | `10.10.41.0/24` | `10.10.141.0/24` |

- RoCEv2 GID index 3 (IPv4-mapped) present on all four devices
  (`cat /sys/class/infiniband/<dev>/ports/1/gids/3`).
- A management network for bootstrap (here a 10 GbE port, `enP7s7`).

## Build

On one GB10, with `git`, `gcc`, `make` and CUDA 13.0:

1. Download NVIDIA NCCL [v2.30.7-1](https://github.com/NVIDIA/nccl/releases/tag/v2.30.7-1),
   "Source code (tar.gz)" (`nccl-2.30.7-1.tar.gz`).
2. Download the switchless-nccl [v0.0.1](https://github.com/alexellis/switchless-nccl/releases)
   release asset `nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz` and
   extract it. It contains the three base patches.
3. Build:

```bash
git clone https://github.com/ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10.git
cd SWITCHLESS-NCCL-RING-4-GB10
tar xzf ~/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz -C ~
scripts/build.sh ~/nccl-2.30.7-1.tar.gz \
    ~/nccl-2.30.7-switchless-hardened-sm121-linux-arm64 ~/nccl-switchless-4hca
```

The script checks the patch checksum and the Git tree after every step and stops
on any mismatch. The build takes about 5 minutes on 20 cores. Copy
`~/nccl-switchless-4hca/` to every node and check `SHA256SUMS` there.

## Use

Same as switchless-nccl v0.0.1: load the library with `LD_PRELOAD` (and, for
vLLM, `VLLM_NCCL_SO_PATH`), bootstrap over the management network, ring-only.
The settings that differ or are required:

```bash
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
NCCL_MIN_NCHANNELS=8
NCCL_MAX_NCHANNELS=8
NCCL_SWITCHLESS_RING_ONLY=1        # required
NCCL_IB_SUBNET_AWARE_ROUTING=1     # required
NCCL_IB_MERGE_NICS=0               # required
NCCL_IB_GID_INDEX=3
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_ALGO=Ring
```

Every rank must use this build. A rank running v0.0.1 or stock NCCL is rejected
during connection setup.

With vLLM, the optional `deep_ep` module logs a `Duplicate NCCL runtime found`
warning at start-up and is skipped; it is only used for MoE expert parallelism.

## Limitations

- **Ring only.** Ranks that share no cable cannot talk directly: send/recv to the
  opposite node fails and all-to-all hangs, with or without this patch. Tensor
  parallelism and neighbour pipeline parallelism work; expert parallelism does not.
- **Exactly 2 or 4 HCAs per node, IPv4-mapped RoCEv2 GIDs only.**
- **GPUDirect RDMA is reported as disabled** on this platform, also with stock
  NCCL. NCCL stages through system memory, which on GB10 is the same unified
  LPDDR5X as GPU memory.
- Tested on one cluster of four nodes only.

## Tests

| File | What it does |
| --- | --- |
| `tests/allreduce.cu` | Minimal all-reduce benchmark with result check (no MPI or PyTorch) |
| `tests/collectives.cu` | all_gather, reduce_scatter, broadcast, reduce, send/recv, plus the two operations a ring cannot carry |
| `tests/run-ring.sh` | Starts a test binary on every node over SSH + Docker |
| `tests/counters.sh` | Prints RoCE and PHY error counters for a before/after check |

Build a test on every node inside a CUDA 13 container, linked against the patched
library mounted at `/pn`:

```bash
nvcc -O2 -std=c++17 -gencode arch=compute_121,code=sm_121 allreduce.cu -o allreduce \
     -L/pn -l:libnccl.so.2 -Xlinker -rpath=/pn
```

`tests/run-ring.sh` defaults to this cluster (hosts `gb10-01`…`gb10-04`, rank 0
at `192.168.1.101`, management interface `enP7s7`). Override with the `NODES`,
`MASTER`, `MGMT_IF`, `LIBDIR` and `IMAGE` environment variables:

```bash
NODES="node1 node2 node3 node4" MASTER=10.0.0.1 tests/run-ring.sh ring8 \
  "-e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1 -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8"
```

## Repository layout

| Path | Contents |
| --- | --- |
| `patches/` | The patch and its SHA-256 |
| `scripts/build.sh` | Verified build from source |
| `tests/` | Benchmarks and helpers |
| `results/` | Full measurements, set-up and method |
| `PROVENANCE.md` | Exact sources, hashes and Git trees |

## Credits

- [NVIDIA NCCL](https://github.com/NVIDIA/nccl), including the DGX Spark
  subnet-aware routing this builds on.
- [SparkRing](https://github.com/FujitsuPolycom/sparkring) for the original
  switchless patches.
- [Alex Ellis / OpenFaaS Ltd](https://github.com/alexellis/switchless-nccl) for
  the hardened switchless patch and the 4-node recipe.
- Joseph Rose for the earlier skip-Tree/skip-PAT approach (prior art, credited
  as in switchless-nccl; no code from it is included).

## Licence

Apache-2.0 for the contents of this repository (see `LICENSE` and `NOTICE`).
NCCL itself is under NVIDIA's licence; keep its `LICENSE.txt` and
`ThirdPartyNotices.txt` with any binary you distribute.
