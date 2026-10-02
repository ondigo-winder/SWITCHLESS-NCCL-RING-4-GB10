# switchless-nccl-4hca

NCCL over a switchless 4-node DGX Spark / GB10 ring, using **both PCIe halves of
every QSFP cable**: about **193 Gb/s** all-reduce bus bandwidth per cable instead
of about 112 Gb/s.

It is one extra patch on top of the hardened switchless NCCL 2.30.7 from
[alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) v0.0.1.
With the old settings (two HCAs) it behaves exactly like v0.0.1.

> Status: experimental. Tested on one 4-node cluster with collective benchmarks,
> after a reboot of all nodes, and under vLLM with tensor parallelism 4, including
> Qwen2.5-72B-Instruct in BF16 (results below).

## Why

On a GB10 each QSFP port shows up as two RoCE devices, one per PCIe x4 Gen5 link
(about 128 Gb/s each). One cable therefore needs both devices to reach its full
rate: `rocep1s0f0` + `roceP2p1s0f0` for the f0 cable, `rocep1s0f1` +
`roceP2p1s0f1` for the f1 cable.

The hardened switchless patch deliberately accepts exactly two listener GIDs,
so it can only use one half of each cable. Adding the other two HCAs stops NCCL
with `SWITCHLESS/HARDENED: more than two eligible listener GIDs`.

## What the patch changes

All changes are in `src/transport/net_ib/connect.cc`.

1. **Four listener addresses instead of two.** Four full GIDs would make
   `ncclIbHandle` exactly 128 bytes, which is over NCCL's limit. In switchless
   mode the same 32 bytes therefore carry up to four IPv4 addresses (RoCEv2
   IPv4-mapped GIDs) plus a tag. A non-IPv4 GID, a duplicate, or a count other
   than 2 or 4 is a hard error, and so is a peer running a different build.
2. **Balanced device choice.** Upstream subnet-aware routing takes the *first*
   local device on the peer's subnet. With both halves of a cable on the peer's
   subnets that would put every channel on one half. In switchless mode the
   choice is now spread over all matching devices.
3. **Quieter log.** The routing decision is logged once per connection instead
   of on every progress call.

Pairing half 1 with half 1 and half 2 with half 2 needs no new code: both sides
already pick their device by subnet, so each half needs its own /24 (see below).
NIC merging stays forbidden (`NCCL_IB_MERGE_NICS=0`).

## Fabric requirements

- Ring cabling, one cable per neighbour, MTU 9000 on all four interfaces of every node.
- **Every PCIe half of every cable gets its own point-to-point /24**, for example:

| Link | f0 half 1 | f0 half 2 |
| --- | --- | --- |
| node1 ↔ node2 (f0) | `10.10.12.0/24` | `10.10.112.0/24` |

| Link | f1 half 1 | f1 half 2 |
| --- | --- | --- |
| node2 ↔ node3 (f1) | `10.10.23.0/24` | `10.10.123.0/24` |

- RoCEv2 GID index 3 (IPv4-mapped) present on all four devices.

## Build

On a GB10 with CUDA 13.0, `gcc` and `make`:

```bash
# 1. NVIDIA NCCL v2.30.7-1 "Source code (tar.gz)" from GitHub
# 2. the switchless-nccl v0.0.1 release asset, extracted
tar xzf nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz
scripts/build.sh nccl-2.30.7-1.tar.gz nccl-2.30.7-switchless-hardened-sm121-linux-arm64 ~/nccl-switchless-4hca
```

The script checks the patch checksum and the Git tree after every step. The
build takes about 5 minutes on 20 cores. Copy the output directory to every node
and check `SHA256SUMS` there.

## Use

Same as switchless-nccl v0.0.1 (`LD_PRELOAD` / `VLLM_NCCL_SO_PATH`, ring-only,
merging off). The differences:

```bash
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1
NCCL_MIN_NCHANNELS=8
NCCL_MAX_NCHANNELS=8
NCCL_SWITCHLESS_RING_ONLY=1        # required
NCCL_IB_SUBNET_AWARE_ROUTING=1     # required
NCCL_IB_MERGE_NICS=0               # required
```

With vLLM, also set `VLLM_NCCL_SO_PATH` to the same library. The optional
`deep_ep` module then logs a `Duplicate NCCL runtime found` warning at start-up
and is skipped; it is only used for MoE expert parallelism and did not affect
the tensor-parallel test.

Every rank must use this build. A rank running v0.0.1 or stock NCCL is rejected
during connection setup.

## Test

`tests/allreduce.cu` is a minimal all-reduce benchmark (no MPI or PyTorch).
`tests/run-ring.sh` starts it on every node over SSH + Docker.
`tests/counters.sh` prints the RoCE and PHY error counters for a before/after check.
`tests/collectives.cu` checks all_gather, reduce_scatter, broadcast, reduce and
send/recv, plus the two operations a switchless ring cannot carry.

```bash
# on every node, inside a CUDA 13 container, link against the patched library:
nvcc -O2 -std=c++17 -gencode arch=compute_121,code=sm_121 allreduce.cu -o allreduce \
     -L/pn -l:libnccl.so.2 -Xlinker -rpath=/pn
# from the operator box:
tests/run-ring.sh ring8 "-e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1 -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8"
```

## Results

See [`results/2026-10-gb10-ring.md`](results/2026-10-gb10-ring.md). In short, 4 × Dell Pro Max GB10 in a ring:

| Configuration | 64 MiB | 1 GiB |
| --- | --- | --- |
| switchless-nccl v0.0.1, 2 HCAs | 109.5 Gb/s | 111.6 Gb/s |
| this patch, 4 HCAs, 8 channels | 182.7 Gb/s | **193.1 Gb/s** |

Raw RDMA (`ib_write_bw`, both halves of one cable at once) tops out at about
196 Gb/s, so the collective now reaches about 98% of what the cable carries.
Small messages are latency-bound and gain little.

all_gather, reduce_scatter, broadcast, reduce and neighbour send/recv gain the
same way (172 to 196 Gb/s). Send/recv between opposite nodes and all-to-all do
not work on a switchless ring, with or without this patch: no expert parallelism.

Under vLLM with Qwen2.5-72B-Instruct (BF16, TP4) the fabric is far from full
(about 30 Gb/s per cable during prefill), yet prefill ran about 10% faster than with
v0.0.1; decode was the same within noise. See the results file for the details.

GPUDirect RDMA is reported as disabled on this platform, also with stock NCCL.
NCCL stages through system memory, which on GB10 is the same unified LPDDR5X as
GPU memory.

## Credits

- NVIDIA NCCL, including the DGX Spark subnet-aware routing this builds on.
- SparkRing (FujitsuPolycom/sparkring) for the original switchless patches.
- Alex Ellis / OpenFaaS Ltd for the hardened switchless patch and recipe.

## Licence

Apache-2.0 for the contents of this repository. NCCL itself is under NVIDIA's
licence; keep its `LICENSE.txt` with any binary you distribute.
