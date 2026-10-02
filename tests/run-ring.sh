#!/usr/bin/env bash
# Run tests/allreduce on N nodes over SSH + Docker, rank i on NODES[i].
# Each node needs: ~/nt/allreduce (built from tests/allreduce.cu against $LIBDIR), Docker with the
# NVIDIA runtime, and the patched library in $LIBDIR. Edit the variables below for your cluster.
#   tests/run-ring.sh <label> "<extra docker -e flags>"
set -uo pipefail
LABEL=${1:-run}; EXTRA=${2:-}
NODES=(${NODES:-gb10-01 gb10-02 gb10-03 gb10-04})
MASTER=${MASTER:-192.168.1.101}             # management IP of NODES[0]
MGMT_IF=${MGMT_IF:-enP7s7}
LIBDIR=${LIBDIR:-\$HOME/nccl-switchless-4hca}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.09-py3}   # any image with CUDA 13 userspace
SSH=${SSH:-ssh}
N=${#NODES[@]}
BASE="-e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_NET=IB -e NCCL_IB_GID_INDEX=3 \
 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_ALGO=Ring \
 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_P2P_LEVEL=SYS -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 \
 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_SOCKET_IFNAME=$MGMT_IF -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
run() { $SSH "${NODES[$1]}" "docker run --rm --network host --ipc host --shm-size 32g --gpus all \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 -v \$HOME/nt:/nt -v $LIBDIR:/pn:ro \
  $BASE $EXTRA $IMAGE /nt/allreduce $1 $N $MASTER"; }
for ((r=N-1; r>=1; r--)); do run $r > "$LABEL.r$r.log" 2>&1 & done
sleep 2
timeout 150 bash -c "$(declare -f run); NODES=(${NODES[*]}); SSH='$SSH' LIBDIR='$LIBDIR' BASE='$BASE' EXTRA='$EXTRA' IMAGE='$IMAGE' N=$N MASTER=$MASTER; run 0" > "$LABEL.r0.log" 2>&1
wait
echo "### $LABEL"; grep -E " MiB |DONE" "$LABEL.r0.log" | grep -v INFO
grep -hE "NCCL WARN|ibv_modify_qp" "$LABEL".r*.log | head -5
