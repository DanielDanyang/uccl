#!/bin/bash
# Devmem Validation Test Runner
# Purpose: Verify single-process can use multiple channels on same NIC

set -e

ROLE=${1:-server}
PEER_IP=${2:-}
GPU_ID=${3:-0}
NIC=${4:-eth1}

echo "=== Devmem Validation Test ==="
echo "Role: $ROLE"
echo "GPU: $GPU_ID"
echo "NIC: $NIC"
echo "Channels: 4 (all on same NIC)"
echo "=============================="
echo ""

# Build if needed
if [ ! -f tests/test_devmem_validation ]; then
    echo "Building test..."
    make test_devmem_validation
fi

# Set environment
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=$NIC
export UCCL_TCPX_NUM_CHANNELS=4
export UCCL_TCPX_WINDOW_SIZE=16
export UCCL_TCPX_CHUNK_BYTES=524288

# Run test
if [ "$ROLE" = "server" ]; then
    echo "Starting server on GPU $GPU_ID, NIC $NIC..."
    ./tests/test_devmem_validation server 0.0.0.0 $GPU_ID $NIC
else
    if [ -z "$PEER_IP" ]; then
        echo "Error: Client mode requires peer IP"
        echo "Usage: $0 client <peer_ip> [gpu_id] [nic]"
        exit 1
    fi
    echo "Starting client on GPU $GPU_ID, NIC $NIC, connecting to $PEER_IP..."
    ./tests/test_devmem_validation client $PEER_IP $GPU_ID $NIC
fi

