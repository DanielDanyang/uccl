#!/bin/bash
# Devmem Validation Test Runner
# Purpose: Verify single-process can use multiple channels on same NIC

set -e

ROLE=${1:-server}
PEER_IP=${2:-}
GPU_ID=${3:-0}
DEV_ID=${4:-0}

# Map dev_id to NIC name for display
case $DEV_ID in
    0) NIC="eth1" ;;
    1) NIC="eth2" ;;
    2) NIC="eth3" ;;
    3) NIC="eth4" ;;
    *) NIC="unknown" ;;
esac

echo "=== Devmem Validation Test ==="
echo "Role: $ROLE"
echo "GPU: $GPU_ID"
echo "Dev: $DEV_ID ($NIC)"
echo "Channels: 4 (all on same NIC)"
echo "=============================="
echo ""

# Build if needed
if [ ! -f tests/test_devmem_validation ]; then
    echo "Building test..."
    make test_devmem_validation
fi

# Set environment - try multiple plugin paths
if [ -f "/usr/local/tcpx/lib64/libnccl-net.so" ]; then
    PLUGIN_PATH="/usr/local/tcpx/lib64/libnccl-net.so"
elif [ -f "/var/lib/tcpx/lib64/libnccl-net.so" ]; then
    PLUGIN_PATH="/var/lib/tcpx/lib64/libnccl-net.so"
else
    echo "Error: TCPX plugin not found"
    echo "Tried: /usr/local/tcpx/lib64/libnccl-net.so"
    echo "Tried: /var/lib/tcpx/lib64/libnccl-net.so"
    exit 1
fi

export NCCL_GPUDIRECTTCPX_PLUGIN_PATH=$PLUGIN_PATH
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=$NIC

echo "Using plugin: $PLUGIN_PATH"

# Run test
if [ "$ROLE" = "server" ]; then
    echo "Starting server on GPU $GPU_ID, Dev $DEV_ID ($NIC)..."
    ./tests/test_devmem_validation server 0.0.0.0 $GPU_ID $DEV_ID
else
    if [ -z "$PEER_IP" ]; then
        echo "Error: Client mode requires peer IP"
        echo "Usage: $0 client <peer_ip> [gpu_id] [dev_id]"
        exit 1
    fi
    echo "Starting client on GPU $GPU_ID, Dev $DEV_ID ($NIC), connecting to $PEER_IP..."
    ./tests/test_devmem_validation client $PEER_IP $GPU_ID $DEV_ID
fi

