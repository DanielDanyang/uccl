#!/bin/bash
# Quick devmem validation test with correct plugin path

ROLE=${1:-server}
PEER_IP=${2:-}

echo "=== Quick Devmem Test ==="
echo "Role: $ROLE"
echo ""

# Use the correct plugin path
export NCCL_GPUDIRECTTCPX_PLUGIN_PATH=/usr/local/tcpx/lib64/libnccl-net.so
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1

if [ "$ROLE" = "server" ]; then
    echo "Starting server..."
    ./tests/test_devmem_validation server 0.0.0.0 0 0
else
    if [ -z "$PEER_IP" ]; then
        echo "Usage: $0 client <peer_ip>"
        exit 1
    fi
    echo "Starting client, connecting to $PEER_IP..."
    ./tests/test_devmem_validation client $PEER_IP 0 0
fi

