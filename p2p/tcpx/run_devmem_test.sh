#!/bin/bash
# Devmem validation test runner

ROLE=${1:-server}
PEER_IP=${2:-}

export NCCL_GPUDIRECTTCPX_PLUGIN_PATH=/usr/local/tcpx/lib64/libnccl-net.so
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export UCCL_TCPX_NUM_CHANNELS=4

if [ "$ROLE" = "server" ]; then
    ./tests/test_devmem_validation server 0 0
else
    if [ -z "$PEER_IP" ]; then
        echo "Usage: $0 client <peer_ip>"
        exit 1
    fi
    ./tests/test_devmem_validation client $PEER_IP 0
fi
