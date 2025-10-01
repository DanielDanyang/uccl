#!/bin/bash
# TCPX Performance Test Runner
# Usage: ./run_perf_test.sh <server|client> <server_ip> <gpu_id>

set -e

MODE=$1
SERVER_IP=$2
GPU_ID=${3:-0}

if [ -z "$MODE" ]; then
    echo "Usage: $0 <server|client> <server_ip> <gpu_id>"
    echo ""
    echo "Examples:"
    echo "  Server (node 0, GPU 0): $0 server localhost 0"
    echo "  Client (node 1, GPU 0): $0 client 10.64.177.42 0"
    exit 1
fi

# Set environment variables
export UCCL_TCPX_UNPACK_IMPL=${UCCL_TCPX_UNPACK_IMPL:-kernel}
export UCCL_TCPX_WARMUP_ITERS=${UCCL_TCPX_WARMUP_ITERS:-5}
export UCCL_TCPX_BENCH_ITERS=${UCCL_TCPX_BENCH_ITERS:-100}

# TCPX environment (from your existing setup)
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_MIN_ZCOPY_SIZE=4096
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4

echo "=== TCPX Performance Test ==="
echo "Mode: $MODE"
echo "GPU: $GPU_ID"
echo "Unpack implementation: $UCCL_TCPX_UNPACK_IMPL"
echo "Warmup iterations: $UCCL_TCPX_WARMUP_ITERS"
echo "Benchmark iterations: $UCCL_TCPX_BENCH_ITERS"
echo ""

if [ "$MODE" == "server" ]; then
    ./tests/test_tcpx_perf server $GPU_ID
elif [ "$MODE" == "client" ]; then
    if [ -z "$SERVER_IP" ]; then
        echo "Error: server_ip required for client mode"
        exit 1
    fi
    ./tests/test_tcpx_perf client $SERVER_IP $GPU_ID
else
    echo "Error: mode must be 'server' or 'client'"
    exit 1
fi

