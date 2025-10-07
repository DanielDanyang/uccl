#!/bin/bash
# Step 3 Test: Data transfer with bandwidth measurement
# This script tests the Step 3 data plane implementation

set -e

ROLE="${1:-}"
SERVER_IP="${2:-}"

if [[ -z "$ROLE" ]]; then
  echo "Usage: $0 <server|client> [server_ip]"
  echo ""
  echo "Step 3 Test Configuration:"
  echo "  - Data transfer enabled"
  echo "  - Round-robin channel selection"
  echo "  - Bandwidth measurement"
  echo "  - Default: 4 channels/GPU, 64MB test size, 20 iterations"
  exit 1
fi

if [[ "$ROLE" == "client" && -z "$SERVER_IP" ]]; then
  echo "Error: client mode requires server_ip"
  echo "Usage: $0 client <server_ip>"
  exit 1
fi

# Test configuration (can be overridden)
export UCCL_TCPX_NUM_CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-4}
export UCCL_TCPX_PERF_SIZE=${UCCL_TCPX_PERF_SIZE:-67108864}  # 64MB
export UCCL_TCPX_PERF_ITERS=${UCCL_TCPX_PERF_ITERS:-20}
export UCCL_TCPX_CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-524288}  # 512KB

echo "=========================================="
echo "Step 3 Test: Data Transfer + Bandwidth"
echo "=========================================="
echo "Role: $ROLE"
echo "Channels per GPU: $UCCL_TCPX_NUM_CHANNELS"
echo "Total channels: $((8 * UCCL_TCPX_NUM_CHANNELS))"
echo "Test size: $UCCL_TCPX_PERF_SIZE bytes ($((UCCL_TCPX_PERF_SIZE / 1024 / 1024)) MB)"
echo "Iterations: $UCCL_TCPX_PERF_ITERS"
echo "Chunk size: $UCCL_TCPX_CHUNK_BYTES bytes ($((UCCL_TCPX_CHUNK_BYTES / 1024)) KB)"
echo "=========================================="
echo ""

# Run the single-process orchestrator
if [[ "$ROLE" == "server" ]]; then
  ./run_p2p_singleproc.sh server
else
  ./run_p2p_singleproc.sh client "$SERVER_IP"
fi

