#!/bin/bash
# Phase 1 Test: 4 channels per GPU with round-robin NIC distribution
# This script tests the Phase 1 fix to verify:
# 1. All 8 GPUs complete accept without stalling
# 2. Channels are distributed across all 4 NICs
# 3. No single NIC is saturated

set -e

ROLE="${1:-}"
SERVER_IP="${2:-}"

if [[ -z "$ROLE" ]]; then
  echo "Usage: $0 <server|client> [server_ip]"
  echo ""
  echo "Phase 1 Test Configuration:"
  echo "  - 4 channels per GPU (reduced from 8)"
  echo "  - Round-robin NIC distribution"
  echo "  - Expected: All 4 NICs used evenly"
  exit 1
fi

if [[ "$ROLE" == "client" && -z "$SERVER_IP" ]]; then
  echo "Error: client mode requires server_ip"
  echo "Usage: $0 client <server_ip>"
  exit 1
fi

# Force 4 channels per GPU for Phase 1 testing
export UCCL_TCPX_NUM_CHANNELS=4

echo "=========================================="
echo "Phase 1 Test: Round-Robin NIC Distribution"
echo "=========================================="
echo "Role: $ROLE"
echo "Channels per GPU: $UCCL_TCPX_NUM_CHANNELS"
echo "Total channels: $((8 * UCCL_TCPX_NUM_CHANNELS))"
echo "Expected distribution: 8 channels per NIC"
echo "=========================================="
echo ""

# Run the single-process orchestrator
if [[ "$ROLE" == "server" ]]; then
  ./run_p2p_singleproc.sh server
else
  ./run_p2p_singleproc.sh client "$SERVER_IP"
fi

