#!/usr/bin/env bash
# Test different TCPX unpack implementations to identify performance bottleneck
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") server [gpu_id]
  $(basename "$0") client <server_ip> [gpu_id]

Description:
  Runs TCPX P2P tests with different unpack implementations (kernel, d2d, host)
  to identify which implementation provides the best performance.

  Tests are run sequentially with the same configuration, only changing
  UCCL_TCPX_UNPACK_IMPL. Results are logged to separate files for comparison.

Environment overrides:
  UCCL_TCPX_NUM_CHANNELS         Connections per GPU (default: 2)
  UCCL_TCPX_PERF_SIZE            Bytes per iteration (default: 67108864)
  UCCL_TCPX_PERF_ITERS           Iterations (default: 20)
  UCCL_TCPX_CHUNK_BYTES          Chunk size (default: 524288)

Examples:
  # Test all unpack modes on GPU 0
  ./test_unpack_modes.sh server 0
  ./test_unpack_modes.sh client <SERVER_IP> 0

  # Test with larger message size
  UCCL_TCPX_PERF_SIZE=536870912 ./test_unpack_modes.sh server 0
USAGE
}

ROLE=${1:-}
if [[ -z "${ROLE}" ]]; then
  usage; exit 1;
fi
shift || true

SERVER_IP=""
GPU_ID=""

case "${ROLE}" in
  server)
    GPU_ID=${1:-0}
    [[ -n "${GPU_ID}" ]] && shift || true
    ;;
  client)
    SERVER_IP=${1:-}
    [[ -z "${SERVER_IP}" ]] && { echo "[ERROR] Missing <server_ip>" >&2; usage; exit 1; }
    shift || true
    GPU_ID=${1:-0}
    [[ -n "${GPU_ID}" ]] && shift || true
    ;;
  *)
    usage; exit 1;
    ;;
esac

# Configuration
CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-2}
PERF_SIZE=${UCCL_TCPX_PERF_SIZE:-67108864}
PERF_ITERS=${UCCL_TCPX_PERF_ITERS:-20}
CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-524288}
LOG_DIR="$(dirname "$0")/logs/unpack_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

UNPACK_MODES=("kernel" "d2d" "host")

echo "========================================="
echo "TCPX Unpack Implementation Test"
echo "========================================="
echo "Role:       ${ROLE}"
echo "GPU:        ${GPU_ID}"
[[ -n "${SERVER_IP}" ]] && echo "Server IP:  ${SERVER_IP}"
echo "Channels:   ${CHANNELS}"
echo "Size:       ${PERF_SIZE} bytes"
echo "Iterations: ${PERF_ITERS}"
echo "Chunk:      ${CHUNK_BYTES} bytes"
echo "Log dir:    ${LOG_DIR}"
echo "========================================="
echo ""

run_test() {
  local mode=$1
  local log_file="${LOG_DIR}/${ROLE}_gpu${GPU_ID}_${mode}.log"
  
  echo "[$(date +%H:%M:%S)] Testing unpack mode: ${mode}"
  
  export UCCL_TCPX_UNPACK_IMPL="${mode}"
  export UCCL_TCPX_NUM_CHANNELS="${CHANNELS}"
  export UCCL_TCPX_PERF_SIZE="${PERF_SIZE}"
  export UCCL_TCPX_PERF_ITERS="${PERF_ITERS}"
  export UCCL_TCPX_CHUNK_BYTES="${CHUNK_BYTES}"
  
  if [[ "${ROLE}" == "server" ]]; then
    ./run_p2p_fullmesh.sh server "${GPU_ID}" &>"${log_file}"
  else
    ./run_p2p_fullmesh.sh client "${SERVER_IP}" "${GPU_ID}" &>"${log_file}"
  fi
  
  # Extract performance summary
  local avg_line=$(grep "Avg:" "${log_file}" 2>/dev/null || echo "N/A")
  echo "  Result: ${avg_line}"
  echo ""
}

# Run tests for each unpack mode
for mode in "${UNPACK_MODES[@]}"; do
  run_test "${mode}"
  
  # Wait a bit between tests to ensure clean state
  if [[ "${mode}" != "${UNPACK_MODES[-1]}" ]]; then
    echo "[$(date +%H:%M:%S)] Waiting 5 seconds before next test..."
    sleep 5
  fi
done

echo "========================================="
echo "All tests completed!"
echo "========================================="
echo ""
echo "Performance Summary:"
echo "-------------------"

for mode in "${UNPACK_MODES[@]}"; do
  log_file="${LOG_DIR}/${ROLE}_gpu${GPU_ID}_${mode}.log"
  avg_line=$(grep "Avg:" "${log_file}" 2>/dev/null || echo "N/A")
  printf "%-10s: %s\n" "${mode}" "${avg_line}"
done

echo ""
echo "Detailed logs in: ${LOG_DIR}"
echo ""
echo "To compare results:"
echo "  grep 'Avg:' ${LOG_DIR}/*.log"
echo ""
echo "To view specific log:"
echo "  less ${LOG_DIR}/${ROLE}_gpu${GPU_ID}_kernel.log"
echo ""

