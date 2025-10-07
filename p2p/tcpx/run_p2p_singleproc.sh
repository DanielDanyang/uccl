#!/usr/bin/env bash
# Single-process P2P launcher
# Purpose: Launch 1 process per node managing all 8 GPUs
# This enables multi-NIC usage per GPU (no devmem conflicts)

set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") server
  $(basename "$0") client <server_ip>

Description:
  Launches a SINGLE process per node that manages all 8 GPUs.
  This architecture enables:
  - Multi-channel per GPU (e.g., 8 channels/GPU)
  - Multi-NIC per GPU (all 4 NICs available to each GPU)
  - No devmem conflicts (single process shares NICs)

Environment overrides:
  UCCL_TCPX_BOOTSTRAP_PORT_BASE  Base port (default: 20000)
  UCCL_TCPX_NUM_CHANNELS         Channels per GPU (default: 8)
  UCCL_TCPX_PERF_SIZE            Bytes per iteration (default: 67108864)
  UCCL_TCPX_PERF_ITERS           Iterations (default: 20)
  UCCL_TCPX_CHUNK_BYTES          Chunk size (default: 524288)
  LOG_DIR                        Output log directory (default: p2p/tcpx/logs)
USAGE
}

ROLE=${1:-}
if [[ -z "${ROLE}" ]]; then
  usage; exit 1;
fi
shift || true

case "${ROLE}" in
  server)
    ;;
  client)
    SERVER_IP=${1:-}
    [[ -z "${SERVER_IP}" ]] && { echo "[ERROR] Missing <server_ip>" >&2; usage; exit 1; }
    shift || true
    ;;
  *)
    usage; exit 1;
    ;;
esac

# Defaults
BOOTSTRAP_BASE=${UCCL_TCPX_BOOTSTRAP_PORT_BASE:-20000}
CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-8}  # 8 channels per GPU
PERF_SIZE=${UCCL_TCPX_PERF_SIZE:-67108864}
PERF_ITERS=${UCCL_TCPX_PERF_ITERS:-20}
CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-524288}
LOG_DIR=${LOG_DIR:-"$(dirname "$0")/logs"}
mkdir -p "${LOG_DIR}"

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/singleproc_${ROLE}_${timestamp}.log"

# Shared environment (same as multi-process version)
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/tcpx/lib64:${LD_LIBRARY_PATH:-}"

# CRITICAL: Advertise ALL 4 NICs to the single process
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"

# Control plane
export NCCL_GPUDIRECTTCPX_CTRL_DEV="eth0"
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"

# NCCL settings
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608

# Thread affinity (from NCCL diagnostics)
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# TCPX tuning
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# Network settings
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0

# Algorithm
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8

# Debug
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ENV}

# UCCL settings
export UCCL_TCPX_BOOTSTRAP_PORT_BASE="${BOOTSTRAP_BASE}"
export UCCL_TCPX_NUM_CHANNELS="${CHANNELS}"
export UCCL_TCPX_PERF_SIZE="${PERF_SIZE}"
export UCCL_TCPX_PERF_ITERS="${PERF_ITERS}"
export UCCL_TCPX_CHUNK_BYTES="${CHUNK_BYTES}"

echo "=== Single-Process P2P Launcher ==="
echo "Role: ${ROLE}"
echo "Channels per GPU: ${CHANNELS}"
echo "NICs: eth1,eth2,eth3,eth4 (all available to all GPUs)"
echo "Bootstrap port base: ${BOOTSTRAP_BASE}"
echo "Log: ${log_file}"
echo "===================================="
echo ""

# Launch single process
if [[ "${ROLE}" == "server" ]]; then
  echo "[INFO] Launching server (single process, 8 GPUs)..."
  ./tests/test_tcpx_perf_orchestrator server &>"${log_file}" &
else
  echo "[INFO] Launching client (single process, 8 GPUs)..."
  ./tests/test_tcpx_perf_orchestrator client "${SERVER_IP}" &>"${log_file}" &
fi

# Wait for completion
wait

echo "[INFO] Process completed. Log: ${log_file}"

