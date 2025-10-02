#!/usr/bin/env bash
set -euo pipefail

# Run a sweep of P2P sizes as client, collect steady-state BW into CSV and Markdown table.
# This must be run on the client node after starting the server sweep on the other node.

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <server_ip> <gpu_id> [options]

Options:
  --ifaces=LIST      NICs, comma-separated (default: eth1,eth2,eth3,eth4)
  --ctrl=DEV         Control NIC (default: eth1)
  --sizes=CSV        Comma-separated sizes in bytes (default: preset sweep 4KB..256MB)
  --iters=N          Iterations per size (default: auto: 50 for <=1MB else 20)
  --chunk=BYTES      Chunk bytes (default: 524288)
  --nsocks=N         NCCL_NSOCKS_PERTHREAD (default: 4)
  --nthreads=N       NCCL_SOCKET_NTHREADS (default: 1)
  --impl=kernel|d2d  Unpack implementation (default: kernel)
  --no-unix          Disable UNIX flow steering (default: on/disabled)
  --skip-warmup=N    Skip first N iter(s) for steady-state calc (default: 1)

Example:
  ./bench_p2p_sweep_client.sh 10.64.52.73 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
EOF
}

SERVER_IP=${1:-}
GPU_ID=${2:-}
[[ -z "${SERVER_IP}" || -z "${GPU_ID}" ]] && { usage; exit 1; }
shift 2 || true

# Defaults
IFACES="eth1,eth2,eth3,eth4"
CTRL_DEV="eth1"
SIZES=""
ITERS="auto"
CHUNK=$((512*1024))
NSOCKS=4
NTHREADS=1
IMPL="kernel"
USE_UNIX=0
SKIP_WARMUP=1

for arg in "$@"; do
  case "$arg" in
    --ifaces=*) IFACES="${arg#*=}" ;;
    --ctrl=*) CTRL_DEV="${arg#*=}" ;;
    --sizes=*) SIZES="${arg#*=}" ;;
    --iters=*) ITERS="${arg#*=}" ;;
    --chunk=*) CHUNK="${arg#*=}" ;;
    --nsocks=*) NSOCKS="${arg#*=}" ;;
    --nthreads=*) NTHREADS="${arg#*=}" ;;
    --impl=*) IMPL="${arg#*=}" ;;
    --no-unix) USE_UNIX=0 ;;
    --skip-warmup=*) SKIP_WARMUP="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; usage; exit 1;;
  esac
done

# Preset sizes: 4KB..256MB (bytes)
if [[ -z "${SIZES}" ]]; then
  SIZES=$(cat <<'CSV' | tr -d '\n'
4096,8192,16384,32768,65536,131072,262144,524288,\
1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456
CSV
)
fi

IFS=',' read -r -a SIZE_ARR <<< "${SIZES}"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
CSV_OUT="logs/p2p_sweep_${TS}.csv"
MD_OUT="logs/p2p_sweep_${TS}.md"

echo "size_bytes,size_mb,iters,chunk_bytes,nsocks,nthreads,ifaces,impl,steady_avg_ms,steady_bw_gbps" > "${CSV_OUT}"

# Helper to get most recent client log
latest_client_log() {
  ls -t logs/bench_client_*.log 2>/dev/null | head -n1 || true
}

for SZ in "${SIZE_ARR[@]}"; do
  if [[ "${ITERS}" == "auto" ]]; then
    if (( SZ <= 1048576 )); then RUN_ITERS=50; else RUN_ITERS=20; fi
  else
    RUN_ITERS=${ITERS}
  fi

  echo "[CLIENT] Running size=${SZ} bytes, iters=${RUN_ITERS} (chunk=${CHUNK})"
  BEFORE=$(latest_client_log)

  if [[ ${USE_UNIX} -eq 1 ]]; then
    ./bench_p2p.sh client "${SERVER_IP}" "${GPU_ID}" \
      --ifaces="${IFACES}" --ctrl="${CTRL_DEV}" \
      --size="${SZ}" --iters="${RUN_ITERS}" --chunk="${CHUNK}" \
      --nsocks="${NSOCKS}" --nthreads="${NTHREADS}" --impl="${IMPL}" \
      --unix-prefix=/tmp/uccl_perf --skip-warmup="${SKIP_WARMUP}" >/dev/null
  else
    ./bench_p2p.sh client "${SERVER_IP}" "${GPU_ID}" \
      --ifaces="${IFACES}" --ctrl="${CTRL_DEV}" \
      --size="${SZ}" --iters="${RUN_ITERS}" --chunk="${CHUNK}" \
      --nsocks="${NSOCKS}" --nthreads="${NTHREADS}" --impl="${IMPL}" \
      --no-unix --skip-warmup="${SKIP_WARMUP}" >/dev/null
  fi

  sleep 1
  AFTER=$(latest_client_log)
  LOG_FILE="${AFTER}"
  if [[ -z "${LOG_FILE}" || "${LOG_FILE}" == "${BEFORE}" ]]; then
    echo "[WARN] Could not determine new client log for size=${SZ}" >&2
    continue
  fi

  # Extract steady-state line
  SS_LINE=$(grep -E "\[STEADY-STATE\] Avg time: .* ms, BW: .* GB/s" "${LOG_FILE}" || true)
  AVG_MS=""; BW_GBPS=""
  if [[ -n "${SS_LINE}" ]]; then
    AVG_MS=$(echo "${SS_LINE}" | sed -E 's/.*Avg time: ([0-9.]+) ms.*/\1/')
    BW_GBPS=$(echo "${SS_LINE}" | sed -E 's/.*BW: ([0-9.]+) GB\/s.*/\1/')
  else
    echo "[WARN] No steady-state line found in ${LOG_FILE} for size=${SZ}" >&2
  fi

  SIZE_MB=$(python3 -c "print(int(${SZ})/(1024**2))")

  echo "${SZ},${SIZE_MB},${RUN_ITERS},${CHUNK},${NSOCKS},${NTHREADS},\"${IFACES}\",${IMPL},${AVG_MS},${BW_GBPS}" >> "${CSV_OUT}"
  echo "[CLIENT] Done size=${SZ}: ${BW_GBPS} GB/s (avg ${AVG_MS} ms)"
done

# Generate Markdown table
{
  echo "# TCPX P2P Performance Sweep (Client)"
  echo ""
  echo "- Ifaces: ${IFACES} (ctrl=${CTRL_DEV}), nsocks=${NSOCKS}, nthreads=${NTHREADS}, impl=${IMPL}, chunk=${CHUNK}"
  echo "- Skip warmup: ${SKIP_WARMUP}"
  echo ""
  echo "| Size (Bytes) | Size (MB) | Iters | Steady Avg (ms) | BW (GB/s) |"
  echo "| ---: | ---: | ---: | ---: | ---: |"
  awk -F',' 'NR>1 {printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, ($9==""?"-":$9), ($10==""?"-":$10)}' "${CSV_OUT}"
} > "${MD_OUT}"

echo "[RESULT] CSV: ${CSV_OUT}"
echo "[RESULT] Markdown: ${MD_OUT}"

