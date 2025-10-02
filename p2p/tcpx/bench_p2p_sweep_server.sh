#!/usr/bin/env bash
set -euo pipefail

# Run a sweep of P2P sizes as server. Launches one test per size sequentially.
# This must be run on the server node. Start the matching client sweep on the other node.

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <gpu_id> [options]

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

Example:
  ./bench_p2p_sweep_server.sh 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
EOF
}

GPU_ID=${1:-}
[[ -z "${GPU_ID}" ]] && { usage; exit 1; }
shift || true

# Defaults
IFACES="eth1,eth2,eth3,eth4"
CTRL_DEV="eth0"  # Control network (eth0), data networks (eth1-4)
SIZES=""
ITERS="auto"
CHUNK=$((512*1024))
NSOCKS=4
NTHREADS=1
IMPL="kernel"
USE_UNIX=0

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

echo "=== P2P Sweep (Server) ==="
echo "GPU: ${GPU_ID} | ifaces=${IFACES} ctrl=${CTRL_DEV} nsocks=${NSOCKS} nthreads=${NTHREADS} impl=${IMPL}"
echo "Sizes: ${SIZES}"

for SZ in "${SIZE_ARR[@]}"; do
  # Determine iterations
  if [[ "${ITERS}" == "auto" ]]; then
    if (( SZ <= 1048576 )); then RUN_ITERS=50; else RUN_ITERS=20; fi
  else
    RUN_ITERS=${ITERS}
  fi
  echo "[SERVER] Running size=${SZ} bytes, iters=${RUN_ITERS} (chunk=${CHUNK})"
  if [[ ${USE_UNIX} -eq 1 ]]; then
    ./bench_p2p.sh server "${GPU_ID}" \
      --ifaces="${IFACES}" --ctrl="${CTRL_DEV}" \
      --size="${SZ}" --iters="${RUN_ITERS}" --chunk="${CHUNK}" \
      --nsocks="${NSOCKS}" --nthreads="${NTHREADS}" --impl="${IMPL}" \
      --unix-prefix=/tmp/uccl_perf
  else
    ./bench_p2p.sh server "${GPU_ID}" \
      --ifaces="${IFACES}" --ctrl="${CTRL_DEV}" \
      --size="${SZ}" --iters="${RUN_ITERS}" --chunk="${CHUNK}" \
      --nsocks="${NSOCKS}" --nthreads="${NTHREADS}" --impl="${IMPL}" \
      --no-unix
  fi
  echo "[SERVER] Done size=${SZ}"
  sleep 1
done

echo "[SERVER] Sweep complete."

