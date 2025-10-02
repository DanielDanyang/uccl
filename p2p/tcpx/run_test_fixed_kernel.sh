#!/bin/bash
# Test fixed kernel performance (async launch)

# Server IP
SERVER_IP="10.64.52.73"

# Test configuration
# Flow-steering via unix client requires a local dp-manager. Disable to avoid connect errors.
# export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=/tmp/uccl_perf
export NCCL_GPUDIRECTTCPX_PROGRAM_CONNECT_TIMEOUT_MS=30000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1
# Minimize complexity during debugging
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=1

# Unpack mode: kernel (fixed async version)
export UCCL_TCPX_UNPACK_IMPL=kernel
export UCCL_TCPX_HOST_RECV_DEBUG=0

# Optional debug
# export UCCL_TCPX_LAUNCH_DEBUG=1

if [ "$1" == "server" ]; then
  GPU_ID=${2:-0}  # Default to GPU 0 if not specified
  echo "=== Running FIXED Kernel Test (Server) ==="
  echo "Mode: GPU receive + Async kernel launch"
  echo "Expected: ~8-10ms per iteration, 0.4-0.5 GB/s"
  echo "GPU: $GPU_ID"
  echo ""
  ./tests/test_tcpx_perf server $GPU_ID | tee logs/test_fixed_kernel_server.log
elif [ "$1" == "client" ]; then
  GPU_ID=${2:-0}  # Default to GPU 0 if not specified
  echo "=== Running FIXED Kernel Test (Client) ==="
  echo "Connecting to: $SERVER_IP"
  echo "GPU: $GPU_ID"
  echo ""
  ./tests/test_tcpx_perf client $SERVER_IP $GPU_ID | tee logs/test_fixed_kernel_client.log
else
  echo "Usage: $0 [server|client] [gpu_id]"
  echo ""
  echo "On Node 1 (10.64.52.73):"
  echo "  ./run_test_fixed_kernel.sh server [gpu_id]"
  echo ""
  echo "On Node 2 (10.64.113.74):"
  echo "  ./run_test_fixed_kernel.sh client [gpu_id]"
  echo ""
  echo "Default GPU ID is 0 if not specified"
  exit 1
fi

