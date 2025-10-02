#!/bin/bash
# Test 3: GPU 直收 + 单网卡 + kernel 解包 (Server 端)

cd /home/daniel/uccl/p2p/tcpx

export UCCL_TCPX_HOST_RECV_DEBUG=0
export UCCL_TCPX_UNPACK_IMPL=kernel
export UCCL_TCPX_LAUNCH_DEBUG=1
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

echo "=========================================="
echo "Test 3: GPU Direct Recv + Single-NIC + Kernel Unpack"
echo "Server Mode"
echo "=========================================="
echo "UCCL_TCPX_HOST_RECV_DEBUG=$UCCL_TCPX_HOST_RECV_DEBUG"
echo "UCCL_TCPX_UNPACK_IMPL=$UCCL_TCPX_UNPACK_IMPL"
echo "NET_GPUDIRECTTCPX_SOCKET_IFNAME=$NET_GPUDIRECTTCPX_SOCKET_IFNAME"
echo "=========================================="

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test3_gpu_single_server.log

