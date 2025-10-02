#!/bin/bash
# Test 4: GPU 直收 + 单网卡 + d2d 解包 (Server 端)

cd /home/daniel/uccl/p2p/tcpx

export UCCL_TCPX_HOST_RECV_DEBUG=0
export UCCL_TCPX_UNPACK_IMPL=d2d
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

echo "=========================================="
echo "Test 4: GPU Direct Recv + Single-NIC + D2D Unpack"
echo "Server Mode"
echo "=========================================="
echo "UCCL_TCPX_HOST_RECV_DEBUG=$UCCL_TCPX_HOST_RECV_DEBUG"
echo "UCCL_TCPX_UNPACK_IMPL=$UCCL_TCPX_UNPACK_IMPL"
echo "NET_GPUDIRECTTCPX_SOCKET_IFNAME=$NET_GPUDIRECTTCPX_SOCKET_IFNAME"
echo "=========================================="

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test4_gpu_d2d_server.log

