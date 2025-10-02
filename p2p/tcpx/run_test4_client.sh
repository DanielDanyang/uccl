#!/bin/bash
# Test 4: GPU 直收 + 单网卡 + d2d 解包 (Client 端)

cd /home/daniel/uccl/p2p/tcpx

export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

echo "=========================================="
echo "Test 4: GPU Direct Recv + Single-NIC + D2D Unpack"
echo "Client Mode"
echo "=========================================="
echo "NET_GPUDIRECTTCPX_SOCKET_IFNAME=$NET_GPUDIRECTTCPX_SOCKET_IFNAME"
echo "Target Server: 10.64.52.73"
echo "=========================================="

./tests/test_tcpx_perf client 10.64.52.73 0 2>&1 | tee logs/test4_gpu_d2d_client.log

