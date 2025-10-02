# TCPX Performance Test 运行指南

## 测试目标

通过逐步排查，定位 "rx no cmsg" 问题的根源：
1. 验证 Host 接收模式（已成功）
2. 验证 GPU 直收模式在不同配置下的表现
3. 确认发送端使用 CUDA 指针
4. 定位是否与多 NIC、chunk 大小相关

---

## 测试环境准备

### 节点信息
- **Node 1 (Server)**: 10.64.52.73
- **Node 2 (Client)**: 10.64.113.74
- **GPU**: 每节点 H100 × 8
- **网络**: eth1, eth2, eth3, eth4 (4×100GbE 或类似)

### 环境变量说明

| 环境变量 | 作用 | 可选值 |
|---------|------|--------|
| `UCCL_TCPX_HOST_RECV_DEBUG` | 服务端使用 Host 内存接收（绕过 devmem-tcp） | `0`=GPU直收, `1`=Host接收 |
| `UCCL_TCPX_UNPACK_IMPL` | GPU 模式下的解包方式 | `kernel`, `d2d`, `host` |
| `UCCL_TCPX_DEBUG` | 打印 wrapper 层详细日志 | `0`=关闭, `1`=开启（默认已开启）|
| `UCCL_TCPX_LAUNCH_DEBUG` | 打印 GPU kernel 详细日志 | `0`=关闭, `1`=开启 |
| `NET_GPUDIRECTTCPX_SOCKET_IFNAME` | 指定使用的网络接口 | `eth1,eth2,eth3,eth4` 或单个如 `eth1` |
| `NCCL_GPUDIRECTTCPX_RECV_SYNC` | 同步接收模式 | `1`=开启（推荐）|

---

## 测试步骤

### 第一轮：Host 接收模式（已验证通过）

**目的**: 验证网络通路、应用逻辑、分片机制正常

#### Server 端 (Node 1: 10.64.52.73)
```bash
cd /home/daniel/uccl/p2p/tcpx
export UCCL_TCPX_HOST_RECV_DEBUG=1
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test1_host_server.log
```

#### Client 端 (Node 2: 10.64.113.74)
```bash
cd /home/daniel/uccl/p2p/tcpx
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf client 10.64.52.73 0 2>&1 | tee logs/test1_host_client.log
```

**预期结果**: ✅ 已通过，带宽 ~7.75 GB/s（Host staging 水平）

---

### 第二轮：GPU 直收 + 多网卡 + kernel 解包

**目的**: 复现 "rx no cmsg" 问题

#### Server 端 (Node 1)
```bash
cd /home/daniel/uccl/p2p/tcpx
export UCCL_TCPX_HOST_RECV_DEBUG=0
export UCCL_TCPX_UNPACK_IMPL=kernel
export UCCL_TCPX_LAUNCH_DEBUG=1
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test2_gpu_multi_server.log
```

#### Client 端 (Node 2)
```bash
cd /home/daniel/uccl/p2p/tcpx
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf client 10.64.52.73 0 2>&1 | tee logs/test2_gpu_multi_client.log
```

**预期结果**: ❌ 可能出现 "rx no cmsg" 并中断

---

### 第三轮：GPU 直收 + 单网卡 + kernel 解包

**目的**: 排除多 NIC flow steering 问题

#### Server 端 (Node 1)
```bash
cd /home/daniel/uccl/p2p/tcpx
export UCCL_TCPX_HOST_RECV_DEBUG=0
export UCCL_TCPX_UNPACK_IMPL=kernel
export UCCL_TCPX_LAUNCH_DEBUG=1
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test3_gpu_single_server.log
```

#### Client 端 (Node 2)
```bash
cd /home/daniel/uccl/p2p/tcpx
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf client 10.64.52.73 0 2>&1 | tee logs/test3_gpu_single_client.log
```

**预期结果**: 观察是否仍出现 "rx no cmsg"

---

### 第四轮：GPU 直收 + 单网卡 + d2d 解包

**目的**: 排除 kernel 解包问题

#### Server 端 (Node 1)
```bash
cd /home/daniel/uccl/p2p/tcpx
export UCCL_TCPX_HOST_RECV_DEBUG=0
export UCCL_TCPX_UNPACK_IMPL=d2d
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf server 0 2>&1 | tee logs/test4_gpu_d2d_server.log
```

#### Client 端 (Node 2)
```bash
cd /home/daniel/uccl/p2p/tcpx
export NET_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1

./tests/test_tcpx_perf client 10.64.52.73 0 2>&1 | tee logs/test4_gpu_d2d_client.log
```

**预期结果**: 观察是否仍出现 "rx no cmsg"

---

## 日志收集

每轮测试后，请将以下文件发给我：
- `logs/testX_*_server.log`
- `logs/testX_*_client.log`

### 关键日志点

#### Server 端关键信息
```
[PERF][SERVER] Registering recv buffer: ptr=... type=NCCL_PTR_CUDA/NCCL_PTR_HOST
[PERF][SERVER] chunk_idx=... tag=... size=... offset=...
[PERF][SERVER] frag_count=...
fatal, ... rx no cmsg  <-- 如果出现这个，说明 devmem-tcp 路径失败
```

#### Client 端关键信息
```
[PERF][CLIENT] Registering send buffer: ptr=... type=NCCL_PTR_CUDA
[PERF][CLIENT] chunk_idx=... tag=... size=... offset=...
Connection reset by peer  <-- 如果出现，说明服务端已断开
```

---

## 快速诊断表

| 现象 | 可能原因 | 下一步 |
|------|---------|--------|
| Host 模式成功，GPU 模式 "rx no cmsg" | devmem-tcp 路径未建立 | 检查内核版本、devmem-tcp 配置 |
| 多网卡失败，单网卡成功 | flow steering 或 socket 配对问题 | 检查插件日志中的 "through cuda/host" 配对 |
| 所有 GPU 模式都失败 | 发送端未使用 CUDA 指针，或内核不支持 | 检查 reg_mr 日志，确认 ptr_type |
| kernel 失败，d2d 成功 | GPU 解包内核问题 | 调整 kernel 参数或使用 d2d |

---

## 创建日志目录

```bash
cd /home/daniel/uccl/p2p/tcpx
mkdir -p logs
```

---

## 注意事项

1. **每次测试前确保上一次的进程已完全退出**
   ```bash
   pkill -9 test_tcpx_perf
   ```

2. **确认环境变量生效**
   ```bash
   echo $UCCL_TCPX_HOST_RECV_DEBUG
   echo $NET_GPUDIRECTTCPX_SOCKET_IFNAME
   ```

3. **先启动 Server，再启动 Client**
   - Server 会监听等待连接
   - Client 启动后会主动连接 Server

4. **测试完成后检查带宽**
   - Host 模式: ~5-15 GB/s 正常
   - GPU 直收模式: 应显著高于 Host 模式，理想情况接近 40-50 GB/s（4×100GbE）

---

## 下一步分析

根据测试结果，我会：
1. 如果单网卡成功，多网卡失败 → 分析 socket 配对与 flow steering
2. 如果所有 GPU 模式都失败 → 在插件层添加 cmsg 详细日志
3. 如果某些配置成功 → 优化参数（chunk size、窗口深度等）提升性能

