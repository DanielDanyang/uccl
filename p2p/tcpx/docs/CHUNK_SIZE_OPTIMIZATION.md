# Chunk 大小优化 + TCPX 环境配置

**日期**: 2025-10-02  
**状态**: ✅ 已完成并编译成功

---

## 🎯 优化目标

### 优化 1: 增大 Chunk 大小

**问题**: 当前 chunk 大小 512 KB 太小，导致：
- 64 MB 数据被分成 128 个 chunks
- 每个 chunk 都有固定开销（syscall, kernel launch 等）
- 总开销 = 128 × 固定开销

**解决方案**: 将 chunk 大小从 **512 KB** 增加到 **2 MB**

**预期效果**:
- Chunk 数量: 128 → **32** (减少 4×)
- Server: 21 ms → **10-15 ms** (2× 提升)
- Client: 77 ms → **30-40 ms** (2× 提升)
- 带宽: 2.96 GB/s → **5-8 GB/s**

---

### 优化 2: 修复网卡配置

**问题**: `CTRL_DEV="eth1"` 错误，应该是 `eth0`

**原因**:
- eth0: Control network (控制网络)
- eth1-4: Data networks (数据网络)

**修复**: 将所有脚本中的 `CTRL_DEV` 改为 `eth0`

---

### 优化 3: 添加 TCPX 环境配置

**问题**: 缺少 NCCL+TCPX 的最佳实践配置

**解决方案**: 从 `run_nccl_test_tcpx.sh` 复制 TCPX 相关配置

**新增配置**:
- CPU bindings (TX/RX)
- Flow steering
- Chunk sizes
- Buffer sizes
- 等等

---

## 📝 详细修改

### 修改 1: 增大 Chunk 大小

**文件**: `tests/test_tcpx_perf.cc`

**位置**: 第 202-205 行

**修改前**:
```cpp
// Chunk 大小：默认 512KB
size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));
```

**修改后**:
```cpp
// Chunk 大小：默认 2MB
// 【优化】从 512KB 增加到 2MB，减少 chunk 数量，降低固定开销
size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 2 * 1024 * 1024));
```

**影响**:
- 64 MB 数据: 128 chunks → **32 chunks**
- 每个 chunk: 512 KB → **2 MB**

---

### 修改 2: 修复网卡配置

**文件**: 
- `bench_p2p.sh`
- `bench_p2p_sweep_server.sh`
- `bench_p2p_sweep_client.sh`

**修改前**:
```bash
CTRL_DEV="eth1"
```

**修改后**:
```bash
CTRL_DEV="eth0"  # Control network (eth0), data networks (eth1-4)
```

---

### 修改 3: 添加 TCPX 环境配置

**文件**: `bench_p2p.sh`

**位置**: 第 100-128 行

**新增配置**:

```bash
# Env for TCPX (adapted from run_nccl_test_tcpx.sh)
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="${IFACES}"
export NCCL_GPUDIRECTTCPX_CTRL_DEV="${CTRL_DEV}"
export NCCL_NSOCKS_PERTHREAD="${NSOCKS}"
export NCCL_SOCKET_NTHREADS="${NTHREADS}"

# TCPX-specific optimizations from NCCL test script
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608

# TCPX TX/RX CPU bindings (H100 specific, from GCP best practices)
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# TCPX flow steering and performance tuning
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_PROGRAM_CONNECT_TIMEOUT_MS=30000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# NCCL general settings
export NCCL_SOCKET_IFNAME=eth0  # Control network
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
```

---

## 🔍 新增配置详解

### 1. NCCL Chunk Sizes

```bash
export NCCL_DYNAMIC_CHUNK_SIZE=524288      # 512 KB
export NCCL_P2P_NET_CHUNKSIZE=524288       # 512 KB (网络传输)
export NCCL_P2P_PCI_CHUNKSIZE=524288       # 512 KB (PCIe 传输)
export NCCL_P2P_NVL_CHUNKSIZE=1048576      # 1 MB (NVLink 传输)
export NCCL_BUFFSIZE=8388608               # 8 MB (缓冲区大小)
```

**注意**: 这些是 NCCL 内部的 chunk 大小，与我们的 `UCCL_TCPX_CHUNK_BYTES` (2 MB) 不同。

---

### 2. CPU Bindings (H100 专用)

```bash
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

**作用**: 将 TCPX 的 TX/RX 线程绑定到特定的 CPU 核心，避免 NUMA 跨节点访问。

**H100 NUMA 拓扑**:
- eth1, eth2: NUMA node 0 (CPU 8-35, 112-139)
- eth3, eth4: NUMA node 1 (CPU 60-87, 164-191)

**TX vs RX**:
- TX (发送): CPU 8-21, 112-125 (eth1/2), 60-73, 164-177 (eth3/4)
- RX (接收): CPU 22-35, 126-139 (eth1/2), 74-87, 178-191 (eth3/4)

**为什么分开**: 避免 TX 和 RX 线程竞争同一个 CPU 核心。

---

### 3. Flow Steering

```bash
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
```

**作用**: Flow steering 编程等待时间（50 毫秒）

**什么是 Flow Steering**: 
- 将特定的网络流（flow）路由到特定的 RX 队列
- 需要 dp-manager 服务支持
- 可以提升多流并发性能

---

### 4. NCCL General Settings

```bash
export NCCL_SOCKET_IFNAME=eth0       # 控制网络（MPI, 协调等）
export NCCL_CROSS_NIC=0              # 禁用跨网卡通信（每个 GPU 绑定到特定网卡）
export NCCL_NET_GDR_LEVEL=PIX        # GPU Direct RDMA 级别（PIX = PCIe + NVLink）
export NCCL_P2P_PXN_LEVEL=0          # 禁用 PXN (PCIe Crossbar Network)
```

---

## ✅ 编译状态

```bash
make clean && make test_tcpx_perf -j4
```

**结果**: ✅ **编译成功！**

---

## 🧪 测试步骤

### 1. 运行测试

**Server 端 (10.65.74.150)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/server_2mb_chunk_$(date +%Y%m%d_%H%M%S).log
```

**Client 端 (10.64.113.77)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/client_2mb_chunk_$(date +%Y%m%d_%H%M%S).log
```

---

### 2. 验证 Chunk 数量

**检查日志**:
```bash
# 应该看到 32 个 chunks (而不是 128 个)
grep "chunk_idx=" logs/server_2mb_chunk_*.log | tail -5
```

**预期输出**:
```
[PERF][SERVER] chunk_idx=31 tag=XXX size=2097152 offset=XXX
```

**注意**: chunk_idx 从 0 开始，所以最后一个是 31 (总共 32 个)。

---

### 3. 验证性能

**查看性能**:
```bash
grep "Avg:" logs/server_2mb_chunk_*.log
grep "Avg:" logs/client_2mb_chunk_*.log
```

**预期结果**:
```
Server: Avg: 10-15 ms, BW: 5-8 GB/s
Client: Avg: 30-40 ms, BW: 2-4 GB/s
```

---

### 4. 验证网卡使用

**在测试运行时，监控网卡流量**:
```bash
# 安装 ifstat (如果没有)
sudo apt-get install ifstat

# 监控四个数据网卡
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

**预期**: 四个网卡都应该有流量（每个约 1-2 GB/s）

---

## 📊 性能对比

### 修改前（512 KB chunk）

| 端 | Chunk 数量 | 时间/迭代 | 带宽 |
|----|-----------|----------|------|
| **Server** | 128 | 21 ms | 2.96 GB/s |
| **Client** | 128 | 77 ms | 0.81 GB/s |

---

### 修改后（2 MB chunk，预期）

| 端 | Chunk 数量 | 时间/迭代 | 带宽 | 提升 |
|----|-----------|----------|------|------|
| **Server** | 32 | **10-15 ms** | **5-8 GB/s** | **2× 提升** |
| **Client** | 32 | **30-40 ms** | **2-4 GB/s** | **2× 提升** |

---

### 与 iperf3 对比

| 指标 | iperf3 | 当前 (512KB) | 预期 (2MB) | 目标 |
|------|--------|-------------|-----------|------|
| **单网卡** | 7.55 GB/s | - | - | - |
| **Server** | - | 2.96 GB/s (39%) | **5-8 GB/s (66-106%)** | 10 GB/s |
| **Client** | - | 0.81 GB/s (11%) | **2-4 GB/s (26-53%)** | 8 GB/s |

**注意**: 百分比是相对于单网卡 iperf3 带宽 (7.55 GB/s)。

---

## 🎯 下一步优化

### 如果达到 5-8 GB/s

**说明**: Chunk 大小优化成功！

**下一步**:
1. 验证四网卡是否都在使用（使用 `ifstat`）
2. 如果只有一个网卡在用，检查 TCPX 配置
3. 如果四网卡都在用，继续优化：
   - 增加 Client 端滑动窗口到 16
   - 批量接收/发送
   - 优化轮询策略

---

### 如果仍然很慢（< 3 GB/s）

**说明**: 还有其他瓶颈

**下一步**:
1. 检查 TCPX 日志，查找错误或警告
2. 检查 CPU bindings 是否生效
3. 检查网络配置（MTU, TCP 窗口等）
4. 使用 `nsys` 或 `nvprof` 分析性能瓶颈

---

## 📝 总结

### 修改内容

1. ✅ **Chunk 大小**: 512 KB → **2 MB** (4× 减少 chunk 数量)
2. ✅ **网卡配置**: `CTRL_DEV="eth1"` → `CTRL_DEV="eth0"`
3. ✅ **TCPX 环境配置**: 添加 20+ 个 NCCL+TCPX 最佳实践配置
4. ✅ **编译成功**: 无错误，无警告

### 预期效果

- **Server**: 21 ms → **10-15 ms** (2× 提升)
- **Client**: 77 ms → **30-40 ms** (2× 提升)
- **带宽**: 2.96 GB/s → **5-8 GB/s** (2× 提升)

### 下一步

1. ⏳ **立即测试** - 运行新的测试
2. ⏳ **验证性能** - 检查是否达到预期
3. ⏳ **验证网卡** - 使用 `ifstat` 检查四网卡是否都在用
4. ⏳ **继续优化** - 根据结果决定下一步

---

**状态**: ✅ 代码已修改并编译成功，等待测试验证

