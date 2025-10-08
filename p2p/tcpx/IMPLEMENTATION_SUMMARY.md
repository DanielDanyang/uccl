# 实施总结：每 GPU 4 个 TCPX 连接

## 🎯 核心理解

### 正确的架构
```
GCP A3-high 拓扑：
- 4 个 NICs: eth1, eth2, eth3, eth4
- 8 个 GPUs per node
- GPU → NIC 映射：
  * GPU 0,1 → eth1
  * GPU 2,3 → eth2
  * GPU 4,5 → eth3
  * GPU 6,7 → eth4

每个 GPU 只能用 1 个 NIC！
```

### TCPX 限制
```c
#define MAX_SOCKETS 8  // 每个 NIC 最多 8 个 sockets
```

**关键点**：
- 一个 NIC 上的所有 TCPX connections 共享这 8 个 socket 槽位
- 2 个 GPUs 共享 1 个 NIC
- 因此：每个 GPU 最多 4 个 connections

### 目标配置
```
每个 GPU：4 个 TCPX connections
每个 NIC：8 个 connections（来自 2 个 GPUs，4+4）

示例：
- GPU 0 → eth1，4 个 connections
- GPU 1 → eth1，4 个 connections
- 总共：eth1 上有 8 个 connections（达到 MAX_SOCKETS 上限）
```

### 关键概念澄清
1. **在 NCCL plugin 中**：1 channel ≈ 1 TCPX connection
2. **在我们的代码中**：
   - `UCCL_TCPX_NUM_CHANNELS=4` → 创建 4 个独立的 TCPX connections
   - 每个 connection 是一次 `tcpx_listen` + `tcpx_connect` 调用
   - **不使用** `NCCL_NSOCKS_PERTHREAD` 来让单个 connection 有多个 sockets
3. **NIC 选择**：
   - 由脚本通过 `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` 环境变量控制
   - `ChannelManager` 会尊重这个环境变量
   - 不需要在代码中硬编码 GPU → NIC 映射

---

## 📝 已完成的修改

### 修改 1：`run_p2p_fullmesh.sh`
**文件**：`p2p/tcpx/run_p2p_fullmesh.sh`

**改动**：
```bash
# 第 51 行
# 之前：CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-1}  # Single channel per GPU (working config)
# 之后：CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-4}  # 4 connections per GPU (pipeline parallelism)
```

**说明**：
- 将默认 channel 数从 1 改为 4
- 脚本已经有正确的 GPU → NIC 映射（`map_gpu_to_ifaces` 函数）
- 脚本会为每个 GPU 设置 `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME`

---

### 修改 2：`bench_p2p.sh`
**文件**：`p2p/tcpx/bench_p2p.sh`

**改动**：
1. 添加 `--channels=N` 选项（第 22-24 行）
2. 添加 `CHANNELS=4` 默认值（第 69 行）
3. 添加 `--channels=*` 参数解析（第 86 行）
4. 导出 `UCCL_TCPX_NUM_CHANNELS` 环境变量（第 158 行）
5. 更新日志输出（第 175-176 行）

**说明**：
- 添加了对多 channel 的支持
- 默认使用 4 个 connections
- 保留了 `NCCL_NSOCKS_PERTHREAD` 和 `NCCL_SOCKET_NTHREADS`，但标注为"不使用"

---

### 修改 3：`test_tcpx_perf_multi.cc`
**文件**：`p2p/tcpx/tests/test_tcpx_perf_multi.cc`

**改动**：
1. 移除了自动配置 `NCCL_NSOCKS_PERTHREAD` 的代码（第 181-193 行）
2. 更新了注释，说明新的架构（第 181-187 行）
3. 简化了配置日志输出（第 225-236 行）
4. 将默认 channel 数从 1 改为 4（第 226 行）

**说明**：
- 移除了之前错误的"每个 channel 多个 sockets"的逻辑
- 现在每个 channel 就是一个独立的 TCPX connection
- 日志更清晰地说明了配置

---

## 🧪 测试步骤

### 编译
```bash
cd /home/daniel/uccl/p2p/tcpx
make clean
make -j
```

### 测试 1：使用 `bench_p2p.sh`（推荐）

**Server 端（Node 0, GPU 0）**：
```bash
./bench_p2p.sh server 0 --ifaces=eth1 --channels=4
```

**Client 端（Node 1, GPU 0）**：
```bash
./bench_p2p.sh client <SERVER_IP> 0 --ifaces=eth1 --channels=4
```

### 测试 2：使用 `run_p2p_fullmesh.sh`（全部 8 个 GPUs）

**Server 端（Node 0）**：
```bash
./run_p2p_fullmesh.sh server
```

**Client 端（Node 1）**：
```bash
./run_p2p_fullmesh.sh client <SERVER_IP>
```

### 测试 3：手动运行（调试用）

**Server 端**：
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export UCCL_TCPX_NUM_CHANNELS=4
export UCCL_TCPX_PERF_SIZE=67108864
export UCCL_TCPX_PERF_ITERS=20

./tests/test_tcpx_perf_multi server 0
```

**Client 端**：
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
export UCCL_TCPX_NUM_CHANNELS=4
export UCCL_TCPX_PERF_SIZE=67108864
export UCCL_TCPX_PERF_ITERS=20

./tests/test_tcpx_perf_multi client <SERVER_IP> 0
```

---

## ✅ 验证清单

### 日志验证

1. **Channel 创建**：
   ```
   [ChannelManager] Channel 0 → netDev 0 (eth1, ...)
   [ChannelManager] Channel 1 → netDev 0 (eth1, ...)
   [ChannelManager] Channel 2 → netDev 0 (eth1, ...)
   [ChannelManager] Channel 3 → netDev 0 (eth1, ...)
   [ChannelManager] Created 4 channel(s) for GPU 0
   ```

2. **配置信息**：
   ```
   [PERF] TCPX Connection Configuration:
   [PERF]   GPU ID: 0
   [PERF]   Connections per GPU: 4
   [PERF]   Note: Each channel = 1 TCPX connection
   [PERF]   Note: 2 GPUs share 1 NIC → 8 connections per NIC
   ```

3. **Bootstrap 握手**：
   ```
   [Bootstrap] Sent 4 handles to client
   [Bootstrap] Received 4 handles from server
   ```

4. **连接建立**：
   ```
   [ChannelManager] All 4 channels listening successfully
   [ChannelManager] All 4 channels accepted successfully
   [ChannelManager] All 4 channels connected successfully
   ```

5. **Chunk 分配**（应该看到 channel_id 从 0-3 循环）：
   ```
   [DEBUG][SERVER] chunk=0 channel=0 ...
   [DEBUG][SERVER] chunk=1 channel=1 ...
   [DEBUG][SERVER] chunk=2 channel=2 ...
   [DEBUG][SERVER] chunk=3 channel=3 ...
   [DEBUG][SERVER] chunk=4 channel=0 ...  # 循环回到 0
   ```

### 性能验证

1. **单 GPU pair 带宽**：
   - 目标：接近 21.26 GB/s（单 NIC 理论上限）
   - 比单 connection 基线提升 3-4 倍

2. **多 GPU pairs 带宽**：
   - 如果 2 个 GPUs 同时使用同一个 NIC，总带宽不应超过 21.26 GB/s
   - 每个 GPU 应该获得约 10-11 GB/s

### 使用 TCPX TRACE 验证（可选）

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
```

查找：
- 每个 connection 的 socket 建立日志
- 应该看到 4 个独立的 connect/accept 过程
- 每个 connection 应该只有 1 个 socket（不是 8 个）

---

## 🚨 常见问题

### 问题 1：带宽没有提升

**可能原因**：
- Chunk size 太小（默认 512KB 应该足够）
- 窗口大小不合适（默认 server=16, client=12）
- 数据量太小（建议至少 64MB）

**调试**：
```bash
export UCCL_TCPX_PERF_SIZE=134217728  # 128MB
export UCCL_TCPX_CHUNK_BYTES=1048576  # 1MB chunks
```

### 问题 2：连接失败

**症状**：`bind() failed: Address already in use`

**原因**：端口冲突

**解决**：
```bash
export NCCL_GPUDIRECTTCPX_PORT_BEGIN=50000
export NCCL_GPUDIRECTTCPX_PORT_END=50100
```

### 问题 3：NIC 选择错误

**症状**：日志显示使用了错误的 NIC

**原因**：`NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` 没有正确设置

**解决**：
- 使用脚本（`bench_p2p.sh` 或 `run_p2p_fullmesh.sh`）
- 或手动设置：`export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1`

---

## 📊 预期结果

### 成功标准

1. ✅ 每个 GPU 创建 4 个 connections
2. ✅ 所有 4 个 connections 使用同一个 NIC
3. ✅ GPU 0,1 使用 eth1；GPU 2,3 使用 eth2；等等
4. ✅ Bootstrap 成功交换 4 个 handles
5. ✅ 所有 4 个 connections 成功建立
6. ✅ Chunks 正确地 round-robin 分配到 4 个 connections
7. ✅ 测试稳定完成，无 deadlock
8. ✅ 带宽比单 connection 基线提升 3-4 倍

### 性能目标

- **单 GPU pair**：~18-21 GB/s（接近单 NIC 上限）
- **2 GPUs 共享 1 NIC**：每个 GPU ~10-11 GB/s（总共 ~21 GB/s）
- **Full-mesh（8×8 GPUs）**：每个 NIC ~21 GB/s，总带宽 ~84 GB/s

---

## 🔄 下一步

1. **编译并测试**：验证 4 个 connections 是否正常工作
2. **性能测试**：对比 1 connection vs 4 connections 的带宽
3. **Full-mesh 测试**：验证多个 GPUs 同时运行时的行为
4. **调优**：如果需要，调整 chunk size、窗口大小等参数

