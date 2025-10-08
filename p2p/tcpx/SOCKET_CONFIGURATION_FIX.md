# Socket 配置修复：为什么 4 个 Channels 没有提升带宽

## 🐛 问题现象

- **1 个 channel**：2.6 GB/s
- **4 个 channels**：2.8 GB/s
- **预期**：应该有明显提升
- **实际**：几乎没有提升

---

## 🔍 根本原因

### 错误的理解（我之前的分析）

我之前以为是**串行发送**导致的，但这是错误的！

实际代码：
```cpp
while (offset < test_size) {
  int channel_id = global_chunk_idx % num_channels;  // Round-robin
  
  // 只有当这个 channel 满了才阻塞等待这个 channel
  while (win.inflight_recvs.size() >= MAX_INFLIGHT_PER_CHANNEL) {
    process_completed_chunk(channel_id, ch, win, /*blocking=*/true);
  }
  
  // Post irecv
  tcpx_irecv(...);
  
  // Opportunistic drain 其他 channels
  for (int other = 0; other < num_channels; ++other) {
    process_completed_chunk(other, ...);
  }
}
```

**这个逻辑是正确的**：
- ✅ Round-robin 分配 chunks
- ✅ 只阻塞等待满的 channel
- ✅ 其他 channels 的请求仍在网络中传输
- ✅ 有 opportunistic drain

---

### 真正的原因：Socket 数量不足！

#### NCCL/TCPX 的 Socket 配置

```
总 socket 数 = NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
```

**之前的配置**：
```bash
UCCL_TCPX_NUM_CHANNELS=4
NCCL_NSOCKS_PERTHREAD=1
NCCL_SOCKET_NTHREADS=1

结果：
- 4 个 TCPX comms per GPU
- 每个 comm: 1 × 1 = 1 socket
- 总共：4 sockets per GPU
```

**问题**：
- 每个 comm 只有 **1 个 socket**
- 虽然有 4 个 comms，但每个 comm 的带宽受限于单个 socket
- 单个 socket 的带宽：~2.6 GB/s
- 4 个 sockets 理论上应该有 ~10 GB/s，但实际上每个 comm 独立工作，无法聚合带宽

---

## 💡 正确的配置

### 目标

- 每个 GPU：4 sockets
- 2 个 GPUs 共享 1 个 NIC：8 sockets（刚好达到 MAX_SOCKETS=8）

### 方案 A：2 channels × 2 sockets（推荐）✅

```bash
export UCCL_TCPX_NUM_CHANNELS=2
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1

结果：
- 2 个 TCPX comms per GPU
- 每个 comm: 2 × 1 = 2 sockets
- 总共：4 sockets per GPU
- 2 GPUs 共享 1 NIC：8 sockets per NIC ✅
```

**优点**：
- ✅ 2 个 channels 提供更好的并行性
- ✅ 每个 channel 2 个 sockets，足够的带宽
- ✅ 总共 4 sockets per GPU，2 GPUs = 8 sockets per NIC

**预期带宽**：
- 每个 socket：~2.5 GB/s
- 每个 channel（2 sockets）：~5 GB/s
- 每个 GPU（2 channels）：~10 GB/s

---

### 方案 B：1 channel × 4 sockets

```bash
export UCCL_TCPX_NUM_CHANNELS=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1

结果：
- 1 个 TCPX comm per GPU
- 每个 comm: 4 × 1 = 4 sockets
- 总共：4 sockets per GPU
- 2 GPUs 共享 1 NIC：8 sockets per NIC ✅
```

**优点**：
- ✅ 简单，只有 1 个 channel
- ✅ 4 个 sockets 提供足够的带宽

**缺点**：
- ❌ 只有 1 个 channel，可能并行性不如方案 A

**预期带宽**：
- 每个 socket：~2.5 GB/s
- 每个 GPU（4 sockets）：~10 GB/s

---

### 方案 C：4 channels × 1 socket（之前的配置）❌

```bash
export UCCL_TCPX_NUM_CHANNELS=4
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=1

结果：
- 4 个 TCPX comms per GPU
- 每个 comm: 1 × 1 = 1 socket
- 总共：4 sockets per GPU
```

**问题**：
- ❌ 每个 comm 只有 1 个 socket，带宽受限
- ❌ 4 个 comms 无法聚合带宽（每个 comm 独立）
- ❌ 实际带宽：~2.8 GB/s（几乎没有提升）

---

## 🔧 已实施的修复

### 修改的文件

#### 1. `p2p/tcpx/run_p2p_fullmesh.sh`

**修改**：
```bash
# 之前
CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-4}
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=1

# 之后
CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-2}
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
```

#### 2. `p2p/tcpx/README.md`

**修改**：
```markdown
# 之前
- Each GPU: 4 TCPX connections (UCCL_TCPX_NUM_CHANNELS=4)
- Each connection: 1 socket (NCCL_NSOCKS_PERTHREAD=1)

# 之后
- Each GPU: 2 channels × 2 sockets = 4 sockets total
- (UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=2)
```

#### 3. `p2p/tcpx/tests/test_tcpx_perf_multi.cc`

**修改**：
- 默认 `num_channels` 从 4 改为 2
- 添加 socket 配置的详细输出
- 更新注释说明推荐配置

---

## 🧪 测试命令

```bash
cd /home/daniel/uccl/p2p/tcpx

# Server (Node 0, GPU 0)
./run_p2p_fullmesh.sh server 0

# Client (Node 1, GPU 0)
./run_p2p_fullmesh.sh client <SERVER_IP> 0
```

---

## 📊 预期效果

### 修复前（4 channels × 1 socket）
```
[PERF] TCPX Connection Configuration:
[PERF]   Channels per GPU: 4
[PERF]   Sockets per channel: 1
[PERF]   Total sockets per GPU: 4
[PERF] Avg (20 iter): 22.892 ms, BW: 2.73 GB/s  ← 几乎没有提升
```

### 修复后（2 channels × 2 sockets）
```
[PERF] TCPX Connection Configuration:
[PERF]   Channels per GPU: 2
[PERF]   Sockets per channel: 2
[PERF]   Total sockets per GPU: 4
[PERF] Avg (20 iter): ~6.5 ms, BW: ~10 GB/s  ← 预期 4 倍提升
```

---

## 📝 验证清单

- [ ] 看到 `Sockets per channel: 2`
- [ ] 看到 `Total sockets per GPU: 4`
- [ ] 所有 20 个 iterations 完成
- [ ] 带宽提升到 ~10 GB/s（接近单 NIC 理论上限 ~12.5 GB/s）

---

## 🎓 关键学习点

### 1. TCPX 的 Socket 配置

```
每个 TCPX comm 的 socket 数 = NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
```

- 如果只有 1 个 socket per comm，带宽受限于单个 socket
- 需要多个 sockets per comm 才能聚合带宽

### 2. Channels vs Sockets

- **Channels（comms）**：提供并行性（多个独立的通信流）
- **Sockets per channel**：提供带宽（每个通信流的吞吐量）
- **最佳配置**：平衡 channels 和 sockets

### 3. MAX_SOCKETS 限制

- TCPX 插件：MAX_SOCKETS=8 per NIC
- 2 个 GPUs 共享 1 个 NIC：每个 GPU 最多 4 sockets
- 配置时需要考虑这个限制

### 4. 为什么之前的配置没有提升？

- 4 个 channels × 1 socket = 4 个独立的单 socket 通信流
- 每个通信流的带宽：~2.6 GB/s
- 但是这些通信流**无法聚合带宽**（每个 comm 独立）
- 结果：总带宽仍然是 ~2.8 GB/s

### 5. 正确的配置

- 2 个 channels × 2 sockets = 2 个双 socket 通信流
- 每个通信流的带宽：~5 GB/s（2 sockets 聚合）
- 2 个通信流并行：~10 GB/s

---

## 🚀 下一步

1. ✅ 修复 socket 配置（已完成）
2. ✅ 重新编译（已完成）
3. ⏳ 在 GCP 上测试新配置
4. ⏳ 验证带宽是否提升到 ~10 GB/s
5. ⏳ 如果仍然不够，考虑进一步优化（chunk size, window size, etc.）

---

**准备就绪！新配置已编译成功，可以在 GCP 上测试了。** 🚀

