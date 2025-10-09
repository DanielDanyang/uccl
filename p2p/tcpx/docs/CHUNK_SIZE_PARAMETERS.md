# TCPX Chunk Size 参数详解

## 快速回答

**需要调整的参数（3 个）：**
1. ✅ `UCCL_TCPX_CHUNK_BYTES` - 应用层 chunk size
2. ✅ `NCCL_DYNAMIC_CHUNK_SIZE` - TCPX plugin chunk size
3. ✅ `NCCL_P2P_NET_CHUNKSIZE` - NCCL 网络传输 chunk size

**建议保持不变的参数（3 个）：**
1. ⚠️ `NCCL_P2P_PCI_CHUNKSIZE` - PCIe 传输 chunk（我们不用 PCIe P2P）
2. ⚠️ `NCCL_P2P_NVL_CHUNKSIZE` - NVLink 传输 chunk（我们不用 NVLink）
3. ⚠️ `NCCL_BUFFSIZE` - NCCL 内部缓冲区大小（与 chunk size 无关）

---

## 详细解释

### 1. UCCL_TCPX_CHUNK_BYTES ✅ **必须调整**

**作用**：
- 你的测试程序 `test_tcpx_perf_multi.cc` 中每个 chunk 的大小
- 控制滑动窗口中每个 slot 处理的数据量
- 每个 chunk 会被拆分成多个 4KB fragments 发送

**当前值**：`524288` (512 KB)

**代码位置**：
```cpp
// tests/test_tcpx_perf_multi.cc, line ~340
const char* chunk_env = std::getenv("UCCL_TCPX_CHUNK_BYTES");
size_t chunk_bytes = chunk_env ? std::atoll(chunk_env) : 2097152;  // 默认 2MB
```

**影响**：
- 每个 chunk = 512 KB → 128 个 4KB fragments
- 每个 chunk 需要一次 unpack kernel 调用
- Chunk 越大 → kernel 调用次数越少 → 可能减少 overhead

**建议值**：
- 当前：512 KB (128 fragments)
- 尝试：1 MB (256 fragments)
- 尝试：2 MB (512 fragments)
- 上限：取决于 GPU 内存和 TCPX 限制

---

### 2. NCCL_DYNAMIC_CHUNK_SIZE ✅ **必须调整**

**作用**：
- TCPX plugin 内部的 chunk size
- 控制每次 `tcpx_isend`/`tcpx_irecv` 的最大数据量
- TCPX plugin 会将大于此值的传输拆分成多个 chunk

**当前值**：`524288` (512 KB)

**代码位置**：
- TCPX plugin 内部（`/usr/local/tcpx/lib64/libnccl-net-tcpx.so`）
- 日志中可见：`NET/GPUDirectTCPX : dynamic chunk size: 524288`

**影响**：
- 如果 `UCCL_TCPX_CHUNK_BYTES > NCCL_DYNAMIC_CHUNK_SIZE`，TCPX 会自动拆分
- 建议保持 `NCCL_DYNAMIC_CHUNK_SIZE >= UCCL_TCPX_CHUNK_BYTES`

**建议值**：
- **与 `UCCL_TCPX_CHUNK_BYTES` 保持一致**
- 例如：`UCCL_TCPX_CHUNK_BYTES=1048576` → `NCCL_DYNAMIC_CHUNK_SIZE=1048576`

---

### 3. NCCL_P2P_NET_CHUNKSIZE ✅ **必须调整**

**作用**：
- NCCL 的 P2P 网络传输 chunk size
- 控制 NCCL collective 操作中每次网络传输的大小
- 虽然我们的测试不是 NCCL collective，但 TCPX plugin 可能会参考此值

**当前值**：`524288` (512 KB)

**影响**：
- 与 `NCCL_DYNAMIC_CHUNK_SIZE` 配合使用
- 建议保持一致

**建议值**：
- **与 `UCCL_TCPX_CHUNK_BYTES` 和 `NCCL_DYNAMIC_CHUNK_SIZE` 保持一致**

---

### 4. NCCL_P2P_PCI_CHUNKSIZE ⚠️ **不需要调整**

**作用**：
- NCCL 的 PCIe P2P 传输 chunk size
- 用于同一节点内不同 GPU 之间通过 PCIe 传输数据

**当前值**：`524288` (512 KB)

**为什么不需要调整**：
- 我们的测试是**跨节点**的 GPU-to-GPU 传输（通过网络）
- 不涉及 PCIe P2P 传输
- 此参数对我们的测试**没有影响**

**建议**：保持默认值即可

---

### 5. NCCL_P2P_NVL_CHUNKSIZE ⚠️ **不需要调整**

**作用**：
- NCCL 的 NVLink 传输 chunk size
- 用于同一节点内不同 GPU 之间通过 NVLink 传输数据

**当前值**：`1048576` (1 MB)

**为什么不需要调整**：
- 我们的测试是**跨节点**的 GPU-to-GPU 传输（通过网络）
- 不涉及 NVLink 传输
- 此参数对我们的测试**没有影响**

**建议**：保持默认值即可

---

### 6. NCCL_BUFFSIZE ⚠️ **不需要调整**

**作用**：
- NCCL 内部缓冲区的大小
- 用于 NCCL collective 操作的中间缓冲区
- **不是** chunk size，而是缓冲区总大小

**当前值**：`8388608` (8 MB)

**为什么不需要调整**：
- 这是 NCCL 内部的缓冲区管理参数
- 与我们的 chunk size 概念不同
- 8 MB 已经足够大，可以容纳多个 chunk

**建议**：保持默认值即可（除非遇到 NCCL 缓冲区不足的错误）

---

## 推荐的调整方案

### 方案 1：增加到 1 MB（保守）

```bash
export UCCL_TCPX_CHUNK_BYTES=1048576        # 1 MB
export NCCL_DYNAMIC_CHUNK_SIZE=1048576      # 1 MB
export NCCL_P2P_NET_CHUNKSIZE=1048576       # 1 MB

# 以下保持不变
export NCCL_P2P_PCI_CHUNKSIZE=524288        # 512 KB (不影响)
export NCCL_P2P_NVL_CHUNKSIZE=1048576       # 1 MB (不影响)
export NCCL_BUFFSIZE=8388608                # 8 MB (足够大)
```

**预期效果**：
- Chunk 数量减半（512 KB → 1 MB）
- Kernel 调用次数减半
- 可能减少 kernel launch overhead

---

### 方案 2：增加到 2 MB（激进）

```bash
export UCCL_TCPX_CHUNK_BYTES=2097152        # 2 MB
export NCCL_DYNAMIC_CHUNK_SIZE=2097152      # 2 MB
export NCCL_P2P_NET_CHUNKSIZE=2097152       # 2 MB

# 以下保持不变
export NCCL_P2P_PCI_CHUNKSIZE=524288        # 512 KB (不影响)
export NCCL_P2P_NVL_CHUNKSIZE=1048576       # 1 MB (不影响)
export NCCL_BUFFSIZE=8388608                # 8 MB (足够大)
```

**预期效果**：
- Chunk 数量减少到 1/4（512 KB → 2 MB）
- Kernel 调用次数减少到 1/4
- 可能进一步减少 overhead，但单个 chunk 处理时间更长

---

### 方案 3：增加到 4 MB（极限测试）

```bash
export UCCL_TCPX_CHUNK_BYTES=4194304        # 4 MB
export NCCL_DYNAMIC_CHUNK_SIZE=4194304      # 4 MB
export NCCL_P2P_NET_CHUNKSIZE=4194304       # 4 MB

# 以下保持不变
export NCCL_P2P_PCI_CHUNKSIZE=524288        # 512 KB (不影响)
export NCCL_P2P_NVL_CHUNKSIZE=1048576       # 1 MB (不影响)
export NCCL_BUFFSIZE=8388608                # 8 MB (需要增加！)
```

**注意**：
- 如果 chunk size > NCCL_BUFFSIZE，可能需要增加 `NCCL_BUFFSIZE`
- 建议：`NCCL_BUFFSIZE >= 2 × UCCL_TCPX_CHUNK_BYTES`

```bash
export NCCL_BUFFSIZE=16777216               # 16 MB (2 × 4 MB + 余量)
```

---

## 如何测试不同的 Chunk Size

### 方法 1：使用环境变量（推荐）

```bash
# 测试 1 MB chunk
UCCL_TCPX_CHUNK_BYTES=1048576 \
NCCL_DYNAMIC_CHUNK_SIZE=1048576 \
NCCL_P2P_NET_CHUNKSIZE=1048576 \
./run_p2p_fullmesh.sh server 0

# 测试 2 MB chunk
UCCL_TCPX_CHUNK_BYTES=2097152 \
NCCL_DYNAMIC_CHUNK_SIZE=2097152 \
NCCL_P2P_NET_CHUNKSIZE=2097152 \
./run_p2p_fullmesh.sh server 0
```

### 方法 2：修改脚本默认值

编辑 `run_p2p_fullmesh.sh`，修改第 80 行：

```bash
# 当前
CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-524288}

# 改为 1 MB
CHUNK_BYTES=${UCCL_TCPX_CHUNK_BYTES:-1048576}
```

同时修改第 105-106 行：

```bash
# 当前
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288

# 改为 1 MB
export NCCL_DYNAMIC_CHUNK_SIZE=1048576
export NCCL_P2P_NET_CHUNKSIZE=1048576
```

---

## 日志验证

### 1. 确认 UCCL_TCPX_CHUNK_BYTES 生效

```bash
grep "Chunk size:" logs/fullmesh_*.log
# 应该看到: [PERF] Chunk size: 1048576 bytes (1 MB)
```

### 2. 确认 NCCL_DYNAMIC_CHUNK_SIZE 生效

```bash
grep "dynamic chunk size:" logs/fullmesh_*.log
# 应该看到: NET/GPUDirectTCPX : dynamic chunk size: 1048576
```

### 3. 观察性能变化

```bash
grep "Avg:" logs/fullmesh_*.log
# 对比不同 chunk size 的平均时间和带宽
```

---

## 预期结果

### 场景 A：Chunk size 增加 → 带宽提升 ✓

```
512 KB: 2.85 GB/s
1 MB:   4.20 GB/s  ← 提升 ~47%
2 MB:   5.50 GB/s  ← 提升 ~93%
```

**结论**：Kernel launch overhead 是主要瓶颈

**下一步**：继续增加 chunk size，或使用 CUDA Graphs 减少 launch overhead

---

### 场景 B：Chunk size 增加 → 带宽不变 ⚠️

```
512 KB: 2.85 GB/s
1 MB:   2.90 GB/s  ← 几乎不变
2 MB:   2.95 GB/s  ← 几乎不变
```

**结论**：瓶颈不在 kernel launch overhead

**可能的瓶颈**：
- Kernel 本身的执行时间（unpack 拷贝速度慢）
- 网络接收速度（TCPX plugin、devmem-tcp）
- CUDA stream 同步（cudaEventSynchronize）

**下一步**：
1. 测试不同的 unpack 实现（kernel/d2d/host）
2. 使用 nsys profiling 分析 kernel 性能
3. 检查网络统计（ethtool -S eth1）

---

### 场景 C：Chunk size 增加 → 带宽下降 ❌

```
512 KB: 2.85 GB/s
1 MB:   2.50 GB/s  ← 下降！
2 MB:   2.20 GB/s  ← 继续下降
```

**结论**：Chunk 太大导致其他问题

**可能的原因**：
- 滑动窗口 slot 不足（16 个 slot × 2 MB = 32 MB，可能不够）
- GPU 内存带宽饱和（单个 kernel 处理太多数据）
- TCPX plugin 内部限制

**下一步**：
1. 增加滑动窗口 slots（当前 16 → 尝试 32）
2. 检查 GPU 内存带宽（nvidia-smi dmon）
3. 回退到较小的 chunk size

---

## 总结

### 需要调整的参数（保持一致）：
```bash
export UCCL_TCPX_CHUNK_BYTES=1048576        # 应用层 chunk
export NCCL_DYNAMIC_CHUNK_SIZE=1048576      # TCPX plugin chunk
export NCCL_P2P_NET_CHUNKSIZE=1048576       # NCCL 网络 chunk
```

### 不需要调整的参数（保持默认）：
```bash
export NCCL_P2P_PCI_CHUNKSIZE=524288        # PCIe P2P (不影响)
export NCCL_P2P_NVL_CHUNKSIZE=1048576       # NVLink (不影响)
export NCCL_BUFFSIZE=8388608                # 内部缓冲区 (足够大)
```

### 快速测试命令：
```bash
# 测试 1 MB chunk
UCCL_TCPX_CHUNK_BYTES=1048576 \
NCCL_DYNAMIC_CHUNK_SIZE=1048576 \
NCCL_P2P_NET_CHUNKSIZE=1048576 \
./test_unpack_modes.sh server 0
```

---

**最后更新**：2025-10-08  
**作者**：AI Assistant

