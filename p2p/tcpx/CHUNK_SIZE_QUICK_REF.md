# Chunk Size 调整快速参考

## 一句话回答

**需要调整 3 个参数（保持一致）：**
```bash
export UCCL_TCPX_CHUNK_BYTES=1048576        # 应用层 chunk
export NCCL_DYNAMIC_CHUNK_SIZE=1048576      # TCPX plugin chunk  
export NCCL_P2P_NET_CHUNKSIZE=1048576       # NCCL 网络 chunk
```

**不需要调整 3 个参数（保持默认）：**
```bash
export NCCL_P2P_PCI_CHUNKSIZE=524288        # PCIe P2P (我们不用)
export NCCL_P2P_NVL_CHUNKSIZE=1048576       # NVLink (我们不用)
export NCCL_BUFFSIZE=8388608                # 内部缓冲区 (足够大)
```

---

## 为什么不需要调整后 3 个？

### NCCL_P2P_PCI_CHUNKSIZE
- **用途**：同节点内 GPU 通过 PCIe 传输
- **我们的场景**：跨节点网络传输
- **结论**：不影响我们的测试 ❌

### NCCL_P2P_NVL_CHUNKSIZE  
- **用途**：同节点内 GPU 通过 NVLink 传输
- **我们的场景**：跨节点网络传输
- **结论**：不影响我们的测试 ❌

### NCCL_BUFFSIZE
- **用途**：NCCL 内部缓冲区总大小（不是 chunk size）
- **当前值**：8 MB（可容纳多个 chunk）
- **结论**：除非 chunk > 4 MB，否则不需要调整 ⚠️

---

## 快速测试不同 Chunk Size

### 自动化测试（推荐）

```bash
# Server 节点
cd /home/daniel/uccl/p2p/tcpx
./test_chunk_sizes.sh server 0

# Client 节点
./test_chunk_sizes.sh client <SERVER_IP> 0
```

**自动测试**：512KB → 1MB → 2MB → 4MB

**预期输出**：
```
Performance Summary:
-------------------
512KB     : [PERF] Avg: 22.45 ms, 2.85 GB/s
1MB       : [PERF] Avg: 18.32 ms, 3.49 GB/s  ← 提升 22%
2MB       : [PERF] Avg: 15.20 ms, 4.21 GB/s  ← 提升 48%
4MB       : [PERF] Avg: 14.50 ms, 4.41 GB/s  ← 提升 55%

Performance Trend:
-------------------
512KB      [#####                ] 2.85 GB/s
1MB        [######                ] 3.49 GB/s
2MB        [########              ] 4.21 GB/s
4MB        [########              ] 4.41 GB/s
```

---

## 手动测试单个 Chunk Size

### 测试 1 MB
```bash
UCCL_TCPX_CHUNK_BYTES=1048576 \
NCCL_DYNAMIC_CHUNK_SIZE=1048576 \
NCCL_P2P_NET_CHUNKSIZE=1048576 \
./run_p2p_fullmesh.sh server 0
```

### 测试 2 MB
```bash
UCCL_TCPX_CHUNK_BYTES=2097152 \
NCCL_DYNAMIC_CHUNK_SIZE=2097152 \
NCCL_P2P_NET_CHUNKSIZE=2097152 \
./run_p2p_fullmesh.sh server 0
```

### 测试 4 MB（需要调整 BUFFSIZE）
```bash
UCCL_TCPX_CHUNK_BYTES=4194304 \
NCCL_DYNAMIC_CHUNK_SIZE=4194304 \
NCCL_P2P_NET_CHUNKSIZE=4194304 \
NCCL_BUFFSIZE=16777216 \
./run_p2p_fullmesh.sh server 0
```

---

## 日志验证

### 1. 确认 Chunk Size 生效
```bash
grep "Chunk size:" logs/chunk_test_*/server_gpu0_1MB.log
# 应该看到: [PERF] Chunk size: 1048576 bytes (1 MB)

grep "dynamic chunk size:" logs/chunk_test_*/server_gpu0_1MB.log
# 应该看到: NET/GPUDirectTCPX : dynamic chunk size: 1048576
```

### 2. 对比性能
```bash
grep "Avg:" logs/chunk_test_*/*.log
```

---

## 结果分析（3 种情况）

### 情况 1：Chunk 越大越快 ✓
```
512KB: 2.85 GB/s
1MB:   3.49 GB/s  ← 提升 22%
2MB:   4.21 GB/s  ← 提升 48%
```
**结论**：Kernel launch overhead 是瓶颈  
**建议**：使用 2MB 或更大的 chunk

---

### 情况 2：Chunk 大小不影响性能 ⚠️
```
512KB: 2.85 GB/s
1MB:   2.90 GB/s  ← 几乎不变
2MB:   2.95 GB/s  ← 几乎不变
```
**结论**：瓶颈不在 kernel launch  
**建议**：测试不同的 unpack 实现（kernel/d2d/host）

---

### 情况 3：Chunk 越大越慢 ❌
```
512KB: 2.85 GB/s
1MB:   2.50 GB/s  ← 下降！
2MB:   2.20 GB/s  ← 继续下降
```
**结论**：Chunk 太大导致其他问题  
**建议**：保持 512KB 或 1MB，检查其他瓶颈

---

## 推荐配置

### 保守配置（1 MB）
```bash
export UCCL_TCPX_CHUNK_BYTES=1048576
export NCCL_DYNAMIC_CHUNK_SIZE=1048576
export NCCL_P2P_NET_CHUNKSIZE=1048576
# NCCL_BUFFSIZE 保持 8MB 即可
```

### 激进配置（2 MB）
```bash
export UCCL_TCPX_CHUNK_BYTES=2097152
export NCCL_DYNAMIC_CHUNK_SIZE=2097152
export NCCL_P2P_NET_CHUNKSIZE=2097152
# NCCL_BUFFSIZE 保持 8MB 即可
```

### 极限配置（4 MB）
```bash
export UCCL_TCPX_CHUNK_BYTES=4194304
export NCCL_DYNAMIC_CHUNK_SIZE=4194304
export NCCL_P2P_NET_CHUNKSIZE=4194304
export NCCL_BUFFSIZE=16777216  # 需要增加！
```

---

## 一键运行（复制粘贴）

```bash
# Server 节点
cd /home/daniel/uccl/p2p/tcpx && ./test_chunk_sizes.sh server 0

# Client 节点
cd /home/daniel/uccl/p2p/tcpx && ./test_chunk_sizes.sh client <SERVER_IP> 0

# 查看结果
grep "Avg:" logs/chunk_test_*/server_gpu0_*.log
```

---

## 详细文档

完整解释请参考：`docs/CHUNK_SIZE_PARAMETERS.md`

---

**预期时间**：10 分钟（4 种 chunk size × ~2 分钟/测试）  
**预期输出**：性能对比 + 趋势图  
**下一步**：根据结果选择最优 chunk size

