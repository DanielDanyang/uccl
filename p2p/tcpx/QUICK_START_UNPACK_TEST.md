# 快速开始：TCPX Unpack 模式测试

## 一句话总结
测试 kernel/d2d/host 三种 unpack 实现，找出性能瓶颈是否在 GPU kernel。

---

## 最快的测试方法（推荐）

### Server 节点
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_unpack_modes.sh server 0
```

### Client 节点
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_unpack_modes.sh client <SERVER_IP> 0
```

**等待 5 分钟**，脚本会自动：
1. 依次测试 kernel、d2d、host 三种模式
2. 每次测试使用相同配置（2 channels, 64MB, 20 iterations）
3. 自动提取性能数据并生成对比报告

---

## 查看结果

测试完成后，会自动显示：
```
Performance Summary:
-------------------
kernel    : [PERF] Avg: 22.45 ms, 2.85 GB/s
d2d       : [PERF] Avg: 18.32 ms, 3.49 GB/s  ← 如果这个更快，说明 kernel 有问题
host      : [PERF] Avg: 45.67 ms, 1.40 GB/s
```

---

## 结果分析（3 种情况）

### 情况 1：kernel 最快 ✓
```
kernel: 2.85 GB/s  ← 最快
d2d:    2.10 GB/s
host:   1.40 GB/s
```
**结论**：kernel 实现正常，瓶颈在其他地方  
**下一步**：优化 chunk size、window size、CUDA stream 配置

---

### 情况 2：d2d 更快 ⚠️
```
kernel: 2.85 GB/s
d2d:    3.49 GB/s  ← 更快！
host:   1.40 GB/s
```
**结论**：当前 kernel 实现有问题  
**下一步**：优化 `device/unpack_kernels.cu`，或使用 nsys profiling 分析

---

### 情况 3：三者相近 🤔
```
kernel: 2.85 GB/s
d2d:    2.80 GB/s
host:   2.75 GB/s
```
**结论**：瓶颈不在 unpack 路径  
**下一步**：检查 TCPX plugin、网络配置、CPU 绑定

---

## 手动测试单个模式

如果只想测试某一个模式：

```bash
# 测试 d2d 模式
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 查看结果
grep "Avg:" logs/fullmesh_*.log
```

---

## 自定义配置

```bash
# 测试更大的消息（512 MB）
UCCL_TCPX_PERF_SIZE=536870912 ./test_unpack_modes.sh server 0

# 测试更多 channels（4 channels × 2 sockets = 8 connections）
UCCL_TCPX_NUM_CHANNELS=4 \
NCCL_NSOCKS_PERTHREAD=2 \
./test_unpack_modes.sh server 0
```

---

## 日志位置

- **自动化测试**：`logs/unpack_test_<timestamp>/`
- **手动测试**：`logs/fullmesh_*.log`

---

## 详细文档

- **完整指南**：`docs/UNPACK_IMPLEMENTATION_TESTING.md`
- **总结文档**：`UNPACK_MODE_TESTING_SUMMARY.md`

---

## 常见问题

### Q: 为什么要测试不同的 unpack 模式？
A: 当前带宽只有 ~2.8 GB/s，远低于理论上限 ~21 GB/s。日志显示瓶颈在 "waiting for chunk kernel to complete"。通过对比三种实现，可以确定是否是 kernel 本身的问题。

### Q: 三种模式有什么区别？
A:
- **kernel**：GPU kernel 并行拷贝（理论最快）
- **d2d**：cuMemcpyDtoD 逐个拷贝（中等速度）
- **host**：DtoH → gather → HtoD（最慢，仅用于调试）

### Q: 测试需要多长时间？
A: 约 5 分钟（3 种模式 × ~1 分钟/模式 + 间隔时间）

### Q: 如果 d2d 更快，下一步怎么办？
A: 说明当前 kernel 实现有问题，需要：
1. 检查 `device/unpack_kernels.cu` 的实现
2. 使用 nsys profiling 分析 kernel 性能
3. 优化 kernel launch 配置（block size、grid size）
4. 检查 CUDA stream 同步逻辑

### Q: 如果三种模式性能相近，说明什么？
A: 说明瓶颈不在 unpack 路径，可能在：
- TCPX plugin 内部（recvmsg、devmem-tcp）
- 网络配置（NIC offload、TCP tuning）
- CPU 绑定（TX/RX bindings）

---

## 一键运行（复制粘贴）

```bash
# Server 节点
cd /home/daniel/uccl/p2p/tcpx && ./test_unpack_modes.sh server 0

# Client 节点（另一个终端）
cd /home/daniel/uccl/p2p/tcpx && ./test_unpack_modes.sh client <SERVER_IP> 0

# 查看结果
grep "Avg:" logs/unpack_test_*/server_gpu0_*.log
```

---

**预期时间**：5 分钟  
**预期输出**：三种模式的性能对比  
**下一步**：根据结果决定优化方向

