# TCPX Unpack Mode Testing - Summary

## 问题背景 (Background)

### 当前性能状况
- **配置**：2 channels × 4 sockets = 8 TCPX connections per NIC
- **实测带宽**：~2.8 GB/s (22 Gbps)
- **理论上限**：~21.26 GB/s (170 Gbps) per NIC (Google 建议)
- **性能差距**：只达到理论上限的 ~13%

### 瓶颈诊断
从日志分析发现：
1. **网络不是瓶颈**：iperf 裸 TCP 可达 ~8.7 GB/s
2. **Socket 配置正确**：日志显示 "Using 1 threads and 4 sockets per thread"
3. **瓶颈位置**：频繁出现 "waiting for chunk kernel to complete"

**结论**：瓶颈在 GPU unpack kernel，而不是网络或 TCPX 连接数

---

## 解决方案 (Solution)

### 已完成的修改

#### 1. 更新 `run_p2p_fullmesh.sh`
添加了 `UCCL_TCPX_UNPACK_IMPL` 环境变量支持：

```bash
# 新增环境变量
UNPACK_IMPL=${UCCL_TCPX_UNPACK_IMPL:-kernel}  # kernel|d2d|host

# 传递给测试程序
export UCCL_TCPX_UNPACK_IMPL="${UNPACK_IMPL}"
```

**修改位置**：
- Line 26: 添加到 usage 文档
- Line 82: 读取环境变量，默认值 `kernel`
- Line 142: 导出到子进程

#### 2. 创建测试脚本 `test_unpack_modes.sh`
自动化测试三种 unpack 实现方式的性能：

```bash
#!/usr/bin/env bash
# 依次测试 kernel、d2d、host 三种模式
# 自动提取性能数据并生成对比报告
```

**功能**：
- 依次运行 `kernel`、`d2d`、`host` 三种模式
- 每次测试使用相同配置（channels、size、iterations）
- 自动提取 "Avg:" 行并生成对比报告
- 日志保存到独立目录，便于后续分析

---

## 使用方法 (How to Use)

### 方法 1：使用自动化测试脚本（推荐）

```bash
cd /home/daniel/uccl/p2p/tcpx

# Server 节点
./test_unpack_modes.sh server 0

# Client 节点（在另一个终端或节点）
./test_unpack_modes.sh client <SERVER_IP> 0
```

**输出示例**：
```
=========================================
TCPX Unpack Implementation Test
=========================================
Role:       server
GPU:        0
Channels:   2
Size:       67108864 bytes
Iterations: 20
Chunk:      524288 bytes
Log dir:    logs/unpack_test_20251008_123456
=========================================

[12:34:56] Testing unpack mode: kernel
  Result: [PERF] Avg: 22.45 ms, 2.85 GB/s

[12:35:10] Waiting 5 seconds before next test...

[12:35:15] Testing unpack mode: d2d
  Result: [PERF] Avg: 18.32 ms, 3.49 GB/s

[12:35:29] Waiting 5 seconds before next test...

[12:35:34] Testing unpack mode: host
  Result: [PERF] Avg: 45.67 ms, 1.40 GB/s

=========================================
All tests completed!
=========================================

Performance Summary:
-------------------
kernel    : [PERF] Avg: 22.45 ms, 2.85 GB/s
d2d       : [PERF] Avg: 18.32 ms, 3.49 GB/s
host      : [PERF] Avg: 45.67 ms, 1.40 GB/s
```

### 方法 2：手动测试单个模式

```bash
# 测试 kernel 模式
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 测试 d2d 模式
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 测试 host 模式
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh client <SERVER_IP> 0
```

### 方法 3：自定义配置测试

```bash
# 测试更大的消息大小
UCCL_TCPX_PERF_SIZE=536870912 \
UCCL_TCPX_PERF_ITERS=10 \
./test_unpack_modes.sh server 0

# 测试不同的 channel 配置
UCCL_TCPX_NUM_CHANNELS=4 \
NCCL_NSOCKS_PERTHREAD=2 \
./test_unpack_modes.sh server 0
```

---

## 三种 Unpack 实现方式对比

| 模式 | 实现方式 | 优点 | 缺点 | 预期性能 |
|------|---------|------|------|---------|
| **kernel** | GPU kernel 并行拷贝 | 理论最快，充分利用 GPU | 如果实现不优化可能成为瓶颈 | 最快 |
| **d2d** | cuMemcpyDtoD 逐个拷贝 | 使用 CUDA runtime 优化 | 逐个 fragment 调用有开销 | 中等 |
| **host** | DtoH → gather → HtoD | 简单，便于调试 | 两次 PCIe 传输 | 最慢 |

---

## 结果分析指南

### 场景 A：kernel 最快
```
kernel: 2.85 GB/s
d2d:    2.10 GB/s
host:   1.40 GB/s
```

**结论**：kernel 实现正常，瓶颈在其他地方

**下一步优化方向**：
1. 增加 chunk size（当前 512KB → 尝试 1MB、2MB）
2. 增加 sliding window slots（当前 16 → 尝试 32、64）
3. 优化 CUDA stream 并发（使用多个 stream）
4. 检查 TCPX plugin 配置（TX/RX bindings、nanosleep）

### 场景 B：d2d 或 host 更快
```
kernel: 2.85 GB/s
d2d:    3.49 GB/s  ← 更快！
host:   1.40 GB/s
```

**结论**：当前 kernel 实现有问题

**下一步优化方向**：
1. 检查 `device/unpack_kernels.cu` 的 kernel 实现
2. 优化 kernel launch 配置（block size、grid size）
3. 检查 CUDA stream 同步逻辑（是否有不必要的同步）
4. 使用 nsys profiling 分析 kernel 性能
5. 考虑使用 CUDA Graphs 减少 launch overhead

### 场景 C：三种模式性能相近
```
kernel: 2.85 GB/s
d2d:    2.80 GB/s
host:   2.75 GB/s
```

**结论**：瓶颈不在 unpack 路径

**可能的瓶颈**：
1. TCPX plugin 内部（recvmsg、devmem-tcp）
2. 网络配置（NIC offload、TCP tuning）
3. 测试逻辑（sliding window、progress engine）
4. CPU 绑定（TX/RX bindings 不合理）

**下一步**：
1. 使用 `nsys` profiling 定位真正瓶颈
2. 检查 NIC 统计（`ethtool -S eth1`）
3. 检查 CPU 利用率（`mpstat -P ALL 1`）
4. 尝试调整 NCCL_GPUDIRECTTCPX_TX/RX_BINDINGS

---

## 日志检查清单

### 1. 确认 Unpack 模式生效
```bash
grep "Unpack impl:" logs/unpack_test_*/server_gpu0_kernel.log
# 应该看到: [PERF] Unpack impl: kernel
```

### 2. 确认 Socket 配置
```bash
grep "sockets per thread" logs/unpack_test_*/server_gpu0_kernel.log
# 应该看到: NET/GPUDirectTCPX: Using 1 threads and 4 sockets per thread
```

### 3. 提取性能数据
```bash
grep "Avg:" logs/unpack_test_*/*.log
# 对比三种模式的平均时间和带宽
```

### 4. 检查瓶颈位置
```bash
# kernel 模式
grep "waiting for chunk kernel" logs/unpack_test_*/server_gpu0_kernel.log | wc -l

# d2d 模式
grep "d2d copy" logs/unpack_test_*/server_gpu0_d2d.log | wc -l

# host 模式
grep "host gather" logs/unpack_test_*/server_gpu0_host.log | wc -l
```

---

## 进一步调试工具

### 1. nsys Profiling
```bash
nsys profile -o tcpx_kernel.qdrep \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  UCCL_TCPX_UNPACK_IMPL=kernel \
  ./tests/test_tcpx_perf_multi server 0

# 分析结果
nsys stats tcpx_kernel.qdrep
```

### 2. GPU 利用率监控
```bash
# 在测试运行时，另一个终端执行
nvidia-smi dmon -s u -i 0 -d 1
```

### 3. NIC 统计
```bash
# 测试前
ethtool -S eth1 > /tmp/nic_before.txt

# 测试后
ethtool -S eth1 > /tmp/nic_after.txt

# 对比
diff /tmp/nic_before.txt /tmp/nic_after.txt
```

### 4. CPU 绑定检查
```bash
# 在测试运行时
ps -eLo pid,tid,psr,comm | grep test_tcpx
# 检查线程是否绑定到正确的 CPU core
```

---

## 文件清单

### 修改的文件
1. **`run_p2p_fullmesh.sh`**
   - 添加 `UCCL_TCPX_UNPACK_IMPL` 环境变量支持
   - 默认值：`kernel`
   - 可选值：`kernel`、`d2d`、`host`

### 新增的文件
1. **`test_unpack_modes.sh`**
   - 自动化测试脚本
   - 依次测试三种 unpack 模式
   - 生成性能对比报告

2. **`docs/UNPACK_IMPLEMENTATION_TESTING.md`**
   - 详细的测试指南
   - 包含原理、使用方法、结果分析

3. **`UNPACK_MODE_TESTING_SUMMARY.md`** (本文件)
   - 快速参考指南
   - 包含使用方法和结果分析

---

## 快速开始（TL;DR）

```bash
# 1. 进入目录
cd /home/daniel/uccl/p2p/tcpx

# 2. Server 节点运行
./test_unpack_modes.sh server 0

# 3. Client 节点运行（在另一个终端或节点）
./test_unpack_modes.sh client <SERVER_IP> 0

# 4. 查看结果
grep "Avg:" logs/unpack_test_*/server_gpu0_*.log

# 5. 根据结果决定下一步优化方向
```

---

## 预期时间线

1. **运行测试**：~5 分钟（3 种模式 × ~1 分钟/模式）
2. **分析结果**：~10 分钟（查看日志、对比性能）
3. **确定优化方向**：~5 分钟（根据结果分析）
4. **实施优化**：取决于瓶颈位置（几小时到几天）

---

## 联系与支持

如果测试结果不符合预期，或需要进一步分析，请提供：
1. 完整的测试日志（`logs/unpack_test_*/*.log`）
2. 性能对比数据（`grep "Avg:" logs/unpack_test_*/*.log`）
3. 系统配置（GPU 型号、NIC 型号、CUDA 版本）
4. nsys profiling 结果（如果已运行）

---

**最后更新**：2025-10-08  
**作者**：AI Assistant  
**状态**：Ready for testing

