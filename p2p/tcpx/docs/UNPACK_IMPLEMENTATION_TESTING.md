# TCPX Unpack Implementation Testing Guide

## 问题诊断 (Problem Diagnosis)

### 当前性能瓶颈 (Current Performance Bottleneck)

从日志分析可以看到：
- **配置正确**：2 个 GPU 进程，每个 2 channels × 4 sockets = 8 TCPX connections total
- **带宽低**：~2.8 GB/s per iteration (64 MB / 22 ms)
- **瓶颈位置**：频繁出现 "sliding window FULL … waiting for chunk kernel to complete"

**根本原因**：
- iperf 裸 TCP 单流可达 ~8.7 GB/s (70 Gbps)，说明网络链路没问题
- TCPX 测试中，网络接收完成后，需要 **unpack kernel** 将 128 个 4KB fragments 从 bounce buffer 拷贝到目标地址
- 当前瓶颈是 **GPU kernel/拷贝流程**，而不是网络或 socket 数量
- 增加 socket 只是让更多 chunk 同时等待同一个瓶颈

### 测试假设 (Hypothesis)

如果切换到 `d2d` 或 `host` 模式，带宽可能有明显变化：
- **host 模式更快** → 说明当前 kernel 路径限制了吞吐
- **kernel 模式更快** → 说明 kernel 本身没问题，可能是其他配置问题

---

## Unpack 实现方式 (Unpack Implementations)

TCPX 支持三种 unpack 实现方式，通过 `UCCL_TCPX_UNPACK_IMPL` 环境变量控制：

### 1. `kernel` (默认，推荐)
- **实现**：使用 GPU kernel 将 scattered fragments 拷贝到连续内存
- **优点**：理论上最快，充分利用 GPU 并行性
- **缺点**：如果 kernel 实现不优化，可能成为瓶颈
- **代码路径**：`device/unpack_kernels.cu` → `tcpx::device::launch_unpack_kernel()`

### 2. `d2d` (Device-to-Device)
- **实现**：使用 `cuMemcpyDtoD` 逐个 fragment 拷贝
- **优点**：简单直接，使用 CUDA runtime 优化的拷贝
- **缺点**：需要逐个 fragment 调用，可能有额外开销
- **代码路径**：循环调用 `cuMemcpyDtoD(dst + offset, src, 4096)`

### 3. `host` (仅用于调试)
- **实现**：先 DtoH 到 host buffer，gather，再 HtoD 到目标地址
- **优点**：最简单，便于调试
- **缺点**：最慢，涉及两次 PCIe 传输
- **代码路径**：`cuMemcpyDtoH` → host gather → `cuMemcpyHtoD`

---

## 如何测试不同实现 (How to Test)

### 方法 1：使用 `run_p2p_fullmesh.sh` (推荐)

脚本已更新，支持 `UCCL_TCPX_UNPACK_IMPL` 环境变量：

```bash
# 测试 kernel 模式 (默认)
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 测试 d2d 模式
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 测试 host 模式
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh server 0
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh client <SERVER_IP> 0
```

### 方法 2：直接运行测试程序

```bash
# kernel 模式
UCCL_TCPX_UNPACK_IMPL=kernel \
UCCL_TCPX_NUM_CHANNELS=2 \
NCCL_NSOCKS_PERTHREAD=4 \
NCCL_SOCKET_NTHREADS=1 \
./tests/test_tcpx_perf_multi server 0

# d2d 模式
UCCL_TCPX_UNPACK_IMPL=d2d \
UCCL_TCPX_NUM_CHANNELS=2 \
NCCL_NSOCKS_PERTHREAD=4 \
NCCL_SOCKET_NTHREADS=1 \
./tests/test_tcpx_perf_multi server 0

# host 模式
UCCL_TCPX_UNPACK_IMPL=host \
UCCL_TCPX_NUM_CHANNELS=2 \
NCCL_NSOCKS_PERTHREAD=4 \
NCCL_SOCKET_NTHREADS=1 \
./tests/test_tcpx_perf_multi server 0
```

---

## 日志检查 (Log Verification)

### 1. 确认 Unpack 模式

在日志开头查找：
```
[PERF] Unpack impl: kernel
```

### 2. 确认 Socket 配置

```
[ncclNet:3] tcpxResult_t tcpxGetNsockNthread(int, int*, int*):201 NET/GPUDirectTCPX: Using 1 threads and 4 sockets per thread
```

### 3. 观察性能差异

比较不同模式下的迭代时间：
```
[PERF] Iteration 0: 22.45 ms, 2.85 GB/s
```

### 4. 检查瓶颈位置

- **kernel 模式**：查找 "waiting for chunk kernel to complete"
- **d2d 模式**：查找 "d2d copy" 相关日志
- **host 模式**：查找 "host gather" 相关日志

---

## 预期结果 (Expected Results)

### 场景 A：kernel 模式最快
- **结论**：kernel 实现正常，瓶颈在其他地方（如 chunk size、window size、CUDA stream 配置）
- **下一步**：优化 chunk size、增加 window slots、调整 CUDA stream 并发

### 场景 B：d2d 或 host 模式更快
- **结论**：当前 kernel 实现有问题，限制了吞吐
- **下一步**：
  1. 检查 `device/unpack_kernels.cu` 的 kernel 实现
  2. 优化 kernel launch 配置（block size、grid size）
  3. 检查 CUDA stream 同步逻辑
  4. 考虑使用 CUDA Graphs 减少 kernel launch overhead

### 场景 C：三种模式性能相近
- **结论**：瓶颈不在 unpack 路径，可能在：
  - TCPX plugin 内部（recvmsg、devmem-tcp）
  - 网络配置（NIC offload、TCP tuning）
  - 测试逻辑（sliding window、progress engine）
- **下一步**：使用 nsys/nvprof profiling，定位真正瓶颈

---

## 性能对比基准 (Performance Baseline)

### iperf 裸 TCP (单流)
```bash
# Server
iperf3 -s -B 10.128.1.129

# Client
iperf3 -c 10.128.1.129 -t 10 -i 1
```
**预期**：~8.7 GB/s (70 Gbps) per NIC

### TCPX 当前性能
- **kernel 模式**：~2.8 GB/s (22 Gbps)
- **理论上限**：~21.26 GB/s (170 Gbps) per NIC (Google 建议)

**性能差距**：当前只达到理论上限的 ~13%

---

## 调试技巧 (Debugging Tips)

### 1. 使用 NCCL_DEBUG=TRACE
```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
```
查看详细的 TCPX plugin 日志

### 2. 使用 nsys profiling
```bash
nsys profile -o tcpx_kernel.qdrep \
  --trace=cuda,nvtx \
  ./tests/test_tcpx_perf_multi server 0
```
分析 kernel launch、memcpy、同步开销

### 3. 检查 GPU 利用率
```bash
nvidia-smi dmon -s u -i 0
```
如果 GPU 利用率低，说明瓶颈在 CPU 或网络

### 4. 检查 NIC 统计
```bash
ethtool -S eth1 | grep -E 'rx_|tx_'
```
查看丢包、重传、错误

---

## 总结 (Summary)

1. **脚本已更新**：`run_p2p_fullmesh.sh` 现在支持 `UCCL_TCPX_UNPACK_IMPL` 环境变量
2. **测试方法**：依次测试 `kernel`、`d2d`、`host` 三种模式，比较带宽
3. **诊断逻辑**：
   - 如果 `host` 或 `d2d` 更快 → kernel 实现有问题
   - 如果 `kernel` 最快 → 瓶颈在其他地方
   - 如果三者相近 → 瓶颈不在 unpack 路径
4. **下一步**：根据测试结果，针对性优化瓶颈环节

---

## 快速测试命令 (Quick Test Commands)

```bash
# 在 server 节点
cd /home/daniel/uccl/p2p/tcpx

# 测试 kernel 模式
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh server 0 &
sleep 2
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 等待完成，查看日志
grep "Avg:" logs/fullmesh_*.log

# 测试 d2d 模式
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh server 0 &
sleep 2
UCCL_TCPX_UNPACK_IMPL=d2d ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 等待完成，查看日志
grep "Avg:" logs/fullmesh_*.log

# 测试 host 模式
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh server 0 &
sleep 2
UCCL_TCPX_UNPACK_IMPL=host ./run_p2p_fullmesh.sh client <SERVER_IP> 0

# 等待完成，查看日志
grep "Avg:" logs/fullmesh_*.log
```

比较三种模式的 "Avg:" 行，找出最快的实现方式。

