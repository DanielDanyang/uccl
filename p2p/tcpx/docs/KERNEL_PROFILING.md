# Kernel 性能分析指南

## 概述

`test_tcpx_perf_multi` 现在支持详细的 kernel 耗时统计，可以帮助你诊断性能瓶颈。

---

## 功能

### 每次迭代输出

每次迭代结束时，会输出：
- **总耗时**：整个迭代的时间（包括网络传输、kernel 执行、同步等）
- **Kernel 耗时**：所有 kernel 执行的总时间
- **Kernel 占比**：kernel 耗时占总耗时的百分比
- **Kernel 启动次数**：这次迭代启动了多少次 kernel
- **平均 Kernel 耗时**：每次 kernel 启动的平均时间

### 最终平均输出

所有迭代结束后，会输出：
- **平均总耗时**：所有迭代的平均时间
- **平均带宽**：基于平均总耗时计算的带宽
- **平均 Kernel 耗时**：每次迭代的平均 kernel 时间
- **Kernel 占比**：kernel 耗时占总耗时的百分比
- **总 Kernel 启动次数**：所有迭代的总启动次数
- **平均每次启动耗时**：所有 kernel 启动的平均时间
- **非 Kernel 开销**：除了 kernel 之外的其他开销（网络、同步、H2D 拷贝等）

---

## 使用方法

### 1. 编译

```bash
cd /home/daniel/uccl/p2p/tcpx
make clean && make
```

### 2. 运行测试

```bash
# Server 节点
./run_p2p_fullmesh.sh server 0

# Client 节点
./run_p2p_fullmesh.sh client <SERVER_IP> 0
```

### 3. 查看输出

#### 每次迭代输出示例

```
[PERF] Iter 0 time=22.45 ms, kernel_time=18.32 ms (81.6%), launches=128, avg_kernel=143.13 μs
[PERF] Iter 1 time=22.38 ms, kernel_time=18.25 ms (81.5%), launches=128, avg_kernel=142.58 μs
[PERF] Iter 2 time=22.41 ms, kernel_time=18.28 ms (81.6%), launches=128, avg_kernel=142.81 μs
...
```

**解读**：
- 每次迭代耗时 ~22.4 ms
- Kernel 执行耗时 ~18.3 ms（占 81.6%）
- 每次迭代启动了 128 次 kernel（512KB chunk ÷ 4KB fragment = 128）
- 每次 kernel 启动平均耗时 ~143 μs

---

#### 最终平均输出示例

```
[PERF] Avg (10 iter): 22.420 ms, BW: 2.85 GB/s
[PERF] Kernel stats:
  - Avg kernel time per iter: 18.29 ms (81.6% of total)
  - Total kernel launches: 1280 (avg 128.0 per iter)
  - Avg kernel time per launch: 142.89 μs
  - Avg non-kernel overhead: 4.13 ms (18.4%)
```

**解读**：
- 平均每次迭代耗时 22.42 ms，带宽 2.85 GB/s
- Kernel 执行占 81.6% 的时间（18.29 ms）
- 非 kernel 开销占 18.4%（4.13 ms），包括：
  - 网络传输时间
  - H2D 拷贝 descriptor block
  - cudaEventSynchronize 同步
  - tcpx_irecv_consumed 调用
  - 其他 CPU 开销

---

## 性能分析

### 场景 1：Kernel 占比很高（> 80%）

**示例**：
```
[PERF] Avg kernel time per iter: 18.29 ms (81.6% of total)
```

**结论**：**Kernel 是主要瓶颈**

**优化方向**：
1. 实施 kernel 优化（参考 `KERNEL_OPTIMIZATION_VERIFIED.md`）
   - 持久化 Metadata 结构
   - 批量处理 Descriptors
   - 128-bit Load/Store
   - Per-Warp Shared Memory
2. 减少 kernel 启动次数
   - 使用 CUDA Graphs
   - 使用 Persistent Kernel

---

### 场景 2：Kernel 占比中等（50-80%）

**示例**：
```
[PERF] Avg kernel time per iter: 12.50 ms (60.0% of total)
[PERF] Avg non-kernel overhead: 8.33 ms (40.0%)
```

**结论**：**Kernel 和其他开销都需要优化**

**优化方向**：
1. 优化 kernel（参考上面）
2. 优化非 kernel 开销：
   - 减少 H2D 拷贝（持久化 metadata）
   - 减少同步次数（批量处理）
   - 优化网络传输（增加 chunk size）

---

### 场景 3：Kernel 占比很低（< 50%）

**示例**：
```
[PERF] Avg kernel time per iter: 5.00 ms (25.0% of total)
[PERF] Avg non-kernel overhead: 15.00 ms (75.0%)
```

**结论**：**非 kernel 开销是主要瓶颈**

**可能原因**：
1. **网络传输慢**
   - 检查 iperf 带宽
   - 检查 NIC 配置
   - 检查 NUMA 绑定
2. **H2D 拷贝慢**
   - 每次迭代拷贝 ~2KB descriptors
   - 优化：持久化 metadata 结构
3. **同步开销大**
   - 频繁的 cudaEventSynchronize
   - 优化：批量处理，减少同步次数
4. **CPU 开销大**
   - tcpx_irecv_consumed 调用
   - 优化：减少调用次数

---

## 诊断流程

### 第一步：运行测试并查看 kernel 占比

```bash
./run_p2p_fullmesh.sh server 0 | tee kernel_profile.log
grep "Avg kernel time per iter" kernel_profile.log
```

---

### 第二步：根据 kernel 占比判断瓶颈

#### 如果 kernel 占比 > 80%

**瓶颈**：Kernel 执行

**验证**：
```bash
# 测试不同 unpack 模式
./test_unpack_modes.sh server 0
grep "Avg:" logs/unpack_test_*/server_gpu0_*.log
```

如果 `d2d` 模式更快 → kernel 实现有问题 → 实施 kernel 优化

---

#### 如果 kernel 占比 < 50%

**瓶颈**：非 kernel 开销

**验证**：
```bash
# 检查网络带宽
iperf3 -c <SERVER_IP> -t 10

# 检查 H2D 拷贝开销
# 查看日志中的 "copyDescriptorBlockToDevice" 时间
```

---

### 第三步：实施优化

根据瓶颈类型，参考：
- **Kernel 优化**：`KERNEL_OPTIMIZATION_VERIFIED.md`
- **Chunk Size 优化**：`CHUNK_SIZE_QUICK_REF.md`
- **RX Timeout 修复**：`RX_TIMEOUT_QUICK_FIX.md`

---

## 高级分析

### 计算理论 Kernel 耗时

假设：
- Chunk size = 512 KB
- Fragment size = 4 KB
- Fragments per chunk = 512 KB ÷ 4 KB = 128
- Kernel launches per chunk = 128（当前实现：1 launch per fragment）

**理论最小 kernel 耗时**：
- 假设每个 fragment 拷贝 4 KB 数据
- H100 GPU memory bandwidth = ~3 TB/s
- 理论拷贝时间 = 4 KB ÷ 3 TB/s = **1.3 ns**
- 128 个 fragments = 128 × 1.3 ns = **0.17 μs**

**实际 kernel 耗时**：~143 μs（从上面的示例）

**差距**：143 μs ÷ 0.17 μs = **841 倍**

**原因**：
1. **Kernel launch overhead**：~5-10 μs per launch
   - 128 launches × 10 μs = **1.28 ms**（主要瓶颈！）
2. **Global memory 访问延迟**：~400 cycles
3. **Volatile load/store**：禁用缓存和合并
4. **H2D 拷贝 descriptor**：每次 ~2 KB

**优化潜力**：
- 批量处理 descriptors：减少 launch 次数 128 → 1，节省 **1.27 ms**
- 128-bit load/store：提升 20-30%，节省 **~3 ms**
- 持久化 metadata：减少 H2D 拷贝，节省 **~0.05 ms**
- **总潜力**：~4.3 ms → 从 18.3 ms 降到 **~14 ms**（提升 23%）

---

## 环境变量

### 启用/禁用 Profiling

Profiling 现在默认启用。如果需要禁用（减少开销），修改代码：

```cpp
// p2p/tcpx/tests/test_tcpx_perf_multi.cc:452
cfg.enable_profiling = false;  // Disable profiling
```

---

## 示例输出（完整）

```
[PERF] Iteration 0: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 0 time=22.45 ms, kernel_time=18.32 ms (81.6%), launches=128, avg_kernel=143.13 μs
[PERF] Iteration 1: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 1 time=22.38 ms, kernel_time=18.25 ms (81.5%), launches=128, avg_kernel=142.58 μs
[PERF] Iteration 2: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 2 time=22.41 ms, kernel_time=18.28 ms (81.6%), launches=128, avg_kernel=142.81 μs
[PERF] Iteration 3: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 3 time=22.39 ms, kernel_time=18.26 ms (81.6%), launches=128, avg_kernel=142.66 μs
[PERF] Iteration 4: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 4 time=22.42 ms, kernel_time=18.29 ms (81.6%), launches=128, avg_kernel=142.89 μs
[PERF] Iteration 5: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 5 time=22.40 ms, kernel_time=18.27 ms (81.6%), launches=128, avg_kernel=142.73 μs
[PERF] Iteration 6: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 6 time=22.43 ms, kernel_time=18.30 ms (81.6%), launches=128, avg_kernel=142.97 μs
[PERF] Iteration 7: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 7 time=22.41 ms, kernel_time=18.28 ms (81.6%), launches=128, avg_kernel=142.81 μs
[PERF] Iteration 8: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 8 time=22.44 ms, kernel_time=18.31 ms (81.6%), launches=128, avg_kernel=143.05 μs
[PERF] Iteration 9: total bytes=67108864, chunk_bytes=524288
[PERF] Iter 9 time=22.39 ms, kernel_time=18.26 ms (81.6%), launches=128, avg_kernel=142.66 μs

[PERF] Avg (10 iter): 22.420 ms, BW: 2.85 GB/s
[PERF] Kernel stats:
  - Avg kernel time per iter: 18.29 ms (81.6% of total)
  - Total kernel launches: 1280 (avg 128.0 per iter)
  - Avg kernel time per launch: 142.89 μs
  - Avg non-kernel overhead: 4.13 ms (18.4%)
```

---

## 总结

现在你可以精确测量 kernel 耗时，并判断 kernel 是否是性能瓶颈！

**关键指标**：
- **Kernel 占比 > 80%** → 优化 kernel
- **Kernel 占比 < 50%** → 优化网络/同步/H2D 拷贝
- **Avg kernel time per launch > 100 μs** → 可能有优化空间

**下一步**：
1. 运行测试，查看 kernel 占比
2. 根据占比判断瓶颈
3. 实施相应的优化

---

**最后更新**：2025-10-08  
**作者**：AI Assistant

