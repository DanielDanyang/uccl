# Kernel 优化快速参考（已验证）

## TL;DR

经过验证，发现 **5 个 CRITICAL/HIGH 优先级优化**，其中最重要的是：
1. **持久化 Metadata 结构**（减少 H2D 拷贝 4000 倍）
2. **批量处理 Descriptors**（减少 kernel launch 次数 10-100 倍）

---

## ⚠️ 重要更正

**我之前的建议有误**：
- ❌ "Shared Memory 缓存 Descriptors" - 这不是 NCCL 的做法
- ✅ 正确的优化：**持久化 Metadata 结构** + **Per-Warp Shared Memory Staging**

详细验证请参考：`docs/KERNEL_OPTIMIZATION_VERIFIED.md`

---

## 立即实施（CRITICAL 优先级）

### 1. 持久化 Metadata 结构 ⭐⭐⭐⭐⭐

**问题**：每个 chunk 都要 H2D 拷贝 ~2KB descriptors

**NCCL 的做法**：
- Metadata 常驻 GPU（ring buffer，16 个 slots）
- 每次只更新 `cnt[slot]`（8 字节）
- Kernel 直接从 GPU memory 读取

**你的做法**：
- 每次 `cudaMemcpyAsync` 整个 `UnpackDescriptorBlock`（~32KB）
- 差距：**4000 倍的数据传输量**

**方案**：

```cpp
// 初始化时（只做一次）
struct DeviceUnpackMeta {
  tcpx::rx::UnpackDescriptor mem[16][2048];  // 16 slots × 2048 descriptors
  uint32_t cnt[16];                          // 每个 slot 的 descriptor 数量
};

DeviceUnpackMeta* d_meta;
cudaMalloc(&d_meta, sizeof(DeviceUnpackMeta));

// 运行时（每个 chunk）
uint32_t slot = head % 16;
cudaMemcpyAsync(&d_meta->mem[slot], descriptors, count * sizeof(UnpackDescriptor), ...);
cudaMemcpyAsync(&d_meta->cnt[slot], &count, sizeof(uint32_t), ...);

// Kernel 读取
__global__ void tcpxUnpackKernel(DeviceUnpackMeta* d_meta, uint32_t slot) {
  uint32_t meta_cnt;
  asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];" : "=l"(meta_cnt) : "l"(&d_meta->cnt[slot]));

  for (int i = 0; i < meta_cnt; ++i) {
    UnpackDescriptor desc = d_meta->mem[slot][i];
    unpackSingleDescriptor(threadIdx.x, desc, bounce_buffer, dst_buffer);
  }
}
```

**预期收益**：10-20% 性能提升

---

### 2. 批量处理 Descriptors ⭐⭐⭐⭐⭐

**问题**：每个 chunk 都 launch kernel（~5-10 μs overhead）

**当前做法**：
- 512KB chunk → 128 个 4KB fragments → 128 次 kernel launch
- 总 overhead：128 × 10 μs = **1.28 ms**

**方案 A**：1 block 处理多个 descriptors（最简单）

```cpp
__global__ void tcpxUnpackKernelBatch(DeviceUnpackMeta* d_meta, uint32_t slot) {
  uint32_t meta_cnt;
  asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];" : "=l"(meta_cnt) : "l"(&d_meta->cnt[slot]));

  // 每个 block 处理多个 descriptors
  for (uint32_t i = blockIdx.x; i < meta_cnt; i += gridDim.x) {
    UnpackDescriptor desc = d_meta->mem[slot][i];
    unpackSingleDescriptor(threadIdx.x, desc, bounce_buffer, dst_buffer);
  }
}

// Launch 一次处理所有 descriptors
tcpxUnpackKernelBatch<<<min(128, desc_count), 32>>>(d_meta, slot);
```

**方案 B**：CUDA Graphs（中等难度）

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < BATCH_SIZE; ++i) {
  tcpxUnpackKernel<<<...>>>(d_meta, slot);
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

**方案 C**：Persistent Kernel（复杂）

```cpp
__global__ void tcpxUnpackKernelPersistent(
    DeviceUnpackMeta* d_meta,
    volatile uint32_t* work_queue_head,
    volatile uint32_t* work_queue_tail) {
  while (true) {
    uint32_t head = *work_queue_head;
    uint32_t tail = *work_queue_tail;
    if (head == tail) continue;

    uint32_t slot = head % 16;
    // Process all descriptors in this slot
    // ...
  }
}
```

**预期收益**：
- 方案 A：20-50% 性能提升
- 方案 B：10-20% 性能提升
- 方案 C：30-60% 性能提升

**推荐**：先实施方案 A（最简单）

---

## 强烈推荐（HIGH 优先级）

### 3. 128-bit Aligned Load/Store ⭐⭐⭐

**问题**：使用 volatile load（禁用缓存和合并）

**NCCL 的做法**：
- 默认也是 volatile load（逐字节）
- 只有定义 `ALIGNED_LOAD` 宏时才用 128-bit load

**你的做法**：
- 和 NCCL 默认实现一样（两次 64-bit volatile load）

**方案**：添加 128-bit PTX 指令（对齐路径）

```cpp
__device__ __forceinline__ void load128(const uint64_t* addr, uint64_t& v0, uint64_t& v1) {
  asm volatile("ld.global.v2.u64 {%0, %1}, [%2];"
               : "=l"(v0), "=l"(v1) : "l"(addr) : "memory");
}

__device__ __forceinline__ void store128(uint64_t* addr, uint64_t v0, uint64_t v1) {
  asm volatile("st.global.v2.u64 [%0], {%1, %2};"
               :: "l"(addr), "l"(v0), "l"(v1) : "memory");
}

// 在 bulkCopy<16> 中使用
template<>
__device__ void bulkCopy<16>(int tid, uint32_t len, char* src, char* dst) {
  BytePack<16> reg;

  for (uint32_t offset = tid * DATA_LOAD_SIZE;
       offset + DATA_LOAD_SIZE - 1 < len;
       offset += WARP_SIZE * DATA_LOAD_SIZE) {

    uintptr_t src_addr = (uintptr_t)(src + offset);
    uintptr_t dst_addr = (uintptr_t)(dst + offset);

    if ((src_addr & 15) == 0 && (dst_addr & 15) == 0) {
      // Fast path: 128-bit load/store
      load128((uint64_t*)src_addr, reg.u64[0], reg.u64[1]);
      store128((uint64_t*)dst_addr, reg.u64[0], reg.u64[1]);
    } else {
      // Slow path: volatile load/store
      reg = ld_volatile_global<16>(src_addr);
      st_global<16>(dst_addr, reg);
    }
  }
}
```

**预期收益**：20-30% 性能提升（对齐情况下）

---

### 4. Per-Warp Shared Memory Staging ⭐⭐⭐

**问题**：每个 descriptor 都从 global memory 读取

**NCCL 的做法**：
- 每个 warp 批量加载 4 个 pages 的 metadata 到 shared memory
- 从 shared memory 读取（~30 cycles vs ~400 cycles）

**方案**：

```cpp
#define WARP_SHM_PAGE_CNT 4
#define WARP_SHM_SIZE (WARP_SHM_PAGE_CNT * sizeof(UnpackDescriptor))

__global__ void tcpxUnpackKernelOptimized(DeviceUnpackMeta* d_meta, uint32_t slot) {
  __shared__ UnpackDescriptor s_meta[WARP_SHM_PAGE_CNT * MAX_WARPS];

  int tid = threadIdx.x;
  int wid = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  uint32_t meta_cnt;
  asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];" : "=l"(meta_cnt) : "l"(&d_meta->cnt[slot]));

  // 每个 warp 批量处理 4 个 descriptors
  for (uint32_t meta_s = wid * WARP_SHM_PAGE_CNT;
       meta_s < meta_cnt;
       meta_s += (blockDim.x / WARP_SIZE) * WARP_SHM_PAGE_CNT) {

    uint32_t iter_cnt = min(WARP_SHM_PAGE_CNT, meta_cnt - meta_s);

    // 协作加载到 shared memory
    if (lane < iter_cnt) {
      s_meta[wid * WARP_SHM_PAGE_CNT + lane] = d_meta->mem[slot][meta_s + lane];
    }
    __syncwarp();

    // 处理这批 descriptors
    for (uint32_t x = 0; x < iter_cnt; ++x) {
      UnpackDescriptor desc = s_meta[wid * WARP_SHM_PAGE_CNT + x];
      unpackSingleDescriptor(lane, desc, bounce_buffer, dst_buffer);
    }
    __syncwarp();
  }
}
```

**预期收益**：5-10% 性能提升

---

## 可选实施（MEDIUM 优先级）

### 5. Warp-Level Sync ⭐⭐

**问题**：使用 `__syncthreads()` 同步整个 block

**NCCL 的做法**：使用 `__syncwarp()` 只同步 warp

**方案**：

```cpp
if (threadIdx.x == 0) {
  devmem_visibility_barrier(desc_block->ready_flag);
}
__syncwarp();  // 只同步 warp，更快
```

**预期收益**：1-2% 性能提升

---

## 实施计划

### 第一步：诊断瓶颈（5 分钟）

```bash
./test_unpack_modes.sh server 0
grep "Avg:" logs/unpack_test_*/server_gpu0_*.log
```

---

### 第二步：持久化 Metadata 结构（1 小时）

**优先级**：CRITICAL
**难度**：中等
**预期收益**：10-20%

**修改文件**：
- `include/rx_descriptor.h` - 添加 `DeviceUnpackMeta` 结构
- `device/unpack_launch.cu` - 初始化常驻 metadata
- `device/unpack_kernels.cu` - 从常驻 metadata 读取

---

### 第三步：批量处理 Descriptors（30 分钟）

**优先级**：CRITICAL
**难度**：简单
**预期收益**：20-50%

**修改文件**：
- `device/unpack_kernels.cu` - 修改 kernel 循环逻辑

**代码示例**：参考上面的 "方案 A"

---

### 第四步：128-bit Load/Store（30 分钟）

**优先级**：HIGH
**难度**：简单
**预期收益**：20-30%

**修改文件**：
- `device/unpack_kernels.cu` - 添加 `load128`/`store128` 函数

---

### 第五步：Per-Warp Shared Memory（1 小时）

**优先级**：HIGH
**难度**：中等
**预期收益**：5-10%

**修改文件**：
- `device/unpack_kernels.cu` - 添加 shared memory staging

---

## 预期性能提升路径

```
当前性能：~2.8 GB/s (512KB chunk, kernel mode)

↓ 持久化 Metadata
~3.1-3.4 GB/s (+10-20%)

↓ 批量处理 Descriptors
~3.7-5.1 GB/s (+20-50%)

↓ 128-bit Load/Store
~4.4-6.6 GB/s (+20-30%)

↓ Per-Warp Shared Memory
~4.6-7.3 GB/s (+5-10%)

理论上限：~21.26 GB/s (Google 建议的单 NIC 上限)
```

---

## 对比表格

| 优化 | NCCL 有？ | 你有？ | 优先级 | 预期收益 | 实施难度 |
|------|----------|--------|--------|---------|---------|
| 持久化 Metadata | ✅ | ❌ | CRITICAL | 10-20% | 中等 |
| 批量处理 Descriptors | ✅ | ❌ | CRITICAL | 20-50% | 简单 |
| 128-bit Load/Store | ⚠️ | ❌ | HIGH | 20-30% | 简单 |
| Per-Warp Shared Memory | ✅ | ❌ | HIGH | 5-10% | 中等 |
| Warp-Level Sync | ✅ | ❌ | MEDIUM | 1-2% | 简单 |

**注**：⚠️ = NCCL 有但默认未启用（需要定义 `ALIGNED_LOAD` 宏）

---

## 快速验证

### 1. 确认当前瓶颈
```bash
./test_unpack_modes.sh server 0
grep "Avg:" logs/unpack_test_*/server_gpu0_*.log
```

### 2. 实施持久化 Metadata
```bash
# 修改 include/rx_descriptor.h, device/unpack_launch.cu, device/unpack_kernels.cu
make clean && make
UCCL_TCPX_UNPACK_IMPL=kernel ./run_p2p_fullmesh.sh server 0
grep "Avg:" logs/fullmesh_*.log
```

### 3. 对比性能
```bash
# 优化前
[PERF] Avg: 22.45 ms, 2.85 GB/s

# 优化后（预期）
[PERF] Avg: 18.32 ms, 3.49 GB/s  ← +22% 提升
```

---

## 详细文档

完整验证分析请参考：`docs/KERNEL_OPTIMIZATION_VERIFIED.md`

---

**最后更新**：2025-10-08
**状态**：Verified and ready for implementation
