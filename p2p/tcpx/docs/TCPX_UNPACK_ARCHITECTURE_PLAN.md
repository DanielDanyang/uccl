# TCPX Unpack 架构分析与更新计划

## 📋 **背景与问题分析**

根据 GCP NCCL-TCPX 团队的回复，我们现在明确了问题的根本原因：

### 🔍 **核心问题**
1. **TCPX插件不直接写入用户缓冲区**：`tcpxIrecv_v5` 不会直接将数据写入 `void *data` 字段
2. **需要NCCL的unpack内核**：数据接收后需要NCCL的设备端内核来执行"unpack"操作
3. **我们的测试缺少unpack步骤**：独立测试中没有运行NCCL的设备端内核，导致GPU缓冲区保持未修改状态

### 🏗️ **TCPX接收架构**
```
网络数据 → NIC → GPU内存页面 → unpack队列元数据 → NCCL设备内核 → 用户缓冲区
```

## 🔬 **技术架构分析**

### 1. **NCCL Unpack 内核机制**

根据 NCCL 源码分析：
- **位置**: `src/device/network/unpack/`
- **调用点**: `src/device/prims_simple.h:242`
- **功能**: 将分散的数据包缓冲区列表复制到连续的用户提供的张量缓冲区

### 2. **Linux Kernel DevMem-TCP API**

根据 Linux 内核文档 (https://docs.kernel.org/networking/devmem.html)：
- **核心机制**: 通过 `recvmsg()` 系统调用的 `cmsg` 传递分散列表
- **控制消息类型**:
  - `SCM_DEVMEM_DMABUF`: 数据落在dmabuf中
  - `SCM_DEVMEM_LINEAR`: 数据落在线性缓冲区中
- **数据结构**: `struct dmabuf_cmsg` 包含偏移量、大小和令牌信息

### 3. **TCPX插件实现细节**

基于源码分析：
- **GPU接收路径**: `gpudirectTCPXRecv()` → `process_recv_cmsg()` → unpack队列
- **元数据处理**: 将 `scatter_list` 复制到 `unpack_slot.mem`
- **设备队列**: 使用 `tcpxNetDeviceQueue` 管理unpack任务

## 🎯 **解决方案设计**

### **方案A: 实现自定义Unpack内核 (推荐)**

#### 优势
- 完全控制unpack逻辑
- 可以优化性能
- 独立于NCCL内部实现

#### 实现步骤
1. **解析unpack元数据**：从TCPX插件获取scatter_list
2. **编写CUDA内核**：实现数据从分散页面到连续缓冲区的复制
3. **集成到测试**：在接收完成后调用自定义unpack内核

### **方案B: 使用HOST内存接收 (临时方案)**

#### 优势
- 绕过GPU unpack队列问题
- 快速验证TCPX传输功能
- 实现简单

#### 限制
- 无法验证真正的GPU Direct传输
- 性能不是最优的

## 📝 **详细实现计划**

### **阶段1: 理解和验证 (1-2天)**

#### 1.1 深入分析NCCL unpack机制
- [ ] 获取NCCL unpack内核源码
- [ ] 分析 `loadMeta` 数据结构
- [ ] 理解scatter-gather到连续内存的映射

#### 1.2 分析TCPX unpack队列
- [ ] 研究 `unpack_slot` 数据结构
- [ ] 理解 `scatter_list` 格式
- [ ] 分析元数据传递机制

#### 1.3 验证DevMem-TCP API
- [ ] 研究Linux内核devmem-tcp文档
- [ ] 分析 `struct dmabuf_cmsg` 结构
- [ ] 理解令牌管理机制

### **阶段2: 设计自定义Unpack内核 (2-3天)**

#### 2.1 数据结构设计
```c
// 自定义unpack元数据结构
struct CustomUnpackMeta {
    uint32_t src_offset;    // 源页面偏移
    uint32_t dst_offset;    // 目标缓冲区偏移  
    uint32_t length;        // 数据长度
    void* src_page_ptr;     // 源页面指针
};

// Unpack任务描述符
struct UnpackTask {
    void* dst_buffer;              // 目标连续缓冲区
    CustomUnpackMeta* meta_list;   // 元数据列表
    int meta_count;                // 元数据数量
    cudaStream_t stream;           // CUDA流
};
```

#### 2.2 CUDA内核设计
```cuda
__global__ void customUnpackKernel(
    void* dst_buffer,
    CustomUnpackMeta* meta_list,
    int meta_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= meta_count) return;
    
    CustomUnpackMeta meta = meta_list[idx];
    
    // 执行内存复制：从分散页面到连续缓冲区
    memcpy_async(
        (char*)dst_buffer + meta.dst_offset,
        (char*)meta.src_page_ptr + meta.src_offset,
        meta.length
    );
}
```

### **阶段3: 集成到TCPX测试 (2-3天)**

#### 3.1 修改测试代码
- [ ] 在 `tcpxItest` 完成后获取unpack元数据
- [ ] 调用自定义unpack内核
- [ ] 验证数据完整性

#### 3.2 元数据提取接口
```c
// 从TCPX请求中提取unpack元数据
int extractUnpackMetadata(
    struct tcpxRequest* request,
    CustomUnpackMeta** meta_list,
    int* meta_count
);

// 执行自定义unpack操作
int executeCustomUnpack(
    void* dst_buffer,
    CustomUnpackMeta* meta_list,
    int meta_count,
    cudaStream_t stream
);
```

### **阶段4: 性能优化与测试 (1-2天)**

#### 4.1 性能优化
- [ ] 优化CUDA内核参数
- [ ] 实现异步执行
- [ ] 添加错误处理

#### 4.2 全面测试
- [ ] 单元测试：验证unpack内核正确性
- [ ] 集成测试：端到端TCPX传输
- [ ] 性能测试：与标准NCCL对比

## 🔧 **需要修改的文件**

### 1. **测试代码修改**
- `p2p/tcpx/tests/test_tcpx_transfer.cc`
  - 添加自定义unpack内核调用
  - 修改验证逻辑

### 2. **新增文件**
- `p2p/tcpx/src/custom_unpack.cu` - 自定义unpack CUDA内核
- `p2p/tcpx/src/custom_unpack.h` - 接口定义
- `p2p/tcpx/src/metadata_extractor.cc` - 元数据提取工具

### 3. **构建系统修改**
- `p2p/tcpx/CMakeLists.txt` - 添加CUDA编译支持

## 🎯 **成功标准**

1. **功能验证**: 自定义unpack内核能正确将分散数据复制到连续缓冲区
2. **数据完整性**: 接收到的数据与发送的数据完全一致
3. **性能基准**: 传输性能接近或超过标准NCCL实现
4. **稳定性**: 在各种数据大小和模式下稳定工作

## 🚀 **下一步行动**

1. **立即开始**: 深入研究NCCL unpack内核源码
2. **并行进行**: 分析TCPX插件的unpack队列机制
3. **快速原型**: 实现基础的自定义unpack内核
4. **迭代优化**: 根据测试结果持续改进

这个计划将使我们能够真正理解和解决TCPX GPU接收路径的问题，实现高性能的GPU Direct网络传输。

## 📚 **技术参考资料**

### **NCCL相关**
- NCCL unpack内核: `https://github.com/NVIDIA/nccl/tree/master/src/device/network/unpack`
- NCCL primitives: `https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h#L242`
- NCCL设备代码: `https://github.com/NVIDIA/nccl/tree/master/src/device`

### **Linux Kernel DevMem-TCP**
- 官方文档: `https://docs.kernel.org/networking/devmem.html`
- 内核实现: `https://www.kernel.org/doc/Documentation/networking/`
- 测试代码: `tools/testing/selftests/drivers/net/hw/ncdevmem.c`

### **TCPX插件**
- Google实现: `https://github.com/google/nccl-plugin-gpudirecttcpx`
- 接收路径: `src/sock/tcpx.h:230` (gpudirectTCPXRecv)
- 控制消息处理: `src/sock/tcpx.h:136` (process_recv_cmsg)

## 🔍 **关键数据结构分析**

### **1. TCPX LoadMeta结构**
```c
// 基于TCPX插件源码分析
union loadMeta {
    struct {
        uint32_t src_off;    // 源偏移量
        uint32_t len;        // 数据长度
        uint64_t dst_off;    // 目标偏移量
    };
    // 可能还有其他字段...
};
```

### **2. Linux DevMem结构**
```c
// 基于Linux内核文档
struct dmabuf_cmsg {
    __u32 frag_offset;   // 片段在dmabuf中的偏移
    __u32 frag_size;     // 片段大小
    __u32 frag_token;    // 用于释放的令牌
    __u32 dmabuf_id;     // dmabuf标识符
};
```

### **3. TCPX Unpack Slot**
```c
// 基于TCPX插件源码
struct unpackSlot {
    void* mem;              // unpack元数据内存
    uint64_t idx;           // 队列索引
    size_t cnt_cache;       // 缓存计数
    bool active;            // 是否活跃
    // 其他字段...
};
```

# TCPX Device-Unpack Architecture (Plan A)

Goal
- Reuse NCCL’s network unpack kernel logic to gather a scatter-list of received TCPX packet buffers directly into a user-provided contiguous GPU buffer.
- Bypass NCCL’s higher-level device kernels (do NOT use `prims_simple.h` or NCCL scheduling). We will adapt the unpack kernel interface and provide our own descriptor structures.
- Keep a host-recv fallback for bring-up and debug.

Context
- In `nccl-plugin-gpudirecttcpX`, `tcpxIrecv_v5` does not write directly into the user’s `void* data` when using CUDA buffers. NCCL typically runs a device-side “unpack” kernel that copies from a scatter-list (delivered via devmem-tcp ancillary data) into user memory.
- Upstream references shared by GCP authors:
  - NCCL unpack kernels: `src/device/network/unpack/` (NCCL repo)
  - NCCL kernels call sites: `src/device/prims_simple.h` (we will not use this; only the unpack logic is needed)
  - devmem-tcp ancillary (cmsg) carries scatter segments to userspace; Google’s plugin references: `src/sock/tcpx.h`
  - Upstream kernel doc: `Documentation/networking/devmem.rst`

What We Will Build
1) A device-side unpack kernel (ported/adapted from NCCL’s network/unpack) that reads a GPU-visible descriptor array describing the packet fragments, and writes into the contiguous destination buffer.
2) A host-side pipeline that, upon `tcpxTest` completion, retrieves or derives the scatter-list for a completed receive and builds GPU descriptors, then launches the unpack kernel, and finally calls `tcpxIrecvConsumed`.
3) A host-recv fallback (NCCL_PTR_HOST) for environments where devmem-tcp/GPU path is not ready.

Key Design Choices
- No NCCL dependency: we will copy/adapt the minimal code we need into `p2p/tcpx/device/` and define our own descriptor types in `p2p/tcpx/rx/`.
- Two sources for scatter information (implementation-time decision):
  - Preferred: consume the devmem-tcp scatter from ancillary data (cmsg) associated with the recv’d request (see `nccl-plugin-gpudirecttcpx/src/sock/tcpx.h` for how the plugin interprets cmsg). This requires a hook/adapter in our userspace to access what the plugin already parsed or to replicate that parsing.
  - Alternative (depending on plugin internals): use `tcpxGetDeviceHandle` (queue exported to GPU) and derive fragment positions from the device queue metadata if exposed. This option depends on the plugin’s device-queue API stability.

File/Module Layout (to be added)
- `p2p/tcpx/rx/`
  - `rx_cmsg_parser.h/.cc`: Parse devmem-tcp cmsg (ancillary data) into a normalized host scatter-list.
  - `rx_descriptor.h`: GPU-visible descriptor PODs (aligned) and conversion helpers.
  - `rx_staging.h/.cc`: Manage pinned host staging for descriptors or device copies of descriptor blocks.
- `p2p/tcpx/device/`
  - `unpack_kernels.cu`: Adapted NCCL unpack logic (vectorized copies from fragments → dst buffer).
  - `unpack_launch.h/.cc`: Kernel launch, stream selection, basic error prop.
- `p2p/tcpx/pipeline/`
  - `rx_pipeline.h/.cc`: Orchestrates irecv → test(done) → parse/derive scatter → build descriptors → launch unpack → record CUDA event → irecvConsumed.
  - `opts.h`: Env flags and tunables.
- `p2p/tcpx/reference/unpack/` (already present)
  - `unpack.h`, `unpack_defs.h`: Source references for porting (read-only references; do not depend on NCCL headers at compile time).

Existing Files We Will Touch Integrate With (no code change in this step)
- `nccl-plugin-gpudirecttcpx/src/net_tcpx.cc` and `.../src/sock/tcpx.h`: Guidance for how cmsg/devmem-tcp scatter is represented.
- `nccl-plugin-gpudirecttcpx/src/net_tcpx.h`: API surface. We may add wrappers in our interface for `getDeviceHandle`/`getDeviceMr` if needed.
- `p2p/tcpx/tcpx_impl.cc` and `p2p/tcpx/tcpx_interface.h`: Will eventually expose a high-level “device-unpack receive” operation (or a flag to turn it on) without changing public test APIs yet.
- `p2p/tcpx/tests/test_tcpx_transfer.cc`: Once the pipeline exists, switch the CUDA receive path to “post → unpack → validate” using our pipeline (host fallback remains for debug).

Descriptor Specifications
Host-normalized scatter list (built from cmsg):
```c
// Host side (neutral format)
typedef struct {
  uint64_t dev_addr;  // device-visible source pointer from devmem-tcp
  uint32_t len;
  uint32_t _pad;
} RxFrag;

typedef struct {
  uint32_t nfrags;
  uint32_t total_len;
  RxFrag   frags[MAX_FRAGS];
} RxList;
```

GPU-visible descriptors (what the kernel consumes):
```c
// Device side (contiguous descriptors copied/mapped to GPU)
typedef struct {
  const uint8_t* src; // device-visible src pointer (from dev_addr)
  uint32_t       len;
  uint32_t       dst_off; // offset within dst buffer
} GpuFrag;

typedef struct {
  uint32_t nfrags;
  uint32_t total_len;
  // Followed by nfrags entries
  GpuFrag  frags[];
} GpuList;
```

Kernel Interface (adapted)
```c++
// device/unpack_kernels.cu
__global__ void tcpx_unpack_kernel(const GpuFrag* __restrict__ list,
                                   uint8_t* __restrict__ dst,
                                   int nfrags) {
  // Baseline: per-fragment CTAs or per-chunk grid; vectorized loads/stores.
  // Copy list[fi].src[0..len) → dst + list[fi].dst_off
}
```

Receive Pipeline (Server)
1) `tcpxIrecv_v5(recv_comm, ..., size=payload, tag)` — as today.
2) Poll `tcpxTest` until `done`.
3) Obtain scatter list for this request:
   - Preferred: parse devmem-tcp cmsg associated with the receive (see `sock/tcpx.h` patterns). If the plugin hides the cmsg fully, add a thin adapter to expose the parsed result.
   - Alternative: derive from device queue handle (`tcpxGetDeviceHandle`) if the queue exposes fragment metadata in device memory accessible to our kernel.
4) Build `GpuList` with dst offsets; copy/make visible to GPU.
5) Launch `tcpx_unpack_kernel` on a CUDA stream; record event.
6) On event completion: call `tcpxIrecvConsumed(recv_comm, 1, request)`; signal upper layer (and send app-level ACK if used).

Send Pipeline (Client)
- Unchanged: `tcpxRegMr(CUDA)` → payload upload to device → `tcpxIsend` → `tcpxTest`.
- Ensure a fence after the payload upload (as we already do) before starting zero-copy.

Fallback (Host-Staged)
- If `NIXL_TCPX_USE_DEVUNPACK=0` or no devmem support:
  - Receive into page-aligned host memory (NCCL_PTR_HOST) and memcpy/cudaMemcpyHtoD to the user dst buffer.

Env/Tunables
- `NIXL_TCPX_USE_DEVUNPACK=1` (default in production)
- `NIXL_TCPX_MAX_FRAGS` (bounds on descriptors; coalesce if exceeded)
- `NIXL_TCPX_COALESCE_THRESHOLD` (bytes; small fragments can be merged by staging)
- `NCCL_MIN_ZCOPY_SIZE`, `NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE` (transport behavior)
- `NCCL_GPUDIRECTTCPX_RECV_SYNC` (useful during debug)
- `NIXL_TCPX_DEBUG=1` (pipeline logs)

Implementation Steps (Milestones)
1) Kernel Port (no NCCL includes):
   - Copy minimal logic from `p2p/tcpx/reference/unpack/` into `device/unpack_kernels.cu`.
   - Replace NCCL types/macros with our own in `rx_descriptor.h`.
   - Provide a simple kernel that iterates `GpuFrag` entries and performs vectorized copies.
2) Descriptor Path:
   - Implement `rx_cmsg_parser` to normalize devmem-tcp scatter into `RxList`.
   - Implement conversion to `GpuList` with computed `dst_off`.
   - Implement `rx_staging` to place descriptors in pinned memory (or cudaMemcpy to device).
3) Launch + Orchestration:
   - Implement `unpack_launch` to launch kernels and manage streams/events.
   - Implement `rx_pipeline` to glue: irecv → test → parse → build → launch → event → irecvConsumed.
4) Integrate With Tests:
   - In `tests/test_tcpx_transfer.cc`, for CUDA receive path, replace direct `cuMemcpyDtoH` verification with: wait → run `rx_pipeline` → verify dst buffer → ACK.
   - Keep host-recv toggle for A/B.
5) Integrate With UCCL/NIXL:
   - Add a thin adapter (e.g., `nixl_tcpx_engine`) that uses `rx_pipeline` for receives under the NIXL API exposed by `uccl_engine.h`.
6) Validation + Perf:
   - Start with large payloads, verify correctness, add metrics, then tune kernel.

Notes on Accessing Scatter Data
- The devmem-tcp cmsg is the authoritative source for packet fragment locations (as pointed by GCP authors). If the current plugin encapsulates cmsg parsing, we will add a small adapter at the plugin/user boundary to surface a normalized scatter list. As an alternative, if the plugin’s device queue already holds the scatter metadata in GPU-visible memory, we can read from that in our kernel directly (requires confirming `tcpxGetDeviceHandle` semantics).

Risks & Mitigations
- Kernel ABI drift (devmem-tcp): abstract via `rx_cmsg_parser` with compile-time guards.
- Many tiny fragments: coalesce on host or stage into a temp device buffer to reduce descriptor count.
- Synchronization: maintain fences before send and after receive completion; verify visibility with tests.

Deliverables (This Phase)
- This document, adapted kernel interfaces, descriptor specs, and a clear file map.
- No code changes yet; next task is to add the skeleton files above and wire the pipeline under a feature flag (方案 A).
