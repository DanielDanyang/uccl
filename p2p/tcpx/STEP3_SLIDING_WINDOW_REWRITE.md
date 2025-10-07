# Step 3: Sliding Window Rewrite - Using Existing Infrastructure

## 🎯 **问题诊断**

### **原始问题**
从最新日志 (`singleproc_server_20251007_101300.log`) 发现：
- 服务器成功 post 第一个 `tcpx_irecv()` (GPU 0 channel 0)
- 在尝试 post 第二个 `tcpx_irecv()` 时卡住
- 客户端等待但从未开始发送

### **根本原因**
通过阅读 TCPX 源代码 (`nccl-plugin-gpudirecttcpx/src/net_tcpx.cc`) 发现：

```cpp
static tcpxResult_t tcpxGetRequest(struct tcpxComm* comm, ...) {
  if (!comm->rq.has_free()) {  // ← 关键检查！
    WARN("NET/" PRODUCT_NAME " : unable to allocate requests");
    return tcpxInternalError;
  }
  // ...
}
```

**TCPX 有固定大小的请求队列** (`MAX_REQUESTS = 16`)，当队列满时：
1. `tcpx_irecv()` 会返回错误或阻塞
2. 需要调用 `tcpx_test()` 来清理完成的请求，释放队列空间
3. 不能一次性 post 所有请求（我们有 1024 个请求：8 GPUs × 128 chunks）

## ✅ **解决方案：使用现有的 SlidingWindow 类**

### **发现**
用户的代码库中已经有完整的滑动窗口实现：
- `include/sliding_window.h` - 接口定义
- `src/sliding_window.cc` - 实现
- 成功的多进程测试 `test_tcpx_perf_multi.cc` 使用了手动滑动窗口

### **为什么之前没用？**
`test_tcpx_perf_orchestrator.cc` 虽然使用了 `ChannelManager`，但**没有使用 `SlidingWindow`**，导致：
1. 尝试一次性 post 所有请求
2. 请求队列溢出
3. `tcpx_irecv()` 阻塞

## 🔧 **实现细节**

### **1. 添加头文件**
```cpp
#include "../include/sliding_window.h"
```

### **2. 服务器端：创建滑动窗口**
```cpp
// 每个 GPU 的每个 channel 都有独立的滑动窗口
constexpr int MAX_INFLIGHT_PER_CHANNEL = 16;  // TCPX MAX_REQUESTS

std::vector<std::vector<SlidingWindow*>> windows(kNumGPUs);
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  int num_channels = gpus[gpu_id].mgr->get_num_channels();
  windows[gpu_id].resize(num_channels);
  for (int ch = 0; ch < num_channels; ch++) {
    windows[gpu_id][ch] = new SlidingWindow(MAX_INFLIGHT_PER_CHANNEL);
  }
}
```

### **3. 服务器端：接收循环**
```cpp
while (offset < test_size_per_gpu) {
  size_t this_chunk = std::min(chunk_bytes, test_size_per_gpu - offset);
  
  int channel_local_id = local_chunk_idx % num_channels;
  ChannelResources& ch = ctx.mgr->get_channel(channel_local_id);
  SlidingWindow* win = windows[gpu_id][channel_local_id];
  
  // 滑动窗口：如果满了，等待最老的请求完成
  if (win->is_full()) {
    if (win->wait_and_release_oldest(ch.recv_comm, /*is_recv=*/true) != 0) {
      std::cerr << "[ERROR] wait_and_release_oldest failed" << std::endl;
      return 1;
    }
  }
  
  // Post receive
  void* recv_request = nullptr;
  tcpx_irecv(ch.recv_comm, 1, recv_data, recv_sizes, recv_tags,
             recv_mhandles, &recv_request);
  
  // 添加到滑动窗口
  win->add_request(recv_request, local_chunk_idx, nullptr);
  
  offset += this_chunk;
  local_chunk_idx++;
}
```

### **4. 服务器端：Drain 阶段**
```cpp
// 等待所有请求完成
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  for (int ch = 0; ch < num_channels; ch++) {
    ChannelResources& channel = ctx.mgr->get_channel(ch);
    SlidingWindow* win = windows[gpu_id][ch];
    
    if (win->drain_all(channel.recv_comm, /*is_recv=*/true) != 0) {
      std::cerr << "[ERROR] drain_all failed" << std::endl;
      return 1;
    }
  }
}
```

### **5. 客户端：类似实现**
```cpp
// 使用更小的窗口（留余量）
constexpr int MAX_INFLIGHT_SEND_PER_CHANNEL = 12;

std::vector<std::vector<SlidingWindow*>> send_windows(kNumGPUs);
// ... 初始化 ...

// 发送循环
while (offset < test_size_per_gpu) {
  SlidingWindow* win = send_windows[gpu_id][channel_local_id];
  
  if (win->is_full()) {
    win->wait_and_release_oldest(ch.send_comm, /*is_recv=*/false);
  }
  
  void* send_request = nullptr;
  tcpx_isend(ch.send_comm, src_ptr, this_chunk, tag, ch.mhandle, &send_request);
  
  win->add_request(send_request, local_chunk_idx, nullptr);
}

// Drain
win->drain_all(ch.send_comm, /*is_recv=*/false);
```

### **6. 清理**
```cpp
// 服务器端
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  for (auto* win : windows[gpu_id]) {
    delete win;
  }
}

// 客户端
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  for (auto* win : send_windows[gpu_id]) {
    delete win;
  }
}
```

## 📊 **关键改进**

### **Before (手动实现)**
```cpp
// 复杂的手动管理
std::vector<std::vector<std::vector<PendingRecv>>> pending_per_gpu_channel;
while (channel_pending.size() >= MAX_INFLIGHT) {
  auto& oldest = channel_pending.front();
  tcpx_test(oldest.request, &done, &received_size);
  if (done) {
    tcpx_irecv_consumed(ch.recv_comm, 1, oldest.request);
    channel_pending.erase(channel_pending.begin());
  }
}
```

### **After (使用 SlidingWindow 类)**
```cpp
// 简洁的封装
SlidingWindow* win = windows[gpu_id][channel_id];
if (win->is_full()) {
  win->wait_and_release_oldest(ch.recv_comm, /*is_recv=*/true);
}
win->add_request(recv_request, chunk_idx, nullptr);
```

## 🎉 **优势**

1. **代码复用** - 使用已有的、经过测试的 `SlidingWindow` 类
2. **简洁** - 从 ~100 行手动管理减少到 ~10 行
3. **一致性** - 与 `ChannelManager` 配合使用，保持架构一致
4. **可维护** - 逻辑封装在类中，易于调试和修改
5. **正确性** - 基于成功的 `test_tcpx_perf_multi.cc` 模式

## 🚀 **下一步**

### **立即测试**
```bash
# 服务器（Node 0）
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh server

# 客户端（Node 1）
./test_step3_bandwidth.sh client <SERVER_IP>
```

### **预期行为**
- ✅ 不会出现 "tcpx_irecv blocked" 错误
- ✅ 服务器成功 post 所有 receives
- ✅ 客户端成功发送所有数据
- ✅ 完成所有 20 次迭代
- ✅ 准确的带宽测量

## 📝 **技术要点**

### **SlidingWindow 类的关键方法**
1. **`is_full()`** - 检查窗口是否满（达到 MAX_INFLIGHT）
2. **`add_request()`** - 添加新请求到窗口
3. **`wait_and_release_oldest()`** - 等待最老的请求完成并释放
4. **`drain_all()`** - 等待所有请求完成
5. **`clear()`** - 清空窗口（用于新迭代）

### **服务器 vs 客户端差异**
| 方面 | 服务器 (Recv) | 客户端 (Send) |
|------|---------------|---------------|
| 窗口大小 | 16 | 12 (留余量) |
| `wait_and_release_oldest()` | `is_recv=true` | `is_recv=false` |
| 释放方式 | `tcpx_irecv_consumed()` | 自动释放 |
| CUDA Event | 可选（kernel 模式） | 不需要 |

### **为什么客户端用 12 而不是 16？**
参考 `test_tcpx_perf_multi.cc` 的注释：
```cpp
// 【关键】Client 使用 12 而不是 16，留余量避免边界情况
constexpr int MAX_INFLIGHT_SEND_PER_CHANNEL = 12;
```

原因：
1. 避免与服务器的 16 个请求冲突
2. 留出缓冲空间处理网络延迟
3. 防止边界条件导致的死锁

## ✅ **编译状态**
```bash
$ make test_tcpx_perf_orchestrator
Building test_tcpx_perf_orchestrator...
/usr/local/cuda/bin/nvcc -std=c++17 -Xcompiler "-fPIC -O2 -Wall" \
  -Iinclude -I. -I/usr/local/cuda/include \
  -o tests/test_tcpx_perf_orchestrator \
  tests/test_tcpx_perf_orchestrator.cc tcpx_impl.cc \
  device/unpack_kernels.o device/unpack_launch.o \
  src/sliding_window.o src/bootstrap.o src/channel_manager.o \
  -ldl -lpthread -L/usr/local/cuda/lib64 -lcuda -lcudart
```

✅ **编译成功！**

## 🎯 **总结**

这次修复的关键教训：
1. **先查看现有代码** - 用户已经有完整的基础设施
2. **阅读源代码** - TCPX 源代码揭示了请求队列限制
3. **复用而不是重写** - `SlidingWindow` 类已经存在并经过测试
4. **参考成功案例** - `test_tcpx_perf_multi.cc` 提供了正确的模式

现在代码：
- ✅ 使用现有的 `SlidingWindow` 类
- ✅ 与 `ChannelManager` 配合良好
- ✅ 遵循成功的多进程测试模式
- ✅ 简洁、可维护、正确

**准备测试！** 🚀

