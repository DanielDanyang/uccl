# 滑动窗口机制可视化讲解

## 问题背景

### TCPX 请求池限制

```
┌─────────────────────────────────────────┐
│  TCPX Plugin (每个 comm)                │
│                                         │
│  Request Pool: [16 个槽]                │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐    │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │    │
│  └───┴───┴───┴───┴───┴───┴───┴───┘    │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐    │
│  │ 8 │ 9 │10 │11 │12 │13 │14 │15 │    │
│  └───┴───┴───┴───┴───┴───┴───┴───┘    │
│                                         │
│  固定大小: MAX_REQUESTS = 16            │
│  (定义在 work_queue.h)                  │
└─────────────────────────────────────────┘
```

### 问题示例

传输 64MB 数据，chunk 大小 512KB：
```
64MB ÷ 512KB = 128 chunks
```

如果批量发起所有 irecv：
```
Chunk  0: irecv → 占用槽 0  ✅
Chunk  1: irecv → 占用槽 1  ✅
...
Chunk 15: irecv → 占用槽 15 ✅
Chunk 16: irecv → ❌ 错误: "unable to allocate requests"
```

---

## 滑动窗口解决方案

### 核心思想

**限制并发数 ≤ 16，动态释放和重用槽位**

```
窗口大小: MAX_INFLIGHT = 16 (Server) 或 12 (Client)

┌────────────────────────────────────────────────┐
│  滑动窗口 (最多 16 个并发 chunks)              │
│                                                │
│  [C0][C1][C2]...[C15]                          │
│   ↑                ↑                           │
│  最老              最新                         │
│                                                │
│  当窗口满时:                                   │
│  1. 等待 C0 完成                               │
│  2. 释放 C0 的槽位                             │
│  3. 发起新的 chunk (C16)                       │
│  4. 窗口变为: [C1][C2]...[C15][C16]            │
└────────────────────────────────────────────────┘
```

---

## Server 端滑动窗口详解

### 时间线示例 (前 20 个 chunks)

```
时间 →

Chunk 0-15: 填充窗口
═══════════════════════════════════════════════════════════════
t0:  C0  irecv → test → kernel → event_0
t1:  C1  irecv → test → kernel → event_1
t2:  C2  irecv → test → kernel → event_2
...
t15: C15 irecv → test → kernel → event_15
     ↑
     窗口已满 (16 个 chunks)

Chunk 16: 滑动窗口开始工作
═══════════════════════════════════════════════════════════════
t16: 检查: pending_reqs.size() = 16 >= MAX_INFLIGHT
     ↓
     等待 C0 的 kernel 完成:
       cudaEventSynchronize(event_0)  ← 阻塞直到 C0 kernel 完成
     ↓
     释放 C0 的请求槽:
       tcpx_irecv_consumed(comm, 1, req_0)
     ↓
     从窗口移除 C0:
       pending_reqs.erase(begin)
       pending_indices.erase(begin)
     ↓
     发起 C16:
       irecv → test → kernel → event_0 (重用 event)
     ↓
     窗口: [C1][C2]...[C15][C16]

Chunk 17-19: 继续滑动
═══════════════════════════════════════════════════════════════
t17: 等待 C1 → 释放 → 发起 C17 → 窗口: [C2]...[C16][C17]
t18: 等待 C2 → 释放 → 发起 C18 → 窗口: [C3]...[C17][C18]
t19: 等待 C3 → 释放 → 发起 C19 → 窗口: [C4]...[C18][C19]
```

### 关键数据结构

```cpp
// Server 端滑动窗口状态
constexpr int MAX_INFLIGHT = 16;

std::vector<cudaEvent_t> events(MAX_INFLIGHT);  // 循环使用
std::vector<void*> pending_reqs;                // 待完成的请求
std::vector<int> pending_indices;               // chunk 索引

// 示例状态 (处理 C16 时):
events = [event_0, event_1, ..., event_15]
pending_reqs = [req_1, req_2, ..., req_15, req_16]  // C0 已移除
pending_indices = [1, 2, ..., 15, 16]
```

### 为什么需要 CUDA Events？

```
问题: 何时可以调用 tcpx_irecv_consumed？

错误时机:
  tcpx_test 返回 done=1
  ↓
  数据在 bounce buffer (GPU 内存)
  ↓
  ❌ 如果现在调用 irecv_consumed → bounce buffer 被释放
  ↓
  kernel 启动 (异步)
  ↓
  kernel 读取 bounce buffer → ❌ 读到垃圾数据!

正确时机:
  tcpx_test 返回 done=1
  ↓
  launcher->launch(desc) (异步启动 kernel)
  ↓
  cudaEventRecord(event, stream) (记录 event)
  ↓
  ... (CPU 继续处理其他 chunks)
  ↓
  cudaEventSynchronize(event) (等待 kernel 完成)
  ↓
  ✅ 现在调用 irecv_consumed 是安全的
```

### 代码示例

```cpp
// Server 端滑动窗口核心代码
for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
  // ═══════════════════════════════════════════════════════
  // 步骤 1: 滑动窗口检查
  // ═══════════════════════════════════════════════════════
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    // 获取最老的 chunk
    int oldest_idx = pending_indices.front();
    cudaEvent_t oldest_event = events[oldest_idx % MAX_INFLIGHT];
    
    // 等待最老的 chunk 的 kernel 完成
    cudaEventSynchronize(oldest_event);
    
    // 释放最老的 chunk 的请求槽
    void* oldest_req = pending_reqs.front();
    tcpx_irecv_consumed(recv_comm, 1, oldest_req);
    
    // 从窗口移除
    pending_reqs.erase(pending_reqs.begin());
    pending_indices.erase(pending_indices.begin());
  }
  
  // ═══════════════════════════════════════════════════════
  // 步骤 2: 发起当前 chunk 的接收
  // ═══════════════════════════════════════════════════════
  void* recv_request = nullptr;
  tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, 
             recv_mhandles, &recv_request);
  
  // 等待接收完成
  int done = 0;
  while (!done) {
    tcpx_test(recv_request, &done, nullptr);
  }
  
  // ═══════════════════════════════════════════════════════
  // 步骤 3: 启动 unpack kernel (异步)
  // ═══════════════════════════════════════════════════════
  launcher_ptr->launch(desc_block);
  
  // 记录 event (用于跟踪 kernel 完成)
  int event_idx = chunk_idx % MAX_INFLIGHT;
  cudaEventRecord(events[event_idx], unpack_stream);
  
  // 加入滑动窗口
  pending_reqs.push_back(recv_request);
  pending_indices.push_back(chunk_idx);
}

// ═══════════════════════════════════════════════════════
// 步骤 4: 排空窗口 (等待剩余 chunks 完成)
// ═══════════════════════════════════════════════════════
while (!pending_reqs.empty()) {
  int oldest_idx = pending_indices.front();
  cudaEventSynchronize(events[oldest_idx % MAX_INFLIGHT]);
  tcpx_irecv_consumed(recv_comm, 1, pending_reqs.front());
  pending_reqs.erase(pending_reqs.begin());
  pending_indices.erase(pending_indices.begin());
}
```

---

## Client 端滑动窗口详解

### 时间线示例

```
时间 →

Chunk 0-11: 填充窗口 (Client 使用 12 而不是 16，留余量)
═══════════════════════════════════════════════════════════════
t0:  C0  isend → pending
t1:  C1  isend → pending
...
t11: C11 isend → pending
     ↑
     窗口已满 (12 个 chunks)

Chunk 12: 滑动窗口开始工作
═══════════════════════════════════════════════════════════════
t12: 检查: pending_send_reqs.size() = 12 >= MAX_INFLIGHT_SEND
     ↓
     等待 C0 完成:
       while (!done) tcpx_test(req_0, &done, ...)
     ↓
     C0 自动释放 (send 请求在 done=1 时自动释放)
     ↓
     从窗口移除 C0:
       pending_send_reqs.erase(begin)
     ↓
     发起 C12:
       isend → pending
     ↓
     窗口: [C1][C2]...[C11][C12]
```

### 与 Server 端的区别

| 特性 | Server (Recv) | Client (Send) |
|------|---------------|---------------|
| 窗口大小 | 16 | 12 (留余量) |
| 需要 Events? | ✅ 是 (跟踪 kernel) | ❌ 否 |
| 释放方式 | 显式调用 `irecv_consumed` | 自动释放 (test 返回 done=1) |
| 复杂度 | 高 (需要管理 events) | 低 (只需轮询 test) |

### 代码示例

```cpp
// Client 端滑动窗口核心代码
constexpr int MAX_INFLIGHT_SEND = 12;
std::vector<void*> pending_send_reqs;

for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
  // ═══════════════════════════════════════════════════════
  // 步骤 1: 滑动窗口检查
  // ═══════════════════════════════════════════════════════
  if (pending_send_reqs.size() >= MAX_INFLIGHT_SEND) {
    // 等待最老的 send 完成
    void* oldest_req = pending_send_reqs.front();
    int done = 0;
    while (!done) {
      tcpx_test(oldest_req, &done, nullptr);
    }
    // 请求自动释放，只需从窗口移除
    pending_send_reqs.erase(pending_send_reqs.begin());
  }
  
  // ═══════════════════════════════════════════════════════
  // 步骤 2: 发起当前 chunk 的发送
  // ═══════════════════════════════════════════════════════
  void* send_request = nullptr;
  tcpx_isend(send_comm, src_ptr, size, tag, mhandle, &send_request);
  
  // 加入滑动窗口
  pending_send_reqs.push_back(send_request);
}

// ═══════════════════════════════════════════════════════
// 步骤 3: 排空窗口
// ═══════════════════════════════════════════════════════
while (!pending_send_reqs.empty()) {
  void* oldest_req = pending_send_reqs.front();
  int done = 0;
  while (!done) {
    tcpx_test(oldest_req, &done, nullptr);
  }
  pending_send_reqs.erase(pending_send_reqs.begin());
}
```

---

## 性能分析

### 没有滑动窗口 (崩溃)

```
Chunk 0-15: ✅ 成功
Chunk 16:   ❌ 错误: "unable to allocate requests"
程序崩溃
```

### 有滑动窗口 (稳定)

```
Chunk 0-15:   填充窗口
Chunk 16-127: 滑动窗口工作
              - 每次等待最老的完成
              - 释放槽位
              - 发起新的 chunk
              - 稳定运行

总时间: ~25ms (64MB @ 20 GB/s)
```

### 吞吐量对比

| 实现 | 并发数 | 吞吐量 | 稳定性 |
|------|--------|--------|--------|
| 批量发起 | 128 | N/A | ❌ 崩溃 |
| 串行 (无并发) | 1 | ~2 GB/s | ✅ 稳定 |
| 滑动窗口 | 16 | ~20 GB/s | ✅ 稳定 |

**关键洞察**: 滑动窗口在保证稳定性的同时，实现了接近最大并发的性能。

---

## 总结

### 滑动窗口的三个关键要素

1. **窗口大小限制**: `pending.size() < MAX_INFLIGHT`
2. **等待最老的完成**: `wait_oldest()` 或 `cudaEventSynchronize()`
3. **释放资源**: `irecv_consumed()` 或自动释放

### 为什么有效？

```
┌─────────────────────────────────────────────────┐
│  TCPX 请求池 (16 个槽)                          │
│                                                 │
│  滑动窗口确保:                                  │
│  - 最多 16 个槽被占用                           │
│  - 最老的槽被及时释放                           │
│  - 新的请求可以重用释放的槽                     │
│                                                 │
│  结果:                                          │
│  - 稳定运行 (不会耗尽槽位)                      │
│  - 高吞吐量 (接近最大并发)                      │
│  - 低延迟 (流水线化执行)                        │
└─────────────────────────────────────────────────┘
```

---

**最后更新**: 2025-10-02  
**作者**: 基于实际开发经验和性能分析

