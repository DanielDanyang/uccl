# 根本原因：为什么 4 个 Channels 没有并行？

## 🐛 问题现象

- **1 个 connection**：2.6 GB/s
- **4 个 connections**：2.8 GB/s
- **预期**：应该有 3-4 倍提升（~10 GB/s）
- **实际**：几乎没有提升

## 🔍 根本原因

### TCPX 的 FIFO 约束

从 TCPX 源码 `/home/daniel/uccl/nccl-plugin-gpudirecttcpx/src/net_tcpx.cc:1323-1328`：

```cpp
tcpxRequest* ni = r->comm->rq.next_transmitting();

if (r != ni) {
  WARN("test called with invalid request");
  return tcpxInternalError;  // ← 只能 test 队列头！
}
```

**关键约束**：
- 每个 `tcpxComm` 内部有一个请求队列（`MAX_REQUESTS=16`）
- **必须按 FIFO 顺序 test**：只能 test 队列头部的请求
- 如果 test 非队列头的请求，会返回错误

**这意味着**：
- 每个 channel（comm）内部是**串行**的
- 但是多个 channels 之间应该是**并行**的

---

### 当前代码的问题

**文件**：`p2p/tcpx/tests/test_tcpx_perf_multi.cc:742-820`

```cpp
while (offset < test_size) {
  // Round-robin 选择 channel
  int channel_id = global_chunk_idx % num_channels;  // 0, 1, 2, 3, 0, 1, 2, 3, ...
  ChannelResources& ch = mgr.get_channel(channel_id);
  ChannelWindow& win = channel_windows[channel_id];
  
  // 等待这个 channel 有空间
  while (win.inflight_recvs.size() >= MAX_INFLIGHT_PER_CHANNEL) {
    process_completed_chunk(channel_id, ch, win, /*blocking=*/true);  // ← 阻塞等待！
  }
  
  // Post irecv
  tcpx_irecv(ch.recv_comm, ...);
  win.inflight_recvs.push_back(...);
  
  // 非阻塞地 drain 当前 channel
  process_completed_chunk(channel_id, ch, win, /*blocking=*/false);
  
  // 顺手 drain 其他 channels
  for (int other = 0; other < num_channels; ++other) {
    if (other == channel_id) continue;
    process_completed_chunk(other, ...);
  }
  
  offset += this_chunk;
  global_chunk_idx++;  // ← 串行递增！
}
```

**问题分析**：

1. **串行发送顺序**：
   ```
   chunk 0 → channel 0
   chunk 1 → channel 1
   chunk 2 → channel 2
   chunk 3 → channel 3
   chunk 4 → channel 0  ← 等待 channel 0 有空间
   chunk 5 → channel 1  ← 等待 channel 1 有空间
   ...
   ```

2. **阻塞等待**：
   - 当 channel 0 满了（16 个 inflight），我们会**阻塞等待** channel 0
   - 在等待期间，channel 1, 2, 3 可能是空闲的，但我们不去使用它们
   - 结果：**串行化**

3. **时序图**：
   ```
   时间 →
   
   Channel 0: [post chunk 0] [wait...] [post chunk 4] [wait...] [post chunk 8]
   Channel 1:                 [post chunk 1] [wait...] [post chunk 5] [wait...]
   Channel 2:                                 [post chunk 2] [wait...] [post chunk 6]
   Channel 3:                                                 [post chunk 3] [wait...]
   
   ← 串行！每次只有一个 channel 在工作
   ```

---

## 💡 正确的并行方式

### 目标时序图

```
时间 →

Channel 0: [post 0] [post 4] [post 8]  [post 12] [post 16] ...
Channel 1: [post 1] [post 5] [post 9]  [post 13] [post 17] ...
Channel 2: [post 2] [post 6] [post 10] [post 14] [post 18] ...
Channel 3: [post 3] [post 7] [post 11] [post 15] [post 19] ...

← 并行！所有 channels 同时工作
```

### 实现策略

**方案 A：批量填充所有 channels**

```cpp
int next_chunk_for_channel[num_channels] = {0, 1, 2, 3};  // 每个 channel 的下一个 chunk

while (还有数据要发送) {
  bool any_posted = false;
  
  // 尝试为每个 channel post 一个 chunk
  for (int ch = 0; ch < num_channels; ++ch) {
    ChannelWindow& win = channel_windows[ch];
    
    // 如果这个 channel 有空间，并且还有数据要发送
    if (win.inflight_recvs.size() < MAX_INFLIGHT_PER_CHANNEL && 
        next_chunk_for_channel[ch] < total_chunks) {
      
      int global_chunk_idx = next_chunk_for_channel[ch];
      
      // Post irecv for this chunk
      tcpx_irecv(...);
      win.inflight_recvs.push_back(...);
      
      // 更新下一个 chunk（跳过 num_channels）
      next_chunk_for_channel[ch] += num_channels;
      any_posted = true;
    }
    
    // 非阻塞地 drain 这个 channel
    process_completed_chunk(ch, ..., /*blocking=*/false);
  }
  
  // 如果所有 channels 都满了，阻塞等待任意一个有空间
  if (!any_posted) {
    for (int ch = 0; ch < num_channels; ++ch) {
      if (!channel_windows[ch].inflight_recvs.empty()) {
        process_completed_chunk(ch, ..., /*blocking=*/true);
        break;  // 只要有一个 channel 释放了空间就继续
      }
    }
  }
}
```

**关键改进**：
1. ✅ 每次循环尝试为**所有 channels** post chunks
2. ✅ 只有当**所有 channels 都满**时才阻塞等待
3. ✅ 阻塞等待时，只要**任意一个 channel** 有空间就继续
4. ✅ 结果：所有 channels **并行工作**

---

## 📊 预期效果

### 修复前（当前）
```
Channel 0: [====    ]  ← 部分时间在工作
Channel 1: [  ====  ]  ← 部分时间在工作
Channel 2: [    ====]  ← 部分时间在工作
Channel 3: [      ==]  ← 部分时间在工作

总利用率：~25%（串行）
带宽：2.8 GB/s
```

### 修复后
```
Channel 0: [========]  ← 一直在工作
Channel 1: [========]  ← 一直在工作
Channel 2: [========]  ← 一直在工作
Channel 3: [========]  ← 一直在工作

总利用率：~100%（并行）
带宽：~10-12 GB/s（4 倍提升）
```

---

## 🔧 实施计划

### 步骤 1：重构 Server 端的 chunk 发送循环

**文件**：`p2p/tcpx/tests/test_tcpx_perf_multi.cc:742-820`

**修改**：
1. 移除串行的 `while (offset < test_size)` 循环
2. 添加并行的 channel 填充逻辑
3. 为每个 channel 维护独立的 `next_chunk_idx`
4. 每次循环尝试为所有 channels post chunks

### 步骤 2：重构 Client 端的 chunk 发送循环

**文件**：`p2p/tcpx/tests/test_tcpx_perf_multi.cc:1075-1152`

**修改**：
1. 同样的并行填充逻辑
2. 确保 client 和 server 的 chunk 顺序一致

### 步骤 3：测试验证

**预期结果**：
- ✅ 4 个 channels 同时工作
- ✅ 带宽提升到 ~10-12 GB/s
- ✅ 日志显示所有 channels 的 inflight 数量都接近 MAX_INFLIGHT_PER_CHANNEL

---

## 🚨 注意事项

### 1. Tag 唯一性

确保每个 chunk 的 tag 仍然是唯一的：
```cpp
const int tag = kTransferTag + iter * 10000 + global_chunk_idx;
```

### 2. FIFO 顺序

每个 channel 内部仍然必须保持 FIFO：
```cpp
// 每个 channel 的 inflight_recvs 必须按 post 顺序排列
win.inflight_recvs.push_back(...);  // ← 保持 FIFO

// Test 时只 test 队列头
auto& entry = win.inflight_recvs.front();  // ← 保持 FIFO
tcpx_test(entry.request, ...);
```

### 3. Client-Server 同步

Client 和 server 必须以相同的顺序发送/接收 chunks：
- Server: chunk 0→ch0, chunk 1→ch1, chunk 2→ch2, chunk 3→ch3, chunk 4→ch0, ...
- Client: chunk 0→ch0, chunk 1→ch1, chunk 2→ch2, chunk 3→ch3, chunk 4→ch0, ...

---

## 📝 总结

### 问题
当前代码虽然有 4 个 channels，但是**串行发送** chunks，导致同一时间只有一个 channel 在工作。

### 根本原因
1. 串行的 `global_chunk_idx++` 循环
2. 阻塞等待单个 channel 有空间
3. 没有并行填充所有 channels

### 解决方案
重构为并行填充模式：
- 每次循环尝试为**所有 channels** post chunks
- 只有当**所有 channels 都满**时才阻塞
- 阻塞时，只要**任意一个 channel** 有空间就继续

### 预期效果
- 带宽从 2.8 GB/s 提升到 ~10-12 GB/s（4 倍）
- 所有 channels 并行工作，利用率接近 100%

