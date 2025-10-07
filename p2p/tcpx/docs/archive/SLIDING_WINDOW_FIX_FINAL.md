# 滑动窗口修复：Server 端只处理 17 个 Chunks 的问题

**日期**: 2025-10-02  
**状态**: ✅ **已修复并编译成功**

---

## 🎯 问题总结

### 现象

Server 端每次迭代只处理 **17 个 chunks (chunk_idx=0-16)**，应该是 **128 个 chunks**。

**日志证据**:
```
Iteration 0:
  ✅ chunk_idx=0-15: tcpx_irecv 成功 (16 个)
  ❌ chunk_idx=16: tcpx_irecv 失败，rc=3 "unable to allocate requests"

Iteration 1-19: 完全相同的模式
```

---

## 🔍 根本原因

### 问题：滑动窗口检查在 `tcpx_irecv` 之后

**错误的代码逻辑**:
```cpp
// 1. 先调用 tcpx_irecv（错误！）
tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request);

// 2. 然后检查滑动窗口（太晚了！）
if (impl == "kernel") {
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    // 等待最老的 chunk 完成
    // 释放 TCPX 请求槽
  }
}
```

**为什么会失败**:
1. 前 16 个 chunks (0-15) 成功，填满了 TCPX 请求池（MAX_REQUESTS=16）
2. 第 17 个 chunk (chunk_idx=16) 时：
   - **先调用 `tcpx_irecv`**，但请求池已满
   - TCPX 返回 rc=3 "unable to allocate requests"
   - 循环 `break`，迭代提前结束
3. 滑动窗口的检查和释放逻辑**从未执行**

---

## 🔧 修复方案

### 解决方案：将滑动窗口检查移到 `tcpx_irecv` 之前

**修复后的代码逻辑**:
```cpp
// 1. 先检查滑动窗口（正确！）
if (!use_host_recv && impl == "kernel") {
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    // 等待最老的 chunk 的 kernel 完成
    cudaEventSynchronize(oldest_event);
    
    // 释放 TCPX 请求槽
    tcpx_irecv_consumed(recv_comm, 1, oldest_req);
    
    // 从滑动窗口中移除
    pending_reqs.erase(pending_reqs.begin());
    pending_indices.erase(pending_indices.begin());
  }
}

// 2. 然后调用 tcpx_irecv（正确！）
tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request);
```

**为什么会成功**:
1. 前 16 个 chunks (0-15) 成功，填满滑动窗口
2. 第 17 个 chunk (chunk_idx=16) 时：
   - **先检查滑动窗口**：`pending_reqs.size() >= MAX_INFLIGHT` (16 >= 16) ✅
   - 等待 chunk 0 的 kernel 完成
   - 调用 `tcpx_irecv_consumed` 释放 chunk 0 的请求槽
   - **然后调用 `tcpx_irecv`**，请求池有可用槽位 ✅
3. 继续处理 chunk 17-127，所有 128 个 chunks 都能成功

---

## 📝 详细修改

### 修改 1: 在 `tcpx_irecv` 之前添加滑动窗口检查

**文件**: `tests/test_tcpx_perf.cc`  
**位置**: 第 519-565 行（`tcpx_irecv` 之前）

```cpp
// ======================================================================
// 【修复】滑动窗口检查 - 必须在 tcpx_irecv 之前！
// ======================================================================

// 【问题】TCPX 插件每个 comm 只有 16 个请求槽
// 如果同时有超过 16 个 irecv 请求未调用 irecv_consumed，会报错
//
// 【解决方案】在发起新的 irecv 之前，检查滑动窗口是否已满
// 如果满了，先等待最老的 chunk 完成并释放请求槽

if (!use_host_recv && impl == "kernel") {
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    // 获取最老的 chunk 的索引和 event
    int oldest_idx = pending_indices.front();
    cudaEvent_t oldest_event = events[oldest_idx % MAX_INFLIGHT];

    std::cout << "[DEBUG] Sliding window FULL (" << pending_reqs.size() 
              << "), waiting for chunk " << oldest_idx << std::endl;

    // 【关键】等待最老的 chunk 的 kernel 完成
    cudaError_t err = cudaEventSynchronize(oldest_event);
    if (err != cudaSuccess) {
      std::cerr << "[ERROR] cudaEventSynchronize (pre-irecv) failed: " 
                << cudaGetErrorString(err) << std::endl;
      break;
    }

    // 【关键】释放最老的 chunk 的 TCPX 请求槽
    void* oldest_req = pending_reqs.front();
    int oldest_chunk_idx = pending_indices.front();

    std::cout << "[DEBUG] Releasing TCPX request: chunk_idx=" << oldest_chunk_idx
              << " request=" << oldest_req << " pending_before=" << pending_reqs.size() << std::endl;

    tcpx_irecv_consumed(recv_comm, 1, oldest_req);

    // 从滑动窗口中移除最老的 chunk
    pending_reqs.erase(pending_reqs.begin());
    pending_indices.erase(pending_indices.begin());

    std::cout << "[DEBUG] Request released: pending_after=" << pending_reqs.size() << std::endl;
  }
}

// ======================================================================
// 发起异步接收（tcpx_irecv）
// ======================================================================

// TCPX irecv 参数（支持批量接收，这里只接收 1 个）
void* recv_data[1] = {dst_ptr};
int recv_sizes[1] = {static_cast<int>(this_chunk)};
int recv_tags[1] = {tag};
void* recv_mhandles[1] = {recv_mhandle};
void* recv_request = nullptr;

int irecv_rc = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request);
if (irecv_rc != 0) {
  std::cerr << "[ERROR] tcpx_irecv failed: rc=" << irecv_rc
            << " chunk_idx=" << chunk_idx << " iter=" << iter
            << " offset=" << offset << " tag=" << tag << std::endl;
  std::cerr.flush();
  break;
}
```

### 修改 2: 删除原位置的重复滑动窗口逻辑

**文件**: `tests/test_tcpx_perf.cc`  
**位置**: 第 676-679 行（原来的滑动窗口检查位置）

```cpp
// ----------------------------------------------------------------
// 【注意】滑动窗口检查已经移到 tcpx_irecv 之前（第 530-565 行）
// ----------------------------------------------------------------
// 这样可以确保在发起新的 irecv 之前，TCPX 请求池有可用的槽位
```

---

## 🧪 预期结果

修复后，Server 端应该：

### 成功标准

1. ✅ **每次迭代处理所有 128 个 chunks**
   ```
   [DEBUG] tcpx_irecv success: chunk_idx=0 tag=99 request=0x...
   [DEBUG] tcpx_irecv success: chunk_idx=1 tag=100 request=0x...
   ...
   [DEBUG] tcpx_irecv success: chunk_idx=127 tag=226 request=0x...
   ```

2. ✅ **滑动窗口正常工作**
   ```
   [DEBUG] Sliding window FULL (16), waiting for chunk 0
   [DEBUG] Releasing TCPX request: chunk_idx=0 request=0x... pending_before=16
   [DEBUG] Request released: pending_after=15
   [DEBUG] tcpx_irecv success: chunk_idx=16 tag=115 request=0x...
   ```

3. ✅ **所有 20 次迭代成功完成**
   ```
   [PERF] Iter 0 time=XXX.XXms
   [PERF] Iter 1 time=XXX.XXms
   ...
   [PERF] Iter 19 time=XXX.XXms
   [PERF] Avg: XXX ms, BW: ~20 GB/s
   ```

4. ✅ **Client 端不会出现 "Connection reset by peer" 错误**

### 性能预期

- **每次迭代时间**: ~20-30 ms（处理 64 MB）
- **平均带宽**: ~20 GB/s（四网卡聚合，每个 25 Gbps）
- **所有迭代**: 成功完成，无超时或连接错误

---

## 🚀 测试步骤

### 1. 编译

```bash
cd /home/daniel/uccl/p2p/tcpx
make clean && make test_tcpx_perf -j4
```

**状态**: ✅ 编译成功

### 2. 运行测试

**Server 端 (10.65.74.150)**:
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/server_fixed_$(date +%Y%m%d_%H%M%S).log
```

**Client 端 (10.64.113.77)**:
```bash
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/client_fixed_$(date +%Y%m%d_%H%M%S).log
```

### 3. 验证结果

**检查 Server 端是否处理所有 128 个 chunks**:
```bash
grep "chunk_idx=127" logs/server_fixed_*.log
```

**检查滑动窗口是否工作**:
```bash
grep "Sliding window FULL" logs/server_fixed_*.log
```

**检查性能**:
```bash
grep "Avg:" logs/server_fixed_*.log
```

---

## 📊 修复前后对比

### 修复前

| 指标 | Server 端 | Client 端 |
|------|-----------|-----------|
| 处理的 chunks | **17 个** (0-16) | 128 个 |
| 迭代完成 | ✅ 20 次 | ❌ 3 次后失败 |
| 错误信息 | `unable to allocate requests` | `Connection reset by peer` |
| 平均时间 | 3.27 ms | - |
| 带宽 | 19.11 GB/s (错误！) | - |

**注意**: Server 端的带宽是基于 64 MB 计算的，但实际只传输了 8.5 MB (17 × 512 KB)，所以实际带宽应该是：
```
实际带宽 = 8.5 MB / 3.27 ms = 2.60 GB/s
```

### 修复后（预期）

| 指标 | Server 端 | Client 端 |
|------|-----------|-----------|
| 处理的 chunks | **128 个** (0-127) ✅ | 128 个 |
| 迭代完成 | ✅ 20 次 | ✅ 20 次 |
| 错误信息 | 无 | 无 |
| 平均时间 | ~25 ms | ~25 ms |
| 带宽 | **~20 GB/s** ✅ | **~20 GB/s** ✅ |

---

## 🎓 经验教训

### 1. 滑动窗口的正确实现

**关键原则**: 在申请资源之前，先检查资源池是否有可用槽位。

**错误模式**:
```cpp
申请资源();  // 可能失败
if (资源池满) {
  释放资源();  // 太晚了！
}
```

**正确模式**:
```cpp
if (资源池满) {
  释放资源();  // 先释放
}
申请资源();  // 然后申请
```

### 2. 调试日志的重要性

添加详细的调试日志帮助我们快速定位问题：
- `[DEBUG] tcpx_irecv success` - 确认每个 chunk 是否成功
- `[DEBUG] Sliding window FULL` - 确认滑动窗口是否触发
- `[DEBUG] Releasing TCPX request` - 确认请求槽是否被释放
- `[ERROR] tcpx_irecv failed: rc=3` - 确认失败的原因

### 3. TCPX 请求池的限制

TCPX 插件每个 comm 只有 **16 个请求槽** (MAX_REQUESTS=16)。

**必须遵守的规则**:
1. 同时最多有 16 个未完成的 `irecv` 请求
2. 必须调用 `tcpx_irecv_consumed` 释放请求槽
3. 必须在 kernel 完成后才能调用 `irecv_consumed`
4. 使用滑动窗口管理请求池

---

## 📚 相关文档

1. **`SERVER_17_CHUNKS_BUG.md`** - 问题的详细分析
2. **`DEBUG_PLAN_20251002.md`** - 调试计划和测试步骤
3. **`TIMEOUT_FIX_20251002.md`** - 超时问题的修复
4. **`BUG_ANALYSIS_20251002.md`** - 早期的 bug 分析

---

## 🎯 下一步

1. ⏳ **在两台机器上运行测试**
2. ⏳ **验证所有 128 个 chunks 都被处理**
3. ⏳ **验证带宽达到 ~20 GB/s**
4. ⏳ **验证 Client 端不会出现连接错误**
5. ⏳ **如果成功，清理调试日志并优化代码**

---

**状态**: ✅ 代码已修复并编译成功，等待测试验证  
**最后更新**: 2025-10-02

