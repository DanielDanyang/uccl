# 调试日志移除 - 性能优化

**日期**: 2025-10-02  
**状态**: ✅ 已完成并编译成功

---

## 🎯 目标

移除所有 `[DEBUG]` 日志以提升性能。

**假设**: 大量的 `std::cout` 输出导致 2-3× 性能下降。

**预期结果**:
- Server: 100 ms → **30-50 ms** (2-3× 提升)
- Client: 157 ms → **50-80 ms** (2-3× 提升)
- 带宽: 0.6 GB/s → **1-2 GB/s**

---

## 📝 修改内容

### 移除的调试日志（10 处）

#### 1. Server 端迭代开始日志

**位置**: `tests/test_tcpx_perf.cc:492-493`

**修改前**:
```cpp
std::cout << "[DEBUG] Iteration " << iter << " start: clearing sliding window (was "
          << pending_reqs.size() << " pending)" << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Iteration start: clearing sliding window (removed for performance)
```

---

#### 2. 滑动窗口满日志

**位置**: `tests/test_tcpx_perf.cc:538-539`

**修改前**:
```cpp
std::cout << "[DEBUG] Sliding window FULL (" << pending_reqs.size() 
          << "), waiting for chunk " << oldest_idx << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Sliding window FULL (removed for performance)
```

---

#### 3. 释放 TCPX 请求日志

**位置**: `tests/test_tcpx_perf.cc:553-554`

**修改前**:
```cpp
std::cout << "[DEBUG] Releasing TCPX request: chunk_idx=" << oldest_chunk_idx
          << " request=" << oldest_req << " pending_before=" << pending_reqs.size() << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Releasing TCPX request (removed for performance)
```

**额外修改**: 注释掉未使用的变量 `oldest_chunk_idx`

---

#### 4. 请求释放完成日志

**位置**: `tests/test_tcpx_perf.cc:562`

**修改前**:
```cpp
std::cout << "[DEBUG] Request released: pending_after=" << pending_reqs.size() << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Request released (removed for performance)
```

---

#### 5. tcpx_irecv 成功日志

**位置**: `tests/test_tcpx_perf.cc:587-588`

**修改前**:
```cpp
std::cout << "[DEBUG] tcpx_irecv success: chunk_idx=" << chunk_idx
          << " tag=" << tag << " request=" << recv_request << std::endl;
```

**修改后**:
```cpp
// [DEBUG] tcpx_irecv success (removed for performance)
```

---

#### 6. Server 端排空滑动窗口开始日志

**位置**: `tests/test_tcpx_perf.cc:798`

**修改前**:
```cpp
std::cout << "[DEBUG] Draining sliding window: " << pending_reqs.size() << " pending requests" << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Draining sliding window (removed for performance)
```

---

#### 7. 等待 chunk 完成日志

**位置**: `tests/test_tcpx_perf.cc:805`

**修改前**:
```cpp
std::cout << "[DEBUG] Waiting for chunk " << oldest_idx << " (event_idx=" << (oldest_idx % MAX_INFLIGHT) << ")" << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Waiting for chunk (removed for performance)
```

---

#### 8. Server 端排空完成日志

**位置**: `tests/test_tcpx_perf.cc:823`

**修改前**:
```cpp
std::cout << "[DEBUG] Sliding window drained, remaining: " << pending_reqs.size() << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Sliding window drained (removed for performance)
```

---

#### 9. Client 端排空滑动窗口开始日志

**位置**: `tests/test_tcpx_perf.cc:1077`

**修改前**:
```cpp
std::cout << "[DEBUG] Draining client sliding window: " << pending_send_reqs.size() << " pending send requests" << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Draining client sliding window (removed for performance)
```

---

#### 10. Client 端排空完成日志

**位置**: `tests/test_tcpx_perf.cc:1093`

**修改前**:
```cpp
std::cout << "[DEBUG] Client sliding window drained, remaining: " << pending_send_reqs.size() << std::endl;
```

**修改后**:
```cpp
// [DEBUG] Client sliding window drained (removed for performance)
```

---

## ✅ 编译状态

```bash
make clean && make test_tcpx_perf -j4
```

**结果**: ✅ **编译成功，无警告！**

---

## 🧪 测试步骤

### 1. 运行测试

**Server 端 (10.65.74.150)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/server_no_debug_$(date +%Y%m%d_%H%M%S).log
```

**Client 端 (10.64.113.77)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/client_no_debug_$(date +%Y%m%d_%H%M%S).log
```

---

### 2. 验证性能

**查看性能**:
```bash
# Server 端
grep "Avg:" logs/server_no_debug_*.log

# Client 端
grep "Avg:" logs/client_no_debug_*.log
```

**预期结果**:
```
Server: Avg: 30-50 ms, BW: 1-2 GB/s
Client: Avg: 50-80 ms, BW: 0.8-1.3 GB/s
```

---

### 3. 对比性能

**修改前**:
```
Server: 100.155 ms, 0.62 GB/s
Client: 156.767 ms, 0.40 GB/s
```

**修改后（预期）**:
```
Server: 30-50 ms, 1-2 GB/s (2-3× 提升)
Client: 50-80 ms, 0.8-1.3 GB/s (2-3× 提升)
```

---

## 📊 保留的日志

### 保留的 [PERF] 日志

以下性能日志**保留**，用于性能分析：

1. `[PERF] Mode: SERVER/CLIENT`
2. `[PERF] GPU: X`
3. `[PERF] Size: X MB`
4. `[PERF] Iterations: X`
5. `[PERF] Unpack impl: kernel`
6. `[PERF][SERVER] chunk_idx=X tag=X size=X offset=X`
7. `[PERF][SERVER] frag_count=X`
8. `[PERF][CLIENT] chunk_idx=X tag=X size=X offset=X`
9. `[PERF] Iter X time=X.XXms`
10. `[PERF] Avg: X.XX ms, BW: X.XX GB/s`

### 保留的 [ERROR] 日志

所有错误日志**保留**，用于调试：

1. `[ERROR] tcpx_irecv failed: rc=X chunk_idx=X ...`
2. `[ERROR] cudaEventSynchronize failed: ...`
3. `[ERROR] Unpack kernel launch failed: ...`
4. `[ERROR] cudaEventRecord failed: ...`
5. 等等

---

## 🔍 如果需要重新启用调试日志

如果性能提升不明显，或需要调试新问题，可以重新启用调试日志：

```bash
# 搜索所有被注释的 DEBUG 日志
grep -n "// \[DEBUG\]" tests/test_tcpx_perf.cc

# 取消注释即可恢复
```

---

## 📝 总结

### 修改内容

- ✅ 移除 10 处 `[DEBUG]` 日志
- ✅ 注释掉 1 个未使用的变量
- ✅ 保留所有 `[PERF]` 和 `[ERROR]` 日志
- ✅ 编译成功，无警告

### 预期效果

- 🎯 Server: 100 ms → 30-50 ms (2-3× 提升)
- 🎯 Client: 157 ms → 50-80 ms (2-3× 提升)
- 🎯 带宽: 0.6 GB/s → 1-2 GB/s

### 下一步

1. ⏳ **立即测试** - 在两台机器上运行测试
2. ⏳ **验证性能** - 检查是否达到预期提升
3. ⏳ **如果成功** - 继续优化轮询策略
4. ⏳ **如果失败** - 进行网络基准测试

---

**状态**: ✅ 代码已修改并编译成功，等待测试验证

