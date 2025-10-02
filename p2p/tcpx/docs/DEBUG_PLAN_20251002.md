# 调试计划：Server 端只处理 17 个 Chunks 问题

**日期**: 2025-10-02  
**问题**: Server 端每次迭代只处理 17 个 chunks，应该是 128 个  
**状态**: 🔧 **已添加调试日志，等待测试**

---

## 📋 已完成的修改

### 1. 添加 `tcpx_irecv` 失败时的详细日志

**位置**: `tests/test_tcpx_perf.cc:531-542`

**修改前**:
```cpp
if (tcpx_irecv(...) != 0) {
  std::cerr << "[ERROR] tcpx_irecv failed (chunk)" << std::endl;
  break;
}
```

**修改后**:
```cpp
int irecv_rc = tcpx_irecv(...);
if (irecv_rc != 0) {
  std::cerr << "[ERROR] tcpx_irecv failed: rc=" << irecv_rc 
            << " chunk_idx=" << chunk_idx << " iter=" << iter 
            << " offset=" << offset << " tag=" << tag << std::endl;
  std::cerr.flush();  // 强制刷新缓冲区
  break;
}

// 【调试】记录成功的 irecv 调用
std::cout << "[DEBUG] tcpx_irecv success: chunk_idx=" << chunk_idx 
          << " tag=" << tag << " request=" << recv_request << std::endl;
```

**目的**:
- 记录 `tcpx_irecv` 的返回值
- 记录失败时的 chunk_idx、iter、offset、tag
- 强制刷新 stderr 缓冲区，确保错误信息写入日志
- 记录每次成功的 irecv 调用，方便定位最后一个成功的 chunk

### 2. 添加滑动窗口释放请求时的调试日志

**位置**: `tests/test_tcpx_perf.cc:657-671`

**修改前**:
```cpp
void* oldest_req = pending_reqs.front();
tcpx_irecv_consumed(recv_comm, 1, oldest_req);

pending_reqs.erase(pending_reqs.begin());
pending_indices.erase(pending_indices.begin());
```

**修改后**:
```cpp
void* oldest_req = pending_reqs.front();
int oldest_chunk_idx = pending_indices.front();

std::cout << "[DEBUG] Releasing TCPX request: chunk_idx=" << oldest_chunk_idx 
          << " request=" << oldest_req << " pending_before=" << pending_reqs.size() << std::endl;

tcpx_irecv_consumed(recv_comm, 1, oldest_req);

pending_reqs.erase(pending_reqs.begin());
pending_indices.erase(pending_indices.begin());

std::cout << "[DEBUG] Request released: pending_after=" << pending_reqs.size() << std::endl;
```

**目的**:
- 记录每次释放 TCPX 请求的 chunk_idx
- 记录释放前后的 pending_reqs 大小
- 验证滑动窗口是否正确释放请求

### 3. 添加迭代开始时的调试日志

**位置**: `tests/test_tcpx_perf.cc:490-496`

**修改前**:
```cpp
if (!use_host_recv && impl == "kernel") {
  pending_reqs.clear();
  pending_indices.clear();
}
```

**修改后**:
```cpp
if (!use_host_recv && impl == "kernel") {
  std::cout << "[DEBUG] Iteration " << iter << " start: clearing sliding window (was " 
            << pending_reqs.size() << " pending)" << std::endl;
  pending_reqs.clear();
  pending_indices.clear();
}
```

**目的**:
- 验证每次迭代开始时滑动窗口是否为空
- 如果不为空，说明上一次迭代没有正确清空

---

## 🧪 测试步骤

### 步骤 1: 编译

```bash
cd /home/daniel/uccl/p2p/tcpx
make clean && make test_tcpx_perf -j4
```

### 步骤 2: 运行测试

**Server 端 (10.65.74.150)**:
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/server_debug_$(date +%Y%m%d_%H%M%S).log
```

**Client 端 (10.64.113.77)**:
```bash
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/client_debug_$(date +%Y%m%d_%H%M%S).log
```

### 步骤 3: 分析日志

**查找关键信息**:

1. **最后一个成功的 `tcpx_irecv`**:
   ```bash
   grep "\[DEBUG\] tcpx_irecv success" logs/server_debug_*.log | tail -20
   ```

2. **`tcpx_irecv` 失败信息**:
   ```bash
   grep "\[ERROR\] tcpx_irecv failed" logs/server_debug_*.log
   ```

3. **滑动窗口释放日志**:
   ```bash
   grep "\[DEBUG\] Releasing TCPX request" logs/server_debug_*.log | tail -20
   ```

4. **迭代开始时的滑动窗口状态**:
   ```bash
   grep "\[DEBUG\] Iteration.*start" logs/server_debug_*.log
   ```

---

## 🔍 预期发现

### 场景 1: `tcpx_irecv` 在第 18 个 chunk 时失败

**预期日志**:
```
[DEBUG] tcpx_irecv success: chunk_idx=0 tag=99 request=0x...
[DEBUG] tcpx_irecv success: chunk_idx=1 tag=100 request=0x...
...
[DEBUG] tcpx_irecv success: chunk_idx=16 tag=115 request=0x...
[ERROR] tcpx_irecv failed: rc=2 chunk_idx=17 iter=0 offset=8912896 tag=116
```

**结论**: TCPX 请求池耗尽，需要修复滑动窗口逻辑或减小 MAX_INFLIGHT

### 场景 2: 滑动窗口没有正确释放请求

**预期日志**:
```
[DEBUG] Releasing TCPX request: chunk_idx=0 request=0x... pending_before=16
[DEBUG] Request released: pending_after=15
[DEBUG] Releasing TCPX request: chunk_idx=1 request=0x... pending_before=16
[DEBUG] Request released: pending_after=15
...
[DEBUG] Iteration 1 start: clearing sliding window (was 16 pending)  ← 问题！
```

**结论**: 滑动窗口在迭代结束时没有完全清空，需要修复排空逻辑

### 场景 3: TCPX 内部状态异常

**预期日志**:
```
[DEBUG] tcpx_irecv success: chunk_idx=0 tag=99 request=0x...
...
[DEBUG] tcpx_irecv success: chunk_idx=16 tag=115 request=0x...
← 没有错误信息，但循环提前退出
```

**结论**: 可能是其他错误（如 kernel launch 失败、cudaEvent 失败）导致 break

---

## 🔧 可能的修复方案

### 方案 A: 减小滑动窗口大小

如果 `tcpx_irecv` 在第 18 个 chunk 时失败（rc=2，表示请求池耗尽）：

```cpp
// 从 MAX_INFLIGHT = 16 减小到 12
constexpr int MAX_INFLIGHT = 12;
```

### 方案 B: 添加延迟确保请求释放

如果滑动窗口释放后 TCPX 内部状态没有立即更新：

```cpp
tcpx_irecv_consumed(recv_comm, 1, oldest_req);

// 添加短暂延迟，确保 TCPX 内部状态更新
std::this_thread::sleep_for(std::chrono::microseconds(100));
```

### 方案 C: 修复其他错误

如果是 kernel launch 或 cudaEvent 失败：

```cpp
// 添加更详细的错误日志
if (lrc != 0) {
  std::cerr << "[ERROR] Unpack kernel launch failed: lrc=" << lrc 
            << " chunk_idx=" << chunk_idx << std::endl;
  std::cerr.flush();
  break;
}
```

---

## 📊 成功标准

修复后，Server 端日志应该显示：

```
[PERF] Iteration 0: total bytes=67108864, chunk_bytes=524288
[DEBUG] tcpx_irecv success: chunk_idx=0 tag=99 request=0x...
[DEBUG] tcpx_irecv success: chunk_idx=1 tag=100 request=0x...
...
[DEBUG] tcpx_irecv success: chunk_idx=127 tag=226 request=0x...  ← 所有 128 个 chunks
[DEBUG] Draining sliding window: 16 pending requests
[DEBUG] Sliding window drained, remaining: 0
[PERF] Iter 0 time=XXX.XXms  ← 时间应该更长（处理了 64 MB 而不是 8.5 MB）
```

**预期性能**:
- 每次迭代时间：~20-30 ms（处理 64 MB）
- 平均带宽：~20 GB/s（四网卡聚合）
- 所有 20 次迭代成功完成
- Client 端不会出现 "Connection reset by peer" 错误

---

## 🎯 下一步行动

1. ✅ **已完成**: 添加详细的调试日志
2. ✅ **已完成**: 重新编译
3. ⏳ **待执行**: 在两台机器上运行测试
4. ⏳ **待执行**: 分析日志，确定根本原因
5. ⏳ **待执行**: 实施相应的修复方案
6. ⏳ **待执行**: 验证修复效果

---

## 📝 备注

- 所有调试日志都使用 `[DEBUG]` 前缀，方便过滤
- 错误日志使用 `std::cerr.flush()` 确保立即写入文件
- 成功日志使用 `std::cout`，会被 `tee` 重定向到日志文件
- 日志文件名包含时间戳，避免覆盖之前的日志

