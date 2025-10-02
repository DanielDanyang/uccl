# TCPX 常见错误和修复方案

## 🎯 快速诊断表

| 症状 | 可能原因 | 快速检查 | 文档位置 |
|------|---------|---------|---------|
| Kernel 比 D2D 慢 100× | 每个 chunk 创建 stream | 搜索代码中的 `cudaStreamCreate` 在循环内 | [错误 1](#错误-1-kernel-比-d2d-慢-100-倍) |
| "unable to allocate requests" | 超过 16 个并发请求 | 检查是否有滑动窗口逻辑 | [错误 2](#错误-2-unable-to-allocate-requests) |
| 数据校验失败 | 过早调用 `irecv_consumed` | 检查是否在 kernel 完成前释放 | [错误 3](#错误-3-数据校验失败垃圾数据) |
| 传输卡住/超时 | Tag 冲突 | 检查每个 chunk 是否有唯一 tag | [错误 4](#错误-4-传输卡住或超时) |
| "rx no cmsg" | devmem-tcp 未启用 | `dmesg \| grep devmem` | [错误 5](#错误-5-rx-no-cmsg) |

---

## 错误 1: Kernel 比 D2D 慢 100× 倍

### 症状

```
[PERF] Kernel mode: Avg: 7040 ms, BW: 0.01 GB/s
[PERF] D2D mode:    Avg: 8 ms,    BW: 8.0 GB/s
```

### 根本原因

每个 chunk 都创建和销毁 CUDA stream 和 launcher：

```cpp
// ❌ 错误代码
for (each chunk) {
  cudaStreamCreate(&stream);              // ~4ms
  UnpackLauncher launcher(stream);        // ~2ms (包含 cudaMalloc)
  launcher.launchSync(desc_block);        // ~48ms (同步等待!)
  cudaStreamDestroy(stream);              // ~1ms
}
// 总开销: ~55ms/chunk
```

### 性能分析

| 操作 | 耗时 | 累计 (128 chunks) |
|------|------|------------------|
| cudaStreamCreate | 4ms | 512ms |
| UnpackLauncher 构造 (cudaMalloc) | 2ms | 256ms |
| launchSync (同步等待) | 48ms | 6144ms |
| cudaStreamDestroy | 1ms | 128ms |
| **总计** | **55ms** | **7040ms** |

### 修复方案

在循环外创建，使用异步 launch：

```cpp
// ✅ 正确代码
cudaStreamCreate(&unpack_stream);                    // 一次
UnpackLauncher* launcher = new UnpackLauncher(cfg);  // 一次

for (each chunk) {
  launcher->launch(desc_block);  // ~0.01ms (异步)
}

cudaStreamSynchronize(unpack_stream);  // 最后同步一次
cudaStreamDestroy(unpack_stream);
delete launcher;
```

### 验证方法

```bash
# 修复前
grep -n "cudaStreamCreate" test_tcpx_perf.cc
# 应该在循环外 (例如第 271 行)，不在循环内

# 修复后运行
UCCL_TCPX_UNPACK_IMPL=kernel ./tests/test_tcpx_perf server 0
# 预期: Avg: 8-10 ms, BW: 6-8 GB/s
```

### 性能提升

```
修复前: 7040ms (0.01 GB/s)
修复后: 8ms    (8.0 GB/s)
提升:   880×
```

---

## 错误 2: "unable to allocate requests"

### 症状

```
[ERROR] tcpx_irecv failed (chunk)
[ncclNet:2] unable to allocate requests
```

### 根本原因

TCPX 插件每个 comm 只有 16 个请求槽：

```cpp
// nccl-plugin-gpudirecttcpx/src/work_queue.h
#define MAX_REQUESTS 16  // 固定大小，不可配置
```

批量发起超过 16 个请求：

```cpp
// ❌ 错误代码
for (int i = 0; i < 128; ++i) {  // 128 chunks
  tcpx_irecv(..., &reqs[i]);     // 第 17 个失败!
}
```

### 修复方案

使用滑动窗口限制并发数：

```cpp
// ✅ 正确代码 (Server 端)
constexpr int MAX_INFLIGHT = 16;
std::vector<void*> pending_reqs;

for (int i = 0; i < 128; ++i) {
  // 如果窗口满，等待最老的完成
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    cudaEventSynchronize(events[pending_indices.front() % MAX_INFLIGHT]);
    tcpx_irecv_consumed(recv_comm, 1, pending_reqs.front());
    pending_reqs.erase(pending_reqs.begin());
    pending_indices.erase(pending_indices.begin());
  }
  
  // 发起新的 irecv
  tcpx_irecv(..., &req);
  launcher->launch(desc_block);
  cudaEventRecord(events[i % MAX_INFLIGHT], stream);
  pending_reqs.push_back(req);
  pending_indices.push_back(i);
}
```

```cpp
// ✅ 正确代码 (Client 端)
constexpr int MAX_INFLIGHT_SEND = 12;  // 留余量
std::vector<void*> pending_send_reqs;

for (int i = 0; i < 128; ++i) {
  // 如果窗口满，等待最老的完成
  if (pending_send_reqs.size() >= MAX_INFLIGHT_SEND) {
    void* oldest = pending_send_reqs.front();
    int done = 0;
    while (!done) tcpx_test(oldest, &done, nullptr);
    pending_send_reqs.erase(pending_send_reqs.begin());
  }
  
  // 发起新的 isend
  tcpx_isend(..., &req);
  pending_send_reqs.push_back(req);
}
```

### 验证方法

```bash
# 检查代码中是否有滑动窗口逻辑
grep -A 5 "MAX_INFLIGHT" test_tcpx_perf.cc

# 运行测试
UCCL_TCPX_PERF_SIZE=67108864 ./tests/test_tcpx_perf server 0
# 应该不会出现 "unable to allocate requests" 错误
```

---

## 错误 3: 数据校验失败/垃圾数据

### 症状

```
[ERROR] Data verification failed
Expected: 0x42, Got: 0x00 (or random value)
```

### 根本原因

在 kernel 完成前调用 `tcpx_irecv_consumed`，导致 bounce buffer 被释放：

```cpp
// ❌ 错误代码
tcpx_test(req, &done, ...);
if (done) {
  launcher->launch(desc_block);      // 异步启动 kernel
  tcpx_irecv_consumed(comm, 1, req); // ❌ kernel 还没完成!
}
// bounce buffer 被释放 → kernel 读到垃圾数据
```

### 时间线分析

```
t0: tcpx_test 返回 done=1
    ↓ 数据在 bounce buffer (GPU 内存)
t1: launcher->launch (异步启动 kernel)
    ↓ kernel 在 GPU 上排队
t2: tcpx_irecv_consumed ❌ 释放 bounce buffer
    ↓ bounce buffer 被重用或清零
t3: kernel 开始执行
    ↓ 读取 bounce buffer → ❌ 读到垃圾数据!
```

### 修复方案

使用 CUDA Event 等待 kernel 完成：

```cpp
// ✅ 正确代码
tcpx_test(req, &done, ...);
if (done) {
  // 异步启动 kernel
  launcher->launch(desc_block);
  
  // 记录 event
  cudaEventRecord(events[chunk_idx % MAX_INFLIGHT], stream);
  
  // 加入滑动窗口
  pending_reqs.push_back(req);
  pending_indices.push_back(chunk_idx);
}

// 稍后 (滑动窗口满时或迭代结束时)
cudaEventSynchronize(events[oldest_idx % MAX_INFLIGHT]);  // ✅ 等待 kernel 完成
tcpx_irecv_consumed(comm, 1, oldest_req);                 // ✅ 现在安全了
```

### 验证方法

```bash
# 添加数据校验
cudaMemset(recv_buf, 0x42, size);  # Server 端初始化
cudaMemset(send_buf, 0x42, size);  # Client 端初始化

# 运行测试
./tests/test_tcpx_perf server 0
./tests/test_tcpx_perf client <server_ip> 0

# 接收完成后校验
unsigned char host_buf[size];
cudaMemcpy(host_buf, recv_buf, size, cudaMemcpyDeviceToHost);
for (int i = 0; i < size; ++i) {
  if (host_buf[i] != 0x42) {
    printf("Error at offset %d: expected 0x42, got 0x%02x\n", i, host_buf[i]);
  }
}
```

---

## 错误 4: 传输卡住或超时

### 症状

```
[PERF][CLIENT] chunk_idx=6 tag=99 size=524288 offset=3145728
[ERROR] Send timeout at iteration 0 chunk=6
```

### 根本原因

所有 chunk 使用相同的 tag，导致 TCPX 插件无法区分不同的请求：

```cpp
// ❌ 错误代码
for (each chunk) {
  tcpx_irecv(..., tag=99, ...);  // 所有 chunk 都是 tag 99
  tcpx_isend(..., tag=99, ...);
}
// TCPX 插件可能将 chunk 1 的数据匹配到 chunk 0 的请求
```

### 修复方案

每个 chunk 使用唯一 tag：

```cpp
// ✅ 正确代码
for (int iter = 0; iter < iterations; ++iter) {
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    // 唯一 tag = 基础 tag + 迭代编号*10000 + chunk 索引
    int tag = kTransferTag + iter * 10000 + chunk_idx;
    
    // Server
    tcpx_irecv(..., tag, ...);
    
    // Client (必须使用相同的 tag)
    tcpx_isend(..., tag, ...);
  }
}
```

### Tag 示例

```
Iteration 0:
  Chunk 0: tag = 99 + 0*10000 + 0 = 99
  Chunk 1: tag = 99 + 0*10000 + 1 = 100
  Chunk 2: tag = 99 + 0*10000 + 2 = 101
  ...

Iteration 1:
  Chunk 0: tag = 99 + 1*10000 + 0 = 10099
  Chunk 1: tag = 99 + 1*10000 + 1 = 10100
  ...
```

### 验证方法

```bash
# 检查 tag 计算逻辑
grep "kTransferTag" test_tcpx_perf.cc

# 运行测试，观察日志
./tests/test_tcpx_perf server 0 2>&1 | grep "tag="
# 应该看到递增的 tag: 99, 100, 101, ...
```

---

## 错误 5: "rx no cmsg"

### 症状

```
[TCPX] rx no cmsg
[ERROR] Failed to receive data
```

### 根本原因

devmem-tcp 未启用或不支持：

1. 内核不支持 devmem-tcp
2. 网卡不支持 devmem-tcp
3. 使用了错误的 IP 地址范围

### 诊断方法

```bash
# 1. 检查内核支持
dmesg | grep devmem
# 应该看到: "TCP: devmem-tcp enabled"

# 2. 检查网卡
ethtool -k eth1 | grep tcp-data-split
# 应该看到: tcp-data-split: on

# 3. 检查 IP 地址
ip addr show eth1
# GCP 环境应该是 10.64.x.x 范围
```

### 修复方案

```bash
# 1. 确保使用正确的网卡
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4

# 2. 确保使用正确的 IP 地址
# Server: 使用 10.65.74.150 (不是 localhost 或 127.0.0.1)
# Client: 连接到 10.65.74.150

# 3. 检查防火墙
sudo iptables -L | grep 50000
# 确保端口 50000-60000 开放
```

---

## 最佳实践检查清单

### 代码审查清单

- [ ] Stream 和 Launcher 在循环外创建
- [ ] 使用 `launch()` 而不是 `launchSync()`
- [ ] 实现了滑动窗口 (Server: 16, Client: 12)
- [ ] Server 端使用 CUDA Events 跟踪 kernel 完成
- [ ] 每个 chunk 使用唯一 tag
- [ ] 在 kernel 完成后才调用 `irecv_consumed`
- [ ] 迭代结束时排空滑动窗口

### 环境配置清单

- [ ] `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` 设置正确
- [ ] `NCCL_GPUDIRECTTCPX_PORT_BEGIN/END` 设置正确
- [ ] devmem-tcp 已启用 (`dmesg | grep devmem`)
- [ ] 使用正确的 IP 地址范围 (GCP: 10.64.x.x)
- [ ] 防火墙允许端口 50000-60000

### 性能验证清单

- [ ] Kernel 模式性能接近 D2D 模式 (±20%)
- [ ] 64MB 传输时间 < 50ms (4-NIC 环境)
- [ ] 带宽 > 15 GB/s (4-NIC 环境)
- [ ] 无 "unable to allocate requests" 错误
- [ ] 无数据校验错误

---

## 调试技巧

### 启用详细日志

```bash
export UCCL_TCPX_DEBUG=1
export UCCL_TCPX_LAUNCH_DEBUG=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

### 性能分析

```bash
# 使用 nsys 分析 kernel 性能
nsys profile --trace=cuda,nvtx ./tests/test_tcpx_perf server 0

# 查看 CUDA API 调用
nsys stats report.nsys-rep --report cudaapisum
```

### 数据校验

```cpp
// 在 Server 端添加校验
std::vector<unsigned char> expected(test_size, 0x42);
std::vector<unsigned char> actual(test_size);
cudaMemcpy(actual.data(), recv_buf, test_size, cudaMemcpyDeviceToHost);
if (memcmp(expected.data(), actual.data(), test_size) != 0) {
  std::cerr << "Data verification failed!" << std::endl;
}
```

---

**最后更新**: 2025-10-02  
**作者**: 基于实际开发经验和错误修复历史

