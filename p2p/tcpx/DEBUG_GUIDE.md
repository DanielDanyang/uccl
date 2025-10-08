# 调试指南 - TCPX 单进程 Orchestrator

## 🔍 **当前问题**

从最新日志 (`singleproc_server_20251007_111638.log`) 看到：
- 服务器在第 408 行卡住：`[SERVER] Processing GPU 0 with 4 channels`
- 没有后续输出
- 客户端等待 10 秒后也没有进展

## 🛠️ **已添加的调试日志**

### **服务器端调试点**

1. **GPU 处理开始**：
   ```
   [DEBUG] GPU X will post ~N chunks
   ```
   - 显示每个 GPU 预计要 post 多少个 chunks

2. **每个 chunk 开始**：
   ```
   [DEBUG] GPU X chunk Y → channel Z (window size=A/B)
   ```
   - 显示 chunk 分配到哪个 channel
   - 显示当前窗口大小

3. **窗口满时**：
   ```
   [DEBUG] GPU X channel Y window FULL, trying to release oldest...
   ```
   - 显示窗口满了，正在尝试释放

4. **tcpx_irecv 调用前**：
   ```
   [DEBUG] GPU X chunk Y calling tcpx_irecv (size=Z, tag=T)...
   ```
   - 显示即将调用 `tcpx_irecv`

5. **tcpx_irecv 调用后**：
   ```
   [DEBUG] GPU X chunk Y tcpx_irecv returned, request=0xADDR
   ```
   - 显示 `tcpx_irecv` 成功返回

## 📊 **如何使用调试日志**

### **步骤 1: 运行测试**

```bash
# 服务器（Node 0）
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh server

# 客户端（Node 1）
./test_step3_bandwidth.sh client <SERVER_IP>
```

### **步骤 2: 查看最新日志**

```bash
cd /home/daniel/uccl/p2p/tcpx
ls -lt logs/*.log | head -5
```

### **步骤 3: 分析服务器日志**

查看服务器卡在哪里：

```bash
tail -50 logs/singleproc_server_YYYYMMDD_HHMMSS.log
```

**可能的卡住点**：

#### **情况 1: 卡在 GPU 处理开始**
```
[SERVER] Processing GPU 0 with 4 channels
<没有后续输出>
```
**原因**: 可能在创建 `SlidingWindow` 对象或初始化时卡住

#### **情况 2: 卡在第一个 chunk**
```
[DEBUG] GPU 0 will post ~128 chunks
<没有后续输出>
```
**原因**: 可能在进入 `while (offset < test_size_per_gpu)` 循环前卡住

#### **情况 3: 卡在 tcpx_irecv 调用**
```
[DEBUG] GPU 0 chunk 0 → channel 0 (window size=0/16)
[DEBUG] GPU 0 chunk 0 calling tcpx_irecv (size=524288, tag=...)...
<没有后续输出>
```
**原因**: `tcpx_irecv()` 调用本身卡住（阻塞）

#### **情况 4: 卡在窗口释放**
```
[DEBUG] GPU 0 chunk 16 → channel 0 (window size=16/16)
[DEBUG] GPU 0 channel 0 window FULL, trying to release oldest...
<没有后续输出>
```
**原因**: `try_release_oldest()` 一直返回 1（未就绪），无限循环

### **步骤 4: 分析客户端日志**

```bash
tail -50 logs/singleproc_client_YYYYMMDD_HHMMSS.log
```

**正常情况**：
```
[CLIENT] Waiting 10 seconds for server to post receives...
[CLIENT] ===== Iteration 0 =====
[CLIENT] Starting to send data for iteration 0
...
```

**异常情况**：
```
[CLIENT] Waiting 10 seconds for server to post receives...
[CLIENT] ===== Iteration 0 =====
<没有后续输出>
```

## 🎯 **根据日志诊断问题**

### **诊断 1: tcpx_irecv() 阻塞**

**症状**：
```
[DEBUG] GPU 0 chunk 0 calling tcpx_irecv...
<卡住>
```

**可能原因**：
1. TCPX 内部请求队列满了（`MAX_REQUESTS=16`）
2. TCPX 连接状态异常
3. 内存注册问题

**调试方法**：
```bash
# 检查 TCPX 环境变量
env | grep NCCL

# 应该看到：
# NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
# NCCL_GPUDIRECTTCPX_CTRL_DEV=eth1,eth2,eth3,eth4
# NCCL_GPUDIRECTTCPX_FORCE_ACK=0
# NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000
# NCCL_NSOCKS_PERTHREAD=8
# NCCL_NTHREADS=4
```

### **诊断 2: try_release_oldest() 无限循环**

**症状**：
```
[DEBUG] GPU 0 channel 0 window FULL, trying to release oldest...
<卡住，没有进度>
```

**可能原因**：
1. `tcpx_test()` 一直返回 `tcpxInternalError`（rc != 0）
2. 请求永远不会进入 `transmitting` 队列
3. TCPX 后台线程没有工作

**调试方法**：
在 `src/sliding_window.cc` 的 `try_release_oldest()` 中添加日志：

```cpp
int rc = tcpx_test(oldest_req, &done, &size);

if (rc != 0) {
  static int warn_count = 0;
  if (warn_count++ < 10) {
    std::cerr << "[DEBUG] tcpx_test returned rc=" << rc
              << " (not ready), chunk_idx=" << oldest_idx << std::endl;
  }
  return 1;
}
```

### **诊断 3: 窗口大小异常**

**症状**：
```
[DEBUG] GPU 0 chunk 0 → channel 0 (window size=16/16)
```
第一个 chunk 就显示窗口满了

**可能原因**：
1. `SlidingWindow` 初始化错误
2. 之前的迭代没有正确清理

**调试方法**：
检查 `windows[gpu_id][channel_id]` 的初始化：
```cpp
// 应该在每次迭代开始时清空
for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
  for (int ch_id = 0; ch_id < num_channels; ch_id++) {
    windows[gpu_id][ch_id]->clear();
  }
}
```

## 🔧 **进一步调试手段**
### 0. 启用 TCPX TRACE（强烈推荐）

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
```

查看要点：
- rq.next_transmitting 指向的请求（地址/序号）是否变化
- 队列长度（active/transmitting/inactive）是否在缩短
- 是否有 tcpxCommProgress() 调用痕迹

速查：done=0 且长时间无 next_transmitting 变化 → 进度驱动不足；done=1 但未调用 consumed → 收端未释放窗口


### **1. 添加 TCPX 内部日志**

设置环境变量：
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

这会让 TCPX 插件打印内部日志。

### **2. 使用 strace 跟踪系统调用**

```bash
# 服务器端
strace -f -e trace=network,poll,epoll_wait -o /tmp/server_strace.log \
  ./tests/test_tcpx_perf_orchestrator server 8 4 20000 > logs/server_debug.log 2>&1
```

查看是否卡在某个系统调用上。

### **3. 使用 gdb 附加到进程**

```bash
# 找到进程 PID
ps aux | grep test_tcpx_perf_orchestrator

# 附加 gdb
gdb -p <PID>

# 查看当前调用栈
(gdb) thread apply all bt

# 查看所有线程
(gdb) info threads
```

### **4. 检查 TCPX 后台线程**

TCPX 使用后台线程处理网络 I/O。检查线程是否在运行：

```bash
# 找到进程 PID
ps aux | grep test_tcpx_perf_orchestrator

# 查看线程
ps -T -p <PID>

# 应该看到多个线程（主线程 + TCPX 后台线程）
```

### **5. 简化测试**

创建一个最小测试：只测试 1 个 GPU，1 个 channel，1 个 chunk：

```bash
# 修改 test_step3_bandwidth.sh
CHANNELS=1  # 只用 1 个 channel

# 修改 test_tcpx_perf_orchestrator.cc
const size_t test_size_per_gpu = 524288;  // 只传 1 个 chunk (512KB)
```

如果最小测试能工作，逐步增加复杂度。

## 📝 **报告问题时提供的信息**

1. **最新的服务器日志**（最后 100 行）
2. **最新的客户端日志**（最后 100 行）
3. **环境变量**：`env | grep NCCL`
4. **TCPX 版本**：`ls -la /usr/local/lib/libnccl-net.so*`
5. **卡住的确切位置**（从调试日志中找到）

## 🚀 **下一步**

1. 运行测试并收集新的调试日志
2. 根据上面的诊断方法分析卡住的位置
3. 如果需要，添加更多调试日志
4. 分享最新的日志和分析结果

准备好调试了！🔍

