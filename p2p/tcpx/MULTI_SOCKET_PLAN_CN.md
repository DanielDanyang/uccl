# TCPX 多 Socket 流水线并行实施计划

## 📊 问题诊断

### 原始问题
- 用户报告：`UCCL_TCPX_NUM_CHANNELS=4` 比 `UCCL_TCPX_NUM_CHANNELS=1` 还要慢
- 预期：4 个 channels 应该提供更好的流水线并行

### 根本原因分析

#### 误解：Channel ≠ Socket
之前我们误以为：
```
1 channel = 1 TCPX connection = 1 TCP socket
```

**实际情况**（从 TCPX 源码发现）：
```
1 channel = 1 tcpxComm = N sockets (N 由 NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS 决定)
```

#### TCPX 架构（来自 nccl-plugin-gpudirecttcpx 源码）

**关键文件**：
- `src/macro.h:36` - `#define MAX_SOCKETS 8`
- `src/common.h:104` - `struct tcpxHandle { int num_socks; int num_threads; }`
- `src/connect.cc:198-200` - GCP 默认配置：`autoNt=6, autoNs=1`
- `src/connect.cc:622` - Connect 循环：`for (int i = comm->num_socks; i >= 0; i--)`

**连接建立过程**：
1. `tcpxListen()` 创建 listen comm，设置 `num_socks` 和 `num_threads`
2. `tcpxConnect()` 在循环中建立 `num_socks + 1` 个连接（+1 是 control socket）
3. `tcpxAccept()` 在循环中接受 `num_socks + 1` 个连接
4. 每个 socket 由独立的线程管理（work queue 模式）

**默认配置问题**：
```bash
# GCP 默认（如果不设置环境变量）
NCCL_SOCKET_NTHREADS=6
NCCL_NSOCKS_PERTHREAD=1
# 结果：每个 comm 有 6 个 sockets

# 但是！如果用户没有设置这些变量，TCPX 可能回退到更保守的配置
# 导致每个 comm 只有 1 个 socket
```

#### 当前实现的实际行为

**场景 1：UCCL_TCPX_NUM_CHANNELS=1（默认）**
```
1 GPU 进程 → 1 channel → 1 tcpxComm → 1 socket (默认)
总连接数：1
```

**场景 2：UCCL_TCPX_NUM_CHANNELS=4（用户尝试）**
```
1 GPU 进程 → 4 channels → 4 tcpxComms → 4 × 1 socket = 4 sockets
总连接数：4
```

**问题**：
- 4 个独立的 comms 可能分散到不同的 NICs
- 没有真正的流水线并行（每个 comm 内部只有 1 个 socket）
- 增加了管理开销（4 个独立的滑动窗口、进度驱动等）
- 反而比 1 个 channel 慢！

---

## 🎯 解决方案

### 方案 A：单 Channel + 多 Sockets（推荐）

**配置**：
```bash
export UCCL_TCPX_NUM_CHANNELS=1
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=1
```

**结果**：
```
1 GPU 进程 → 1 channel → 1 tcpxComm → 8 sockets
总连接数：8 ✓
```

**优点**：
- 简单直接，符合 TCPX 设计理念
- 所有 sockets 在同一个 comm 内，共享进度驱动
- 单一滑动窗口，管理简单
- 无需修改代码（只需环境变量）

**缺点**：
- 所有流量通过一个 channel，可能有轻微的序列化开销

---

### 方案 B：双 Channel + 每个 4 Sockets

**配置**：
```bash
export UCCL_TCPX_NUM_CHANNELS=2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
```

**结果**：
```
1 GPU 进程 → 2 channels → 2 tcpxComms → 2 × 4 sockets = 8 sockets
总连接数：8 ✓
```

**优点**：
- 更好的负载均衡（2 个独立的 comms）
- 可以绑定到不同的 NICs（如果需要）

**缺点**：
- 需要管理 2 个滑动窗口
- 需要轮询 2 个 channels 的进度

---

### 方案 C：四 Channel + 每个 2 Sockets

**配置**：
```bash
export UCCL_TCPX_NUM_CHANNELS=4
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
```

**结果**：
```
1 GPU 进程 → 4 channels → 4 tcpxComms → 4 × 2 sockets = 8 sockets
总连接数：8 ✓
```

**优点**：
- 最细粒度的控制

**缺点**：
- 管理开销最大（4 个滑动窗口）
- 可能不如方案 A 和 B 高效

---

## 🔧 代码修改

### 已实现的修改

#### 1. `tests/test_tcpx_perf_multi.cc`

**添加自动配置逻辑**（line 177-219）：
```cpp
// 根据 UCCL_TCPX_NUM_CHANNELS 自动计算 NCCL_NSOCKS_PERTHREAD
int num_channels_env = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 1);
int target_total_sockets = 8;
int socks_per_channel = std::max(1, target_total_sockets / num_channels_env);

if (!std::getenv("NCCL_NSOCKS_PERTHREAD")) {
  std::string nsocks_str = std::to_string(socks_per_channel);
  setenv("NCCL_NSOCKS_PERTHREAD", nsocks_str.c_str(), 0);
}
if (!std::getenv("NCCL_SOCKET_NTHREADS")) {
  setenv("NCCL_SOCKET_NTHREADS", "1", 0);
}
```

**添加详细日志**（line 245-260）：
```cpp
std::cout << "[PERF] ========================================" << std::endl;
std::cout << "[PERF] TCPX Connection Configuration:" << std::endl;
std::cout << "[PERF]   Channels per GPU: " << num_channels << std::endl;
std::cout << "[PERF]   Sockets per channel: " << sockets_per_channel << std::endl;
std::cout << "[PERF]   Total sockets per GPU: " << total_expected_sockets << std::endl;
std::cout << "[PERF] ========================================" << std::endl;
```

---

## 🧪 测试计划

### 阶段 1：验证单 Channel + 8 Sockets（最高优先级）

**目标**：确认方案 A 能够创建 8 个 TCP 连接并提升带宽

**步骤**：

1. **编译**：
   ```bash
   cd /home/daniel/uccl/p2p/tcpx
   make clean
   make -j
   ```

2. **Server 端（Node 0）**：
   ```bash
   export UCCL_TCPX_NUM_CHANNELS=1
   export NCCL_NSOCKS_PERTHREAD=8
   export NCCL_SOCKET_NTHREADS=1
   export UCCL_TCPX_PERF_SIZE=67108864  # 64MB
   export UCCL_TCPX_PERF_ITERS=20
   
   ./tests/test_tcpx_perf_multi server 0
   ```

3. **Client 端（Node 1）**：
   ```bash
   export UCCL_TCPX_NUM_CHANNELS=1
   export NCCL_NSOCKS_PERTHREAD=8
   export NCCL_SOCKET_NTHREADS=1
   export UCCL_TCPX_PERF_SIZE=67108864
   export UCCL_TCPX_PERF_ITERS=20
   
   ./tests/test_tcpx_perf_multi client <SERVER_IP> 0
   ```

4. **验证日志**：
   - 查找：`[PERF] Total sockets per GPU: 8`
   - 查找：`NET/GPUDIRECTTCPX: Connected 8 socks`（TCPX 插件日志）
   - 查找：`NET/GPUDIRECTTCPX: Accepted 8 socks`（TCPX 插件日志）

5. **验证带宽**：
   - 目标：接近 21.26 GB/s（单 200Gbps NIC 理论上限）
   - 对比：与之前的 1 socket 配置对比

---

### 阶段 2：对比不同配置

**测试矩阵**：

| 配置 | Channels | Sockets/Channel | Total Sockets | 预期带宽 |
|------|----------|-----------------|---------------|----------|
| 基线 | 1 | 1 | 1 | ~2-3 GB/s |
| 方案 A | 1 | 8 | 8 | ~18-21 GB/s |
| 方案 B | 2 | 4 | 8 | ~18-21 GB/s |
| 方案 C | 4 | 2 | 8 | ~18-21 GB/s |

**运行脚本**：
```bash
# 创建测试脚本
cat > test_multi_socket.sh << 'EOF'
#!/bin/bash
CONFIGS=(
  "1:8"  # 1 channel, 8 sockets
  "2:4"  # 2 channels, 4 sockets each
  "4:2"  # 4 channels, 2 sockets each
)

for config in "${CONFIGS[@]}"; do
  IFS=':' read -r channels sockets <<< "$config"
  echo "=========================================="
  echo "Testing: $channels channels × $sockets sockets"
  echo "=========================================="
  
  export UCCL_TCPX_NUM_CHANNELS=$channels
  export NCCL_NSOCKS_PERTHREAD=$sockets
  export NCCL_SOCKET_NTHREADS=1
  
  # Run test...
  # (需要在两个节点上同步运行)
done
EOF
chmod +x test_multi_socket.sh
```

---

### 阶段 3：启用 TCPX TRACE 验证

**目标**：从 TCPX 插件内部确认 socket 数量

**步骤**：

1. **启用 TRACE**：
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=NET,INIT
   export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
   ```

2. **查找关键日志**：
   ```
   NET/GPUDIRECTTCPX: Using 1 threads and 8 sockets per thread
   NET/GPUDIRECTTCPX: connecting (0) through ...
   NET/GPUDIRECTTCPX: connecting (1) through ...
   ...
   NET/GPUDIRECTTCPX: connecting (7) through ...
   NET/GPUDIRECTTCPX: Connected 8 socks
   ```

3. **验证 socket 状态**：
   ```bash
   # 在测试运行时，在另一个终端查看
   ss -tn | grep <SERVER_IP> | wc -l
   # 应该看到 8 个 ESTABLISHED 连接（每个 GPU）
   ```

---

## 📈 成功标准

### 必须满足：
1. ✅ 日志显示：`Total sockets per GPU: 8`
2. ✅ TCPX 日志显示：`Connected 8 socks` 和 `Accepted 8 socks`
3. ✅ 带宽提升：从 ~2-3 GB/s（1 socket）提升到 ~18-21 GB/s（8 sockets）
4. ✅ 测试稳定完成，无 deadlock

### 期望达到：
1. ✅ 单 NIC 带宽接近 21.26 GB/s（200Gbps ÷ 8 = 25 GB/s 理论值，实际约 85%）
2. ✅ 不同配置（1×8, 2×4, 4×2）性能相近
3. ✅ 2 GPUs 共享 1 NIC 时，总带宽接近 NIC 上限

---

## 🚨 常见问题排查

### 问题 1：日志显示 "Connected 1 socks"

**原因**：环境变量没有生效

**解决**：
```bash
# 确认环境变量
env | grep NCCL

# 应该看到：
# NCCL_NSOCKS_PERTHREAD=8
# NCCL_SOCKET_NTHREADS=1
```

### 问题 2：带宽没有提升

**可能原因**：
1. NIC 绑定问题（多个 channels 分散到不同 NICs）
2. NUMA 亲和性问题
3. Chunk size 不合适

**调试**：
```bash
# 查看 NIC 使用情况
export NCCL_DEBUG=INFO
# 查找：Channel X → netDev Y
```

### 问题 3：连接建立失败

**可能原因**：
1. 端口范围不足（需要 8 个端口）
2. 防火墙限制

**解决**：
```bash
# 确保端口范围足够
export NCCL_GPUDIRECTTCPX_PORT_BEGIN=50000
export NCCL_GPUDIRECTTCPX_PORT_END=50100
```

---

## 📚 参考资料

### TCPX 源码关键位置
- `nccl-plugin-gpudirecttcpx/src/macro.h:36` - MAX_SOCKETS 定义
- `nccl-plugin-gpudirecttcpx/src/connect.cc:198-220` - Socket 数量计算
- `nccl-plugin-gpudirecttcpx/src/connect.cc:622-659` - Connect 循环
- `nccl-plugin-gpudirecttcpx/src/connect.cc:761-804` - Accept 循环

### NCCL 源码关键位置
- `thirdparty/nccl/src/transport/net.cc:1295` - irecv 调用（支持 subCount）
- `thirdparty/nccl/src/transport/net.cc:504` - p2pnChannels 使用

### 相关文档
- `HANDOFF_README.md` - 项目交接文档
- `DEBUG_GUIDE.md` - 调试指南
- `docs/AI_HANDOFF_PROMPT.md` - AI 交接提示（已更新）

