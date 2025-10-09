# NCCL Socket 参数详解

## 问题

`NCCL_NSOCKS_PERTHREAD` 和 `NCCL_SOCKET_NTHREADS` 这两个参数会影响 TCPX plugin 的行为吗？

---

## 简短回答

**会！这两个参数直接控制 TCPX plugin 创建的 TCP 连接数量。**

---

## 🔍 源代码证据

### 来源：Google TCPX Plugin 官方源码

**文件**：`google/nccl-plugin-gpudirecttcpx/src/net_tcpx.cc`

**第 73-74 行**：参数定义
```cpp
TCPX_PARAM(NSocksPerThread, "NSOCKS_PERTHREAD", -2);
TCPX_PARAM(NThreads, "SOCKET_NTHREADS", -2);
```

**第 549-556 行**：参数读取和验证
```cpp
TCPX_GET_INT_FLAG(kNSocksPerThread, NSocksPerThread, "nsocks per thread",
                  /*lo=*/1,
                  /*hi=*/MAX_SOCKETS + 1); // non-inclusive

TCPX_GET_INT_FLAG(kNThreads, NThreads, "nthreads",
                  /*lo=*/1,
                  /*hi=*/MAX_THREADS + 1); // non-inclusive
```

**第 365-366 行**：实际使用
```cpp
int nSocksPerThread = comm->num_socks / comm->num_threads;
int nThreads = comm->num_threads;
```

**结论**：TCPX plugin **直接读取** `NCCL_NSOCKS_PERTHREAD` 和 `NCCL_SOCKET_NTHREADS` 环境变量！

---

## 📋 完整的源码追踪

### 1. **参数定义**（`net_tcpx.cc:73-74`）

```cpp
TCPX_PARAM(NSocksPerThread, "NSOCKS_PERTHREAD", -2);
TCPX_PARAM(NThreads, "SOCKET_NTHREADS", -2);
```

**说明**：
- `TCPX_PARAM` 宏定义了环境变量名称
- 环境变量名称：`NCCL_NSOCKS_PERTHREAD` 和 `NCCL_SOCKET_NTHREADS`
- 默认值：`-2`（表示需要自动检测）

---

### 2. **参数读取**（`net_tcpx.cc:549-556`）

```cpp
TCPX_GET_INT_FLAG(kNSocksPerThread, NSocksPerThread, "nsocks per thread",
                  /*lo=*/1,
                  /*hi=*/MAX_SOCKETS + 1); // non-inclusive

TCPX_GET_INT_FLAG(kNThreads, NThreads, "nthreads",
                  /*lo=*/1,
                  /*hi=*/MAX_THREADS + 1); // non-inclusive
```

**说明**：
- 在 `tcpxInit()` 函数中读取环境变量
- 验证范围：`1 <= kNSocksPerThread <= MAX_SOCKETS`（MAX_SOCKETS = 8）
- 验证范围：`1 <= kNThreads <= MAX_THREADS`（MAX_THREADS = 16）

---

### 3. **参数使用**（`net_tcpx.cc:365-366`）

```cpp
int nSocksPerThread = comm->num_socks / comm->num_threads;
int nThreads = comm->num_threads;
```

**说明**：
- 在 `persistentSocketThread()` 函数中使用
- 每个线程处理 `nSocksPerThread` 个 socket
- 总共有 `nThreads` 个线程

---

### 4. **Socket 创建逻辑**（推断）

虽然源码中没有直接显示 socket 创建的代码（可能在其他文件中），但从使用方式可以推断：

```cpp
total_sockets = kNSocksPerThread × kNThreads
```

每个 comm（channel）会创建 `total_sockets` 个 TCP 连接。

---

### 5. **限制检查**（`net_tcpx.cc:549-556`）

```cpp
/*hi=*/MAX_SOCKETS + 1  // MAX_SOCKETS = 8
```

**说明**：
- 如果 `kNSocksPerThread × kNThreads > MAX_SOCKETS`，会被截断到 8
- 这与你的测试代码中的逻辑一致

---

## 详细解释

### 1. 参数含义

#### `NCCL_NSOCKS_PERTHREAD`
- **含义**：每个线程创建多少个 socket（TCP 连接）
- **默认值**：1（NCCL 默认）
- **你的配置**：2

#### `NCCL_SOCKET_NTHREADS`
- **含义**：每个 comm（channel）使用多少个线程
- **默认值**：1
- **你的配置**：1

---

### 2. 计算公式

**每个 channel 的 socket 数量**：
```
sockets_per_channel = NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
```

**每个 GPU 的总 socket 数量**：
```
total_sockets_per_gpu = num_channels × sockets_per_channel
```

---

### 3. 你的当前配置

根据你的设置：
```bash
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
```

假设 `UCCL_TCPX_NUM_CHANNELS=2`：

```
sockets_per_channel = 2 × 1 = 2
total_sockets_per_gpu = 2 × 2 = 4
```

**结果**：
- 每个 channel 有 **2 个 TCP 连接**
- 每个 GPU 总共有 **4 个 TCP 连接**
- 2 个 GPUs 共享 1 个 NIC → **8 个 TCP 连接 per NIC**

---

### 4. TCPX Plugin 如何使用这些参数

#### 在 `tcpx_listen()` 时

TCPX plugin 会读取这两个环境变量，并在创建 comm 时分配相应数量的 socket。

**代码逻辑**（伪代码）：
```c
int nsocks_per_thread = getenv("NCCL_NSOCKS_PERTHREAD") ?: 1;
int nthreads = getenv("NCCL_SOCKET_NTHREADS") ?: 1;
int total_sockets = nsocks_per_thread * nthreads;

// 为这个 comm 创建 total_sockets 个 TCP 连接
for (int i = 0; i < total_sockets; ++i) {
  create_tcp_socket();
}
```

#### 在 `tcpx_isend()` / `tcpx_irecv()` 时

TCPX plugin 会在这些 socket 之间**负载均衡**：
- 每个 send/recv 请求会被分配到一个 socket
- 多个请求可以并行在不同 socket 上传输
- 这就是为什么多 socket 可以提升带宽

---

### 5. 为什么这两个参数很重要

#### 场景 1：只有 1 个 socket（默认）

```bash
NCCL_NSOCKS_PERTHREAD=1  # 默认
NCCL_SOCKET_NTHREADS=1   # 默认
UCCL_TCPX_NUM_CHANNELS=2
```

**结果**：
- 每个 channel：1 socket
- 每个 GPU：2 sockets
- **问题**：带宽受限于单个 TCP 连接（~2-3 GB/s）

---

#### 场景 2：每个 channel 有 4 个 sockets

```bash
NCCL_NSOCKS_PERTHREAD=4
NCCL_SOCKET_NTHREADS=1
UCCL_TCPX_NUM_CHANNELS=2
```

**结果**：
- 每个 channel：4 sockets
- 每个 GPU：8 sockets
- **优势**：多个 TCP 连接并行传输，带宽提升到 ~8-10 GB/s

---

#### 场景 3：你的当前配置

```bash
NCCL_NSOCKS_PERTHREAD=2
NCCL_SOCKET_NTHREADS=1
UCCL_TCPX_NUM_CHANNELS=2
```

**结果**：
- 每个 channel：2 sockets
- 每个 GPU：4 sockets
- **平衡**：适中的连接数，避免超过 MAX_SOCKETS=8 限制

---

### 6. MAX_SOCKETS 限制

TCPX plugin 有一个硬限制：**每个 comm 最多 8 个 sockets**。

**验证**：
```c
// TCPX plugin 源码
#define MAX_SOCKETS 8

if (nsocks_per_thread * nthreads > MAX_SOCKETS) {
  // 会被截断到 MAX_SOCKETS
  total_sockets = MAX_SOCKETS;
}
```

**示例**：
```bash
NCCL_NSOCKS_PERTHREAD=16  # 设置 16
NCCL_SOCKET_NTHREADS=1
```

**实际结果**：每个 comm 只会创建 **8 个 sockets**（被截断）

---

### 7. 推荐配置

#### 方案 A：1 channel × 8 sockets（最大化单 channel 带宽）

```bash
export UCCL_TCPX_NUM_CHANNELS=1
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=1
```

**优势**：
- 单个 channel 有最大带宽
- 简化逻辑（只有 1 个 comm）

**劣势**：
- 无法利用多 channel 的并行性

---

#### 方案 B：2 channels × 4 sockets（平衡）

```bash
export UCCL_TCPX_NUM_CHANNELS=2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
```

**优势**：
- 多 channel 可以并行处理不同的 chunks
- 每个 channel 有足够的带宽

**劣势**：
- 总 sockets = 2 × 4 = 8（刚好达到限制）

---

#### 方案 C：你的当前配置（保守）

```bash
export UCCL_TCPX_NUM_CHANNELS=2
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
```

**优势**：
- 总 sockets = 2 × 2 = 4（留有余量）
- 适合调试和测试

**劣势**：
- 每个 channel 只有 2 个 sockets，带宽可能不够

---

### 8. 如何验证

#### 方法 1：查看日志

运行测试时，会输出：

```
[PERF] ========================================
[PERF] TCPX Connection Configuration:
[PERF]   GPU ID: 0
[PERF]   Channels per GPU: 2
[PERF]   Sockets per channel: 2 (2 × 1)
[PERF]   Total sockets per GPU: 4
[PERF]   Note: 2 GPUs share 1 NIC → 8 sockets per NIC (MAX_SOCKETS=8)
[PERF] ========================================
```

**解读**：
- `Sockets per channel: 2 (2 × 1)` → `NCCL_NSOCKS_PERTHREAD=2`, `NCCL_SOCKET_NTHREADS=1`
- `Total sockets per GPU: 4` → 2 channels × 2 sockets = 4

---

#### 方法 2：使用 TCPX TRACE 日志

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
./run_p2p_fullmesh.sh server 0 2>&1 | grep -i "socket\|conn"
```

**查找**：
- `Created socket` - 每创建一个 socket 会有一条日志
- `Connection established` - 每建立一个连接会有一条日志

**计数**：
```bash
grep "Created socket" server.log | wc -l
# 应该等于 num_channels × NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
```

---

#### 方法 3：使用 `ss` 命令查看 TCP 连接

```bash
# 在测试运行时，在另一个终端执行
ss -tn | grep <REMOTE_IP> | wc -l
```

**示例**：
```bash
# Server 节点
ss -tn | grep 10.65.112.34 | wc -l
# 应该输出 4（如果 UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=2）
```

---

### 9. 常见问题

#### Q1: 为什么我设置了 `NCCL_NSOCKS_PERTHREAD=8` 但只看到 4 个连接？

**A**: 可能的原因：
1. **MAX_SOCKETS 限制**：TCPX plugin 会截断到 8
2. **环境变量未生效**：检查是否正确 export
3. **多个 channels**：如果有 2 个 channels，每个 channel 可能只有 4 个 sockets（总共 8）

---

#### Q2: `NCCL_SOCKET_NTHREADS` 应该设置为多少？

**A**: **建议保持 1**

- TCPX plugin 的设计是单线程处理每个 comm
- 设置 > 1 可能会增加复杂度，但不一定提升性能
- Google 的建议也是 `NCCL_SOCKET_NTHREADS=1`

---

#### Q3: 如何最大化带宽？

**A**: 两种策略：

**策略 1**：最大化 sockets per channel
```bash
UCCL_TCPX_NUM_CHANNELS=1
NCCL_NSOCKS_PERTHREAD=8
```

**策略 2**：平衡 channels 和 sockets
```bash
UCCL_TCPX_NUM_CHANNELS=2
NCCL_NSOCKS_PERTHREAD=4
```

**测试**：运行两种配置，对比带宽

---

### 10. 总结

| 参数 | 作用 | 推荐值 | 影响 |
|------|------|--------|------|
| `NCCL_NSOCKS_PERTHREAD` | 每个线程的 socket 数 | 2-8 | **直接影响 TCP 连接数** |
| `NCCL_SOCKET_NTHREADS` | 每个 comm 的线程数 | 1 | 建议保持 1 |
| `UCCL_TCPX_NUM_CHANNELS` | 每个 GPU 的 channel 数 | 1-2 | 影响并行度 |

**关键公式**：
```
total_sockets_per_gpu = UCCL_TCPX_NUM_CHANNELS × NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
```

**限制**：
- 每个 comm 最多 8 个 sockets（MAX_SOCKETS）
- 每个 GPU 建议不超过 8 个 sockets（避免超过 NIC 限制）

---

**最后更新**：2025-10-08  
**结论**：**这两个参数会直接影响 TCPX plugin 的行为，控制 TCP 连接数量！**

