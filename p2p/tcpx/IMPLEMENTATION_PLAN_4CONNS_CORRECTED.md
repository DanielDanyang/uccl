# 实施计划：每 GPU 4 个 TCPX 连接（修正版）

## 🎯 理解

### TCPX 的限制
```c
#define MAX_SOCKETS 8  // 每个 NIC 最多 8 个 sockets
```

**关键点**：
- `MAX_SOCKETS=8` 是**每个 NIC** 的限制（不是每个 comm）
- 一个 NIC 上的所有 comms 共享这 8 个 socket 槽位
- 2 个 GPUs 共享 1 个 NIC → 每个 GPU 最多 4 个 connections

### 正确的拓扑（GCP A3-high）
```
GPU 0,1 → eth1 (共享)
GPU 2,3 → eth2 (共享)
GPU 4,5 → eth3 (共享)
GPU 6,7 → eth4 (共享)

每个 GPU 只能用 1 个 NIC！
```

### 正确的架构
```
每个 NIC 最多 8 个 sockets
2 个 GPUs 共享 1 个 NIC

连接分配：
- GPU 0 → eth1，4 个 TCPX connections
- GPU 1 → eth1，4 个 TCPX connections
- 总共：eth1 上有 8 个 sockets（4+4，达到上限）

同理：
- GPU 2,3 → eth2（各 4 个 connections）
- GPU 4,5 → eth3（各 4 个 connections）
- GPU 6,7 → eth4（各 4 个 connections）

结果：
- 每个 GPU：4 个 connections
- 每个 NIC：8 个 connections（来自 2 个 GPUs）
- 每个 NIC 可达 ~21.26 GB/s
```

---

## 📋 实施步骤

### 阶段 1：修改 ChannelManager

#### 1.1 添加 GPU → NIC 映射函数

在 `src/channel_manager.cc` 中添加：

```cpp
namespace {
  // GCP A3-high topology: 4 NICs, 8 GPUs
  // GPU 0,1 → eth1 (dev 0)
  // GPU 2,3 → eth2 (dev 1)
  // GPU 4,5 → eth3 (dev 2)
  // GPU 6,7 → eth4 (dev 3)
  int get_nic_for_gpu(int gpu_id) {
    // Simple mapping: gpu_id / 2
    int nic_dev = gpu_id / 2;
    
    // Validate
    int tcpx_dev_count = tcpx_get_device_count();
    if (nic_dev >= tcpx_dev_count) {
      std::cerr << "[ChannelManager] Warning: Computed NIC " << nic_dev
                << " exceeds available TCPX devices (" << tcpx_dev_count
                << "). Using NIC 0." << std::endl;
      return 0;
    }
    
    return nic_dev;
  }
  
  // Optional: Allow override via environment variable
  int get_nic_for_gpu_with_override(int gpu_id) {
    const char* override_env = std::getenv("UCCL_TCPX_GPU_NIC_MAP");
    if (override_env) {
      // Format: "0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3"
      // Parse and return the mapped NIC
      // (Implementation omitted for brevity)
    }
    return get_nic_for_gpu(gpu_id);
  }
}
```

#### 1.2 修改构造函数

```cpp
ChannelManager::ChannelManager(int gpu_id) 
    : gpu_id_(gpu_id) {
  
  // Fixed: always create 4 connections per GPU
  const int NUM_CONNECTIONS = 4;
  num_channels_ = NUM_CONNECTIONS;
  
  // Get the single NIC for this GPU
  int nic_dev = get_nic_for_gpu(gpu_id);
  
  // Get NIC properties
  tcpx_net_properties props{};
  if (tcpx_get_properties(nic_dev, &props) != 0) {
    std::cerr << "[ChannelManager] Failed to get properties for NIC " << nic_dev << std::endl;
    num_channels_ = 0;
    return;
  }
  
  const char* nic_name = props.name ? props.name : "unknown";
  std::cout << "[ChannelManager] GPU " << gpu_id 
            << " will use NIC " << nic_dev << " (" << nic_name << ")" << std::endl;
  
  // Create 4 connections, all using the same NIC
  channels_.resize(NUM_CONNECTIONS);
  
  for (int i = 0; i < NUM_CONNECTIONS; ++i) {
    ChannelResources& ch = channels_[i];
    
    ch.channel_id = i;
    ch.net_dev = nic_dev;  // All connections use the same NIC
    ch.nic_name = nic_name;
    
    std::cout << "[ChannelManager] Connection " << i 
              << " → netDev " << ch.net_dev 
              << " (" << ch.nic_name << ")" << std::endl;
    
    // Initialize connection handles
    ch.listen_comm = nullptr;
    ch.recv_comm = nullptr;
    ch.send_comm = nullptr;
    
    // Initialize device handles
    ch.recv_dev_handle = ch.recv_dev_handle_storage.data();
    ch.send_dev_handle = ch.send_dev_handle_storage.data();
    std::memset(ch.recv_dev_handle_storage.data(), 0, ch.recv_dev_handle_storage.size());
    std::memset(ch.send_dev_handle_storage.data(), 0, ch.send_dev_handle_storage.size());
    
    // Initialize memory handle
    ch.mhandle = nullptr;
    
    // Create sliding window (16 for recv, will be 12 for send)
    ch.sliding_window = new SlidingWindow(16);
    
    // Initialize statistics
    ch.bytes_transferred = 0;
    ch.chunks_processed = 0;
  }
  
  std::cout << "[ChannelManager] Created " << NUM_CONNECTIONS 
            << " connections for GPU " << gpu_id 
            << " on NIC " << nic_dev << " (" << nic_name << ")"
            << std::endl;
  std::cout << "[ChannelManager] Note: 2 GPUs share this NIC, "
            << "total sockets on NIC will be 8 (4+4)" << std::endl;
}
```

#### 1.3 更新头文件签名

在 `include/channel_manager.h` 中：

```cpp
class ChannelManager {
public:
  /**
   * @brief Constructor
   * @param gpu_id GPU device ID (determines which NIC to use)
   * 
   * Creates 4 TCPX connections for this GPU, all using the same NUMA-local NIC.
   * GPU-to-NIC mapping: GPU 0,1→eth1; GPU 2,3→eth2; GPU 4,5→eth3; GPU 6,7→eth4
   */
  ChannelManager(int gpu_id);  // 移除 num_channels 参数
  
  // ... rest of the class
};
```

---

### 阶段 2：修改 test_tcpx_perf_multi.cc

#### 2.1 移除环境变量配置

删除或注释掉之前添加的自动配置代码：

```cpp
// 删除这些代码：
// int num_channels_env = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 1);
// int target_total_sockets = 8;
// int socks_per_channel = std::max(1, target_total_sockets / num_channels_env);
// setenv("NCCL_NSOCKS_PERTHREAD", ..., 0);
// setenv("NCCL_SOCKET_NTHREADS", ..., 0);
```

#### 2.2 更新 ChannelManager 创建

```cpp
// Server 端
std::cout << "[PERF] Starting SERVER mode" << std::endl;

// Fixed: always create 4 connections per GPU
ChannelManager mgr(gpu_id);  // 只传 gpu_id
int num_channels = mgr.get_num_channels();  // 应该返回 4

std::cout << "[PERF] Created " << num_channels << " connections for GPU " << gpu_id << std::endl;
```

```cpp
// Client 端
std::cout << "[PERF] Starting CLIENT mode" << std::endl;

// Bootstrap 接收 handles
std::vector<ncclNetHandle_v7> handles;
if (bootstrap_client_recv_handles(bootstrap_fd, handles) != 0) {
  std::cerr << "[ERROR] bootstrap_client_recv_handles failed" << std::endl;
  close(bootstrap_fd);
  return 1;
}

std::cout << "[PERF] Received " << handles.size() << " handles from server" << std::endl;

// Create ChannelManager (will create 4 connections)
ChannelManager mgr(gpu_id);  // 只传 gpu_id
int num_channels = mgr.get_num_channels();  // 应该返回 4

// Verify handle count matches
if (handles.size() != static_cast<size_t>(num_channels)) {
  std::cerr << "[ERROR] Handle count mismatch: expected " << num_channels
            << ", got " << handles.size() << std::endl;
  close(bootstrap_fd);
  return 1;
}
```

#### 2.3 更新日志输出

```cpp
std::cout << "[PERF] ========================================" << std::endl;
std::cout << "[PERF] TCPX Connection Configuration:" << std::endl;
std::cout << "[PERF]   GPU ID: " << gpu_id << std::endl;
std::cout << "[PERF]   Connections per GPU: 4" << std::endl;
std::cout << "[PERF]   NIC: " << (gpu_id / 2) << " (shared with GPU " 
          << (gpu_id % 2 == 0 ? gpu_id + 1 : gpu_id - 1) << ")" << std::endl;
std::cout << "[PERF]   Total sockets on NIC: 8 (4 from this GPU + 4 from peer)" << std::endl;
std::cout << "[PERF]   Target bandwidth: ~21.26 GB/s per NIC" << std::endl;
std::cout << "[PERF] ========================================" << std::endl;
```

#### 2.4 保持现有的 round-robin 逻辑

现有的 chunk 分配逻辑已经是 round-robin，无需修改：
```cpp
// 这段代码保持不变
int channel_id = global_chunk_idx % num_channels;  // 0-3
ChannelResources& ch = mgr.get_channel(channel_id);
```

---

### 阶段 3：编译和测试

#### 3.1 编译
```bash
cd /home/daniel/uccl/p2p/tcpx
make clean
make -j
```

#### 3.2 运行测试

**Server 端（Node 0, GPU 0）**：
```bash
export UCCL_TCPX_PERF_SIZE=67108864  # 64MB
export UCCL_TCPX_PERF_ITERS=20

./tests/test_tcpx_perf_multi server 0
```

**Client 端（Node 1, GPU 0）**：
```bash
export UCCL_TCPX_PERF_SIZE=67108864
export UCCL_TCPX_PERF_ITERS=20

./tests/test_tcpx_perf_multi client <SERVER_IP> 0
```

#### 3.3 验证日志

查找以下关键信息：

1. **NIC 选择**：
   ```
   [ChannelManager] GPU 0 will use NIC 0 (eth1)
   ```

2. **Connection 创建**：
   ```
   [ChannelManager] Connection 0 → netDev 0 (eth1)
   [ChannelManager] Connection 1 → netDev 0 (eth1)
   [ChannelManager] Connection 2 → netDev 0 (eth1)
   [ChannelManager] Connection 3 → netDev 0 (eth1)
   [ChannelManager] Created 4 connections for GPU 0 on NIC 0 (eth1)
   [ChannelManager] Note: 2 GPUs share this NIC, total sockets on NIC will be 8 (4+4)
   ```

3. **Bootstrap 握手**：
   ```
   [Bootstrap] Sent 4 handles to client
   [Bootstrap] Received 4 handles from server
   ```

4. **连接建立**：
   ```
   [ChannelManager] All 4 channels listening successfully
   [ChannelManager] All 4 channels accepted successfully
   [ChannelManager] All 4 channels connected successfully
   ```

5. **Chunk 分配**（应该看到 channel_id 从 0-3 循环）：
   ```
   [DEBUG][SERVER] chunk=0 channel=0 ...
   [DEBUG][SERVER] chunk=1 channel=1 ...
   [DEBUG][SERVER] chunk=2 channel=2 ...
   [DEBUG][SERVER] chunk=3 channel=3 ...
   [DEBUG][SERVER] chunk=4 channel=0 ...  # 循环回到 0
   ```

6. **带宽结果**：
   ```
   [PERF] Avg: XXX ms, BW: YY.YY GB/s
   ```
   - 目标：接近 21.26 GB/s（单 NIC 理论上限）

---

## 📊 成功标准

### 必须满足
1. ✅ 每个 GPU 创建 4 个 connections
2. ✅ 所有 4 个 connections 使用同一个 NIC
3. ✅ GPU 0,1 使用 eth1；GPU 2,3 使用 eth2；等等
4. ✅ Bootstrap 成功交换 4 个 handles
5. ✅ 所有 4 个 connections 成功建立
6. ✅ Chunks 正确地 round-robin 分配到 4 个 connections
7. ✅ 测试稳定完成，无 deadlock

### 期望达到
1. ✅ 单 GPU pair 带宽：接近 21.26 GB/s
2. ✅ 比单 connection 基线提升 4 倍
3. ✅ 2 个 GPUs 同时使用同一个 NIC 时，总带宽不超过 21.26 GB/s

---

## 🧪 测试场景

### 场景 1：单 GPU pair（同一个 NIC）
```bash
# Server: GPU 0 (eth1)
./tests/test_tcpx_perf_multi server 0

# Client: GPU 1 (eth1)
./tests/test_tcpx_perf_multi client <SERVER_IP> 1
```
**预期**：
- 两个 GPUs 共享 eth1
- 总共 8 个 connections（4+4）
- 带宽接近 21.26 GB/s

### 场景 2：单 GPU pair（不同 NIC）
```bash
# Server: GPU 0 (eth1)
./tests/test_tcpx_perf_multi server 0

# Client: GPU 2 (eth2)
./tests/test_tcpx_perf_multi client <SERVER_IP> 2
```
**预期**：
- GPU 0 使用 eth1（4 connections）
- GPU 2 使用 eth2（4 connections）
- 不共享 NIC，无竞争
- 带宽接近 21.26 GB/s

### 场景 3：多 GPU pairs（压力测试）
同时运行多个 GPU pairs，验证 NIC 共享是否正常工作。

---

## 🚨 常见问题

### 问题 1：NIC 索引不匹配

**症状**：`netDev 0` 不是 `eth1`

**解决**：使用 `tcpx_get_properties` 查找正确的 NIC 索引

### 问题 2：超过 MAX_SOCKETS 限制

**症状**：第 3 个 GPU 尝试使用同一个 NIC 时失败

**原因**：3 个 GPUs × 4 connections = 12 > 8

**解决**：确保每个 NIC 只被 2 个 GPUs 使用

### 问题 3：带宽没有提升

**可能原因**：
- Chunk size 太小
- 窗口大小不合适
- NUMA 亲和性问题

**调试**：启用 TCPX TRACE 查看详细日志

