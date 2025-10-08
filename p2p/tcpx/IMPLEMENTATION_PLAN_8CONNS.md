# 实施计划：每 GPU 8 个 TCPX 连接

## 🎯 目标架构

### 核心理念
```
在 NIXL plugin 中，不要过度使用 "channel" 概念。
直接管理 TCPX connections：
- 1 GPU 进程 = 8 个独立的 TCPX connections
- 这 8 个 connections 分布在 2 个 NUMA-local NICs 上
- 每个 NIC 4 个 connections
```

### 具体映射（GCP A3-high）
```
NUMA 拓扑：
- NUMA 0: GPUs 0-3, eth1-2
- NUMA 1: GPUs 4-7, eth3-4

连接分配：
GPU 0 → eth1 (conn 0,1,2,3) + eth2 (conn 4,5,6,7) = 8 conns
GPU 1 → eth1 (conn 0,1,2,3) + eth2 (conn 4,5,6,7) = 8 conns
GPU 2 → eth1 (conn 0,1,2,3) + eth2 (conn 4,5,6,7) = 8 conns
GPU 3 → eth1 (conn 0,1,2,3) + eth2 (conn 4,5,6,7) = 8 conns
GPU 4 → eth3 (conn 0,1,2,3) + eth4 (conn 4,5,6,7) = 8 conns
GPU 5 → eth3 (conn 0,1,2,3) + eth4 (conn 4,5,6,7) = 8 conns
GPU 6 → eth3 (conn 0,1,2,3) + eth4 (conn 4,5,6,7) = 8 conns
GPU 7 → eth3 (conn 0,1,2,3) + eth4 (conn 4,5,6,7) = 8 conns

结果：
- 每个 NIC 上有 8 个 connections（2 GPUs × 4 conns）
- 符合 MAX_SOCKETS=8 的限制
- 每个 NIC 可以达到 ~21.26 GB/s 的上限
```

---

## 📋 实施步骤

### 阶段 1：修改 ChannelManager

#### 目标
将 `ChannelManager` 从"管理多个 channels"改为"管理 8 个独立的 connections"

#### 修改点

**1.1 构造函数逻辑**

当前逻辑：
```cpp
ChannelManager(int num_channels, int gpu_id)
// num_channels 由 UCCL_TCPX_NUM_CHANNELS 环境变量决定
// 选择 num_channels 个 NICs（round-robin）
```

新逻辑：
```cpp
ChannelManager(int gpu_id)  // 移除 num_channels 参数
// 固定创建 8 个 connections
// 根据 gpu_id 确定 NUMA-local NICs
// 前 4 个 connections 用第一个 NIC
// 后 4 个 connections 用第二个 NIC
```

**1.2 硬编码 GPU → NIC 映射**

```cpp
// 在 channel_manager.cc 中添加
namespace {
  // GCP A3-high NUMA topology
  struct GPUNICMapping {
    int nic1_dev;  // First NUMA-local NIC
    int nic2_dev;  // Second NUMA-local NIC
  };
  
  // Hardcoded mapping (can be overridden by env var)
  GPUNICMapping get_nics_for_gpu(int gpu_id) {
    // GPU 0-3 → eth1 (dev 0), eth2 (dev 1)
    // GPU 4-7 → eth3 (dev 2), eth4 (dev 3)
    if (gpu_id >= 0 && gpu_id <= 3) {
      return {0, 1};  // eth1, eth2
    } else if (gpu_id >= 4 && gpu_id <= 7) {
      return {2, 3};  // eth3, eth4
    } else {
      // Fallback: use first two NICs
      return {0, 1};
    }
  }
}
```

**1.3 创建 8 个 connections**

```cpp
ChannelManager::ChannelManager(int gpu_id) : gpu_id_(gpu_id) {
  // Fixed: always create 8 connections
  const int NUM_CONNECTIONS = 8;
  const int CONNS_PER_NIC = 4;
  
  // Get NUMA-local NICs for this GPU
  auto nic_mapping = get_nics_for_gpu(gpu_id);
  
  channels_.resize(NUM_CONNECTIONS);
  
  for (int i = 0; i < NUM_CONNECTIONS; ++i) {
    ChannelResources& ch = channels_[i];
    ch.channel_id = i;
    
    // First 4 connections use nic1, next 4 use nic2
    if (i < CONNS_PER_NIC) {
      ch.net_dev = nic_mapping.nic1_dev;
    } else {
      ch.net_dev = nic_mapping.nic2_dev;
    }
    
    // Get NIC properties
    tcpx_net_properties props{};
    tcpx_get_properties(ch.net_dev, &props);
    ch.nic_name = props.name ? props.name : "unknown";
    
    std::cout << "[ChannelManager] Connection " << i 
              << " → netDev " << ch.net_dev 
              << " (" << ch.nic_name << ")" << std::endl;
    
    // Initialize other fields...
    ch.listen_comm = nullptr;
    ch.recv_comm = nullptr;
    ch.send_comm = nullptr;
    ch.mhandle = nullptr;
    ch.sliding_window = new SlidingWindow(16);
    ch.bytes_transferred = 0;
    ch.chunks_processed = 0;
  }
  
  std::cout << "[ChannelManager] Created " << NUM_CONNECTIONS 
            << " connections for GPU " << gpu_id 
            << " (NICs: " << nic_mapping.nic1_dev << ", " << nic_mapping.nic2_dev << ")"
            << std::endl;
}
```

---

### 阶段 2：修改 test_tcpx_perf_multi.cc

#### 目标
移除 `UCCL_TCPX_NUM_CHANNELS` 环境变量，固定使用 8 个 connections

#### 修改点

**2.1 移除环境变量读取**

当前代码：
```cpp
int num_channels = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 1);
ChannelManager mgr(num_channels, gpu_id);
```

新代码：
```cpp
// Fixed: always use 8 connections per GPU
const int NUM_CONNECTIONS = 8;
ChannelManager mgr(gpu_id);  // 只传 gpu_id
int num_channels = mgr.get_num_channels();  // 应该返回 8
```

**2.2 更新日志输出**

```cpp
std::cout << "[PERF] ========================================" << std::endl;
std::cout << "[PERF] TCPX Connection Configuration:" << std::endl;
std::cout << "[PERF]   GPU ID: " << gpu_id << std::endl;
std::cout << "[PERF]   Total connections: " << num_channels << std::endl;
std::cout << "[PERF]   Connections per NIC: 4" << std::endl;
std::cout << "[PERF]   Target bandwidth: ~21.26 GB/s per NIC" << std::endl;
std::cout << "[PERF] ========================================" << std::endl;
```

**2.3 保持现有的 round-robin 逻辑**

现有的 chunk 分配逻辑已经是 round-robin，无需修改：
```cpp
int channel_id = global_chunk_idx % num_channels;  // 0-7
ChannelResources& ch = mgr.get_channel(channel_id);
```

---

### 阶段 3：更新 Bootstrap 协议

#### 目标
确保 server 和 client 都知道有 8 个 connections

#### 修改点

**3.1 Server 端**

```cpp
// server_listen_all 会创建 8 个 listen comms
std::vector<ncclNetHandle_v7> handles;
if (mgr.server_listen_all(handles) != 0) {
  std::cerr << "[ERROR] server_listen_all failed" << std::endl;
  return 1;
}
// handles.size() 应该是 8

// Bootstrap 发送 8 个 handles
if (bootstrap_server_send_handles(bootstrap_fd, handles) != 0) {
  std::cerr << "[ERROR] bootstrap_server_send_handles failed" << std::endl;
  return 1;
}
```

**3.2 Client 端**

```cpp
// Bootstrap 接收 handles
std::vector<ncclNetHandle_v7> handles;
if (bootstrap_client_recv_handles(bootstrap_fd, handles) != 0) {
  std::cerr << "[ERROR] bootstrap_client_recv_handles failed" << std::endl;
  return 1;
}
// handles.size() 应该是 8

// 创建 ChannelManager（会自动创建 8 个 connections）
ChannelManager mgr(gpu_id);

// 连接所有 8 个 connections
if (mgr.client_connect_all(handles) != 0) {
  std::cerr << "[ERROR] client_connect_all failed" << std::endl;
  return 1;
}
```

---

### 阶段 4：验证和测试

#### 4.1 编译
```bash
cd /home/daniel/uccl/p2p/tcpx
make clean
make -j
```

#### 4.2 运行测试

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

#### 4.3 验证日志

查找以下关键信息：

1. **Connection 创建**：
   ```
   [ChannelManager] Connection 0 → netDev 0 (eth1)
   [ChannelManager] Connection 1 → netDev 0 (eth1)
   [ChannelManager] Connection 2 → netDev 0 (eth1)
   [ChannelManager] Connection 3 → netDev 0 (eth1)
   [ChannelManager] Connection 4 → netDev 1 (eth2)
   [ChannelManager] Connection 5 → netDev 1 (eth2)
   [ChannelManager] Connection 6 → netDev 1 (eth2)
   [ChannelManager] Connection 7 → netDev 1 (eth2)
   [ChannelManager] Created 8 connections for GPU 0 (NICs: 0, 1)
   ```

2. **Bootstrap 握手**：
   ```
   [Bootstrap] Sent 8 handles to client
   [Bootstrap] Received 8 handles from server
   ```

3. **连接建立**：
   ```
   [ChannelManager] All 8 channels listening successfully
   [ChannelManager] All 8 channels accepted successfully
   [ChannelManager] All 8 channels connected successfully
   ```

4. **Chunk 分配**（应该看到 channel_id 从 0-7 循环）：
   ```
   [DEBUG][SERVER] chunk=0 channel=0 ...
   [DEBUG][SERVER] chunk=1 channel=1 ...
   [DEBUG][SERVER] chunk=2 channel=2 ...
   ...
   [DEBUG][SERVER] chunk=7 channel=7 ...
   [DEBUG][SERVER] chunk=8 channel=0 ...  # 循环回到 0
   ```

5. **带宽结果**：
   ```
   [PERF] Avg: XXX ms, BW: YY.YY GB/s
   ```
   - 目标：接近 21.26 GB/s（单 NIC 理论上限）

#### 4.4 使用 TCPX TRACE 验证

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
```

查找：
- 每个 connection 的 socket 建立日志
- 应该看到 8 个独立的 connect/accept 过程

---

## 🚨 潜在问题和解决方案

### 问题 1：NIC 索引不匹配

**症状**：日志显示 `netDev 0` 不是 `eth1`

**原因**：TCPX 的 device 索引可能与网卡名称不对应

**解决**：
```cpp
// 在 get_nics_for_gpu 中，通过 tcpx_get_properties 查找 NIC 名称
int find_nic_by_name(const char* name) {
  int ndev = tcpx_get_device_count();
  for (int i = 0; i < ndev; ++i) {
    tcpx_net_properties props{};
    if (tcpx_get_properties(i, &props) == 0) {
      if (props.name && strcmp(props.name, name) == 0) {
        return i;
      }
    }
  }
  return -1;
}
```

### 问题 2：端口冲突

**症状**：`bind() failed: Address already in use`

**原因**：8 个 connections 需要 8 个不同的端口

**解决**：
- 确保 `NCCL_GPUDIRECTTCPX_PORT_BEGIN` 和 `PORT_END` 范围足够大
- 或者使用动态端口分配（port=0）

### 问题 3：Bootstrap 超时

**症状**：Client 无法接收到 8 个 handles

**原因**：Server 端 listen 失败或 bootstrap 协议不匹配

**解决**：
- 检查 `server_listen_all` 的返回值
- 确保 `bootstrap_server_send_handles` 发送了正确数量的 handles

---

## 📊 成功标准

### 必须满足
1. ✅ 日志显示创建了 8 个 connections
2. ✅ 每个 connection 绑定到正确的 NIC（前 4 个到 nic1，后 4 个到 nic2）
3. ✅ Bootstrap 成功交换 8 个 handles
4. ✅ 所有 8 个 connections 成功建立（listen/accept/connect）
5. ✅ Chunks 正确地 round-robin 分配到 8 个 connections
6. ✅ 测试稳定完成，无 deadlock

### 期望达到
1. ✅ 带宽接近 21.26 GB/s（单 NIC 上限）
2. ✅ 比单 connection 基线提升 8-10 倍
3. ✅ 多个 GPUs 同时运行时，每个 NIC 的总带宽不超过 21.26 GB/s

---

## 🔄 下一步

完成上述修改后：
1. 测试单 GPU pair（GPU0 ↔ GPU0）
2. 测试同 NUMA 的 GPU pair（GPU0 ↔ GPU1）
3. 测试跨 NUMA 的 GPU pair（GPU0 ↔ GPU4）
4. 测试 full-mesh（所有 8×8 pairs）

最终目标：
- 每个 NIC 达到 ~21.26 GB/s
- 4 个 NICs 总带宽 ~85 GB/s
- 稳定运行，无 deadlock

