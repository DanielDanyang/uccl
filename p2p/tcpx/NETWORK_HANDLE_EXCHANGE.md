# TCPX 网络化句柄交换实现

## 🎯 问题背景

之前的TCPX连接测试依赖共享文件系统进行句柄交换：
- 服务器将连接句柄保存到 `/tmp/tcpx_handle.dat`
- 客户端从同一文件读取句柄
- **问题**: 两个节点(10.0.0.107和10.0.1.25)没有共享文件系统

## ✅ 解决方案：网络化句柄交换

参考RDMA版本的实现，我们实现了基于TCP socket的句柄交换机制。

### 核心改进

#### 1. TCPX句柄结构
```cpp
struct tcpxHandle {
  uint32_t ip_addr_u32;    // 服务器IP地址
  uint16_t listen_port;    // TCPX监听端口
  int remote_dev;          // 设备ID
  int remote_gpuidx;       // GPU索引
};
```

#### 2. 网络化交换流程

**服务器端 (10.0.0.107)**:
1. 调用 `tcpx_listen()` 创建TCPX监听socket
2. 创建bootstrap TCP服务器 (端口12345)
3. 等待客户端连接到bootstrap服务器
4. 通过bootstrap连接发送TCPX句柄给客户端
5. 调用 `tcpx_accept_v5()` 接受TCPX连接

**客户端 (10.0.1.25)**:
1. 连接到服务器的bootstrap socket (10.0.0.107:12345)
2. 通过bootstrap连接接收TCPX句柄
3. 解析句柄获取连接信息
4. 调用 `tcpx_connect_v5()` 连接到服务器

### 关键函数实现

#### Bootstrap服务器创建
```cpp
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  // 绑定到端口12345，等待客户端连接
  // 返回已连接的客户端socket
}
```

#### Bootstrap客户端连接
```cpp
int connect_to_bootstrap_server(const char* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  // 连接到服务器的12345端口，带重试逻辑
  // 返回连接的socket
}
```

#### IP地址转换工具
```cpp
uint32_t str_to_ip(const char* ip_str);  // "10.0.0.107" -> uint32_t
std::string ip_to_str(uint32_t ip_u32);  // uint32_t -> "10.0.0.107"
```

## 🔄 完整交换流程

### 阶段1: Bootstrap连接建立
```
服务器                           客户端
  |                               |
  | 1. 创建bootstrap服务器         |
  |    (监听端口12345)            |
  |                               |
  |                               | 2. 连接bootstrap服务器
  |                               |    (10.0.0.107:12345)
  | 3. 接受bootstrap连接          |
  |<------------------------------|
```

### 阶段2: 句柄交换
```
服务器                           客户端
  |                               |
  | 4. 创建TCPX句柄               |
  |    (IP, 端口, 设备信息)        |
  |                               |
  | 5. 发送句柄                   |
  |------------------------------>| 6. 接收句柄
  |                               |    解析连接信息
  |                               |
  | 7. 关闭bootstrap连接          | 8. 关闭bootstrap连接
```

### 阶段3: TCPX连接建立
```
服务器                           客户端
  |                               |
  | 9. tcpx_accept_v5()          |
  |    等待TCPX连接               |
  |                               |
  |                               | 10. tcpx_connect_v5()
  |                               |     使用句柄中的信息连接
  | 11. 连接建立成功              |
  |<------------------------------|
```

## 🚀 使用方法

### 编译测试
```bash
cd /mnt/user_storage/uccl/p2p/tcpx
make clean && make test_connection
```

### 运行测试

**节点1 (10.0.0.107) - 服务器**:
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection server
```

**节点2 (10.0.1.25) - 客户端**:
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection client 10.0.0.107
```

## 📊 预期输出

### 服务器端
```
=== TCPX Connection Test ===
[Step 1] Initializing TCPX...
✓ SUCCESS: Found 4 TCPX devices

[Step 2] Starting as SERVER...
✓ SUCCESS: Listening on device 0

[Step 3] Creating TCPX handle for client...
Bootstrap server listening on port 12345
✓ SUCCESS: Handle sent to client

[Step 4] Testing data transfer (receive)...
✓ SUCCESS: Received 25 bytes
Data: 'Hello from TCPX client!'
```

### 客户端
```
=== TCPX Connection Test ===
[Step 1] Initializing TCPX...
✓ SUCCESS: Found 4 TCPX devices

[Step 2] Starting as CLIENT...
[Step 3] Connecting to server for handle exchange...
Connected to bootstrap server at 10.0.0.107
✓ SUCCESS: Handle received from server
Server info - IP: 127.0.0.1, Port: 43443, Dev: 0

[Step 4] Testing data transfer (send)...
✓ SUCCESS: Sent 25 bytes
```

## 🔧 技术优势

### 相比文件系统方式的改进
1. **无需共享存储** - 纯网络通信
2. **实时同步** - 无需手动文件复制
3. **错误处理** - 网络连接失败时自动重试
4. **标准化** - 遵循RDMA版本的设计模式

### 与RDMA版本的一致性
1. **相同的句柄结构** - IP、端口、设备信息
2. **相同的交换流程** - bootstrap连接 + 句柄传输
3. **相同的错误处理** - 重试逻辑和超时机制
4. **相同的接口设计** - 便于后续集成

## 🎯 下一步计划

1. **测试验证** - 在两个节点间验证网络化句柄交换
2. **数据传输** - 验证完整的发送/接收功能
3. **集成到Endpoint** - 将机制集成到生产代码
4. **性能优化** - 减少bootstrap连接开销

这个网络化句柄交换机制解决了共享文件系统的依赖问题，为TCPX P2P通信提供了可靠的连接建立基础。
