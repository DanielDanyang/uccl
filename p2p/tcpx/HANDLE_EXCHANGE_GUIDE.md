# TCPX 句柄交换连接测试指南

## 🎯 目标
测试带有正确句柄交换机制的TCPX连接，解决之前的栈溢出问题。

## 🔧 编译
```bash
make -f Makefile.simple test_connection_v2
```

## 🚀 运行测试

### 方法1：使用共享文件系统（推荐）

如果两个节点共享同一个文件系统（如NFS），可以直接使用：

**步骤1：在节点1 (10.0.0.107) 启动服务器**
```bash
export UCCL_TCPX_DEBUG=1
./test_connection_v2 server
```

服务器会：
1. 初始化TCPX并开始监听
2. 将连接句柄保存到 `/tmp/tcpx_handle.dat`
3. 等待用户按Enter键继续

**步骤2：在节点2 (10.0.1.25) 启动客户端**
```bash
export UCCL_TCPX_DEBUG=1
./test_connection_v2 client 10.0.0.107
```

客户端会：
1. 从 `/tmp/tcpx_handle.dat` 读取连接句柄
2. 使用句柄连接到服务器

**步骤3：完成连接**
在服务器端按Enter键，服务器将接受客户端连接。

### 方法2：手动复制句柄文件

如果节点不共享文件系统：

**步骤1：启动服务器**
```bash
# 在节点1
export UCCL_TCPX_DEBUG=1
./test_connection_v2 server
```

**步骤2：复制句柄文件**
```bash
# 在节点1，将句柄文件复制到节点2
scp /tmp/tcpx_handle.dat user@10.0.1.25:/tmp/
```

**步骤3：启动客户端**
```bash
# 在节点2
export UCCL_TCPX_DEBUG=1
./test_connection_v2 client 10.0.0.107
```

**步骤4：完成连接**
在服务器端按Enter键。

## 🔍 预期结果

### 成功情况：
```
=== TCPX Connection Test V2 (with Handle Exchange) ===
[Step 1] Initializing TCPX...
✓ SUCCESS: Found 4 TCPX devices

[Step 2] Starting as SERVER...
✓ SUCCESS: Listening on device 0
✓ SUCCESS: Handle saved to /tmp/tcpx_handle.dat

[Step 3] Calling tcpx_accept_v5...
✓ SUCCESS: Connection accepted!
Recv comm: 0x...
Recv dev handle: 0x...
```

### 关键改进：
1. **正确的句柄交换**：服务器生成句柄，客户端使用相同句柄
2. **增大缓冲区**：句柄大小从64字节增加到128字节
3. **同步机制**：服务器等待客户端准备好再接受连接
4. **文件清理**：测试完成后自动删除句柄文件

## 🐛 故障排除

### 问题1：客户端找不到句柄文件
```
✗ FAILED: Cannot open handle file /tmp/tcpx_handle.dat
```
**解决**：确保服务器已经运行并生成了句柄文件

### 问题2：连接超时
**解决**：检查网络连通性和防火墙设置

### 问题3：栈溢出
**解决**：已增大句柄缓冲区，如果仍有问题，可进一步增大`NCCL_NET_HANDLE_MAXSIZE`

## 📝 下一步

成功后可以：
1. 测试数据传输功能
2. 集成到Endpoint类中
3. 实现内存注册和数据传输API
