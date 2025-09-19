# TCPX 连接测试 - 正确的IP配置

## 🎯 IP地址配置

根据您的提醒，正确的IP配置是：
- **服务器**: 10.0.0.238 (gcp5-h100-1-64)
- **客户端**: 10.0.0.107 (gcp5-h100-2-65)

## 🔧 修复的问题

### 1. 编译错误修复
- 添加了 `#include <cstring>` 到 `tcpx_handle_utils.h`
- 修复了 `memset` 未声明的问题

### 2. IP地址智能检测
更新了 `extract_ip_from_tcpx_handle()` 函数：
```cpp
// 智能选择主接口IP (10.0.0.x)
uint32_t ip2_host = ntohl(ip2);
if ((ip2_host & 0xFFFFFF00) == 0x0A000000) {  // 10.0.0.x
  return ip2;  // 使用主接口IP
} else {
  return ip1;  // 使用备用IP
}
```

## 🚀 测试步骤

### 编译测试
```bash
cd /mnt/user_storage/uccl/p2p/tcpx
make clean && make test_connection
```

### 运行服务器 (10.0.0.238)
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection server
```

### 运行客户端 (10.0.0.107)  
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection client 10.0.0.238
```

## 📊 预期结果

### 服务器端输出
```
=== TCPX Connection Test ===
[Step 1] Initializing TCPX...
✓ SUCCESS: Found 4 TCPX devices

[Step 2] Starting as SERVER...
✓ SUCCESS: Listening on device 0
Listen comm: 0x...

[Step 3] Extracting TCPX connection info...
TCPX handle data (first 64 bytes):
02 00 XX XX 0a 00 00 ee ...  # 0a 00 00 ee = 10.0.0.238
...
IP extraction attempts:
  bytes[4-7]: 10.128.0.XX
  bytes[52-55]: 10.0.0.238
  Using bytes[52-55] as main IP
Extracted: IP=10.0.0.238, Port=XXXXX, Dev=0

Creating bootstrap server for handle exchange...
Bootstrap server listening on port 12345
```

### 客户端输出
```
=== TCPX Connection Test ===
[Step 1] Initializing TCPX...
✓ SUCCESS: Found 4 TCPX devices

[Step 2] Starting as CLIENT...
Connecting to server at 10.0.0.238

[Step 3] Connecting to server for handle exchange...
Connected to bootstrap server at 10.0.0.238
✓ SUCCESS: Handle received from server
Server info - IP: 10.0.0.238, Port: XXXXX, Dev: 0

Attempting to connect to 10.0.0.238...
✓ SUCCESS: Connected to server!

[Step 4] Testing data transfer (send)...
✓ SUCCESS: Sent XX bytes
```

## 🔍 关键改进点

### 1. 正确的服务器IP
- 之前：硬编码 127.0.0.1 或 10.0.0.107
- 现在：从TCPX句柄智能提取 10.0.0.238

### 2. 动态端口提取
- 之前：硬编码 43443
- 现在：从TCPX句柄提取真实端口 (如 45599)

### 3. 智能IP选择
- 自动识别 10.0.0.x 范围的主接口IP
- 避免使用 eth1-eth4 的内部IP

## 🎯 成功标志

如果看到以下输出，说明连接成功：
1. ✅ Bootstrap连接建立
2. ✅ 句柄交换成功
3. ✅ TCPX连接建立 (不再有 "Connection refused")
4. ✅ 数据传输完成

## 🚨 故障排除

### 如果仍然连接失败
1. **检查防火墙**: 确保端口12345和TCPX端口开放
2. **检查网络**: `ping 10.0.0.238` 确认连通性
3. **检查句柄提取**: 查看调试输出中的IP地址是否正确
4. **检查TCPX端口**: 从服务器日志中确认监听端口

### 调试命令
```bash
# 检查网络连通性
ping 10.0.0.238

# 检查端口占用
netstat -an | grep 12345

# 查看TCPX接口
ip addr show | grep "10.0.0"
```

这个修复应该解决之前的"Connection refused"问题，因为现在我们使用正确的服务器IP地址 (10.0.0.238) 而不是错误的客户端IP (10.0.0.107)。
