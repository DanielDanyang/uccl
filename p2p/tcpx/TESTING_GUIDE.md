# TCPX Transfer 测试指南

## 🚀 快速开始

### 1. 编译

```bash
cd /mnt/user_storage/uccl/p2p/tcpx
make clean
make test_tcpx_transfer
```

### 2. 运行测试

#### D2D 模式（推荐）

**Server 端**：
```bash
export UCCL_TCPX_UNPACK_IMPL=d2d
./tests/test_tcpx_transfer server
```

**Client 端**（另一个终端/节点）：
```bash
./tests/test_tcpx_transfer client <server_ip>
```

#### Host 模式（备用）

**Server 端**：
```bash
export UCCL_TCPX_UNPACK_IMPL=host
./tests/test_tcpx_transfer server
```

**Client 端**：
```bash
./tests/test_tcpx_transfer client <server_ip>
```

---

## 📋 环境变量说明

| 变量 | 值 | 说明 |
|------|-----|------|
| `UCCL_TCPX_UNPACK_IMPL` | `d2d` | 使用 Device-to-Device 拷贝（默认，推荐）|
| | `host` | 使用 Host 中转（DtoH + memcpy + HtoD）|
| | `kernel` | 使用 GPU 内核（暂不可用，需要 staging buffer）|
| `UCCL_TCPX_DEBUG` | `1` | 启用详细调试日志 |

---

## ✅ 预期输出

### 成功的测试输出示例

**Server 端**：
```
[DEBUG] === TCPX GPU-to-GPU transfer test ===
[TCPX] Loading plugin: /usr/local/tcpx/lib64/libnccl-net-tcpx.so
[DEBUG] Using payload_bytes=23
[DEBUG] Running in SERVER mode
[DEBUG] Listening on device 0
[DEBUG] Bootstrap connection established, sending handle
[DEBUG] Connection accepted; recv_comm=0x...
[DEBUG] Registered server receive buffer ptr=0x... bytes=4096
[DEBUG] Waiting for client data, expected bytes=23
[DEBUG] Request metadata: request_ptr=0x... active=1 idx=0 cnt_cache=1 ...
[DEBUG] Device handle: meta=0x... bounce_buf=0x... head=...
[DEBUG] descriptor[0] src_off=... len=23 dst_off=0
[DEBUG] Bounce probe (23B) from src_off=...: 48 65 6c 6c 6f 20 66 72 6f 6d 20 54 43 50 58 20 63 6c 69 65 6e 74 21
[DEBUG] Launching device unpack (D2D copies), total_bytes=23
[DEBUG] Device unpack completed successfully
[DEBUG] Received data (23 bytes): Hello from TCPX client!
[DEBUG] ✓ Data validation PASSED
[DEBUG] Server test completed successfully
```

**Client 端**：
```
[DEBUG] === TCPX GPU-to-GPU transfer test ===
[TCPX] Loading plugin: /usr/local/tcpx/lib64/libnccl-net-tcpx.so
[DEBUG] Using payload_bytes=23
[DEBUG] Running in CLIENT mode, server=<server_ip>
[DEBUG] Connecting to device 0
[DEBUG] Connection established; send_comm=0x...
[DEBUG] Registered client send buffer ptr=0x... bytes=4096
[DEBUG] Wrote test message to GPU buffer: Hello from TCPX client!
[DEBUG] Sending 23 bytes with tag 42
[DEBUG] Send completed successfully
[DEBUG] Client test completed successfully
```

---

## 🐛 常见问题

### 1. 编译错误：找不到 `rx_cmsg_parser.h`

**原因**：旧的构建缓存

**解决**：
```bash
make clean
make test_tcpx_transfer
```

### 2. 运行时错误：`loadMeta` 未定义

**原因**：头文件包含顺序问题

**解决**：确保 `tcpx_structs.h` 在 `rx_descriptor.h` 之前包含

### 3. 数据验证失败

**可能原因**：
- TCPX 插件版本不匹配
- GPU 设备不支持 devmem-tcp
- 网络配置问题

**调试**：
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_tcpx_transfer server
```

---

## 📊 性能对比

| 模式 | 延迟 | 带宽 | 适用场景 |
|------|------|------|----------|
| **D2D** | 低 | 高 | 生产环境（推荐）|
| **Host** | 高 | 低 | 调试/验证 |
| **Kernel** | 最低 | 最高 | 未来优化（需要解决 devmem-tcp 访问问题）|

---

## 🔍 调试技巧

### 查看 bounce buffer 内容

在 `test_tcpx_transfer.cc` 中已包含 bounce buffer probe：
```cpp
[DEBUG] Bounce probe (23B) from src_off=...: 48 65 6c 6c 6f ...
```

这会以十六进制显示 bounce buffer 的前 23 字节。

### 验证 descriptor 构建

查看日志中的 descriptor 信息：
```
[DEBUG] descriptor[0] src_off=20480 len=23 dst_off=0
```

确保：
- `src_off` 在合理范围内（通常 < 4MB）
- `len` 等于预期的 payload 大小
- `dst_off` 从 0 开始

### 检查 TCPX 插件加载

```
[TCPX] Loading plugin: /usr/local/tcpx/lib64/libnccl-net-tcpx.so
[TCPX] net->init rc=0
[TCPX] net->devices rc=0 ndev=4
```

如果看到错误，检查：
- TCPX 插件是否正确安装
- 路径是否正确
- 权限是否足够

---

## 📝 代码结构说明

### 核心文件

```
p2p/tcpx/
├── tcpx_interface.h          # TCPX API 接口
├── tcpx_impl.cc              # TCPX 插件封装
├── include/tcpx_structs.h    # 结构体定义（loadMeta 等）
├── rx/rx_descriptor.h        # Descriptor 构建（header-only）
└── tests/test_tcpx_transfer.cc  # 端到端测试
```

### 关键数据流

```
Client                          Server
  |                               |
  | 1. tcpx_connect_v5()         | 1. tcpx_listen()
  |----------------------------->| 2. tcpx_accept_v5()
  |                               |
  | 2. tcpx_reg_mr()             | 3. tcpx_reg_mr()
  | 3. Write data to GPU buffer  |
  | 4. tcpx_isend()              |
  |----------------------------->| 4. tcpx_irecv()
  |                               | 5. tcpx_test() (poll)
  |                               | 6. Parse metadata
  |                               | 7. buildDescriptorBlock()
  |                               | 8. D2D/Host unpack
  |                               | 9. Validate data
  |                               |
  | 5. tcpx_test() (completion)  |
  |                               |
```

---

## 🎯 下一步

1. ✅ 验证 D2D 和 Host 模式都能正常工作
2. ✅ 确认数据验证通过
3. ⏳ 准备 PR（不包含 device/ 目录）
4. ⏳ 后续优化：解决 kernel 模式的 devmem-tcp 访问问题

---

## 📞 支持

如果遇到问题，请检查：
1. TCPX 插件是否正确安装
2. GPU 驱动版本是否支持 devmem-tcp
3. 网络配置是否正确（eth1-eth4）
4. 环境变量是否正确设置

详细日志请使用 `UCCL_TCPX_DEBUG=1`。

