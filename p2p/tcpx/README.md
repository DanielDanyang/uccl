# TCPX NIXL Backend

为 `benchmark_nixl.py` 创建 TCPX 后端插件，参考 mooncake 实现。

## 📁 项目结构

```
p2p/tcpx/
├── tcpx_interface.h          # TCPX API 定义
├── tcpx_impl.cc              # TCPX 插件集成实现 (经过测试)
├── tests/
│   ├── test_device_discovery.cc  # 设备发现测试
│   ├── test_connection.cc         # 连接测试
│   ├── test_tcpx.cc              # 基础功能测试
│   └── test_perf.cc              # 性能测试框架
├── Makefile                  # 构建系统
├── CONVERSATION_MEMORY.md    # 项目记录
└── README.md                 # 本文件
```

## 🚀 快速开始

### 编译测试
```bash
# 编译所有测试
make all

# 或编译单个测试
make test_device_discovery
make test_connection
make test_tcpx
```

### 运行测试
```bash
# 基础功能测试
export UCCL_TCPX_DEBUG=1
./tests/test_tcpx

# 设备发现测试
./tests/test_device_discovery

# 性能测试框架 (单节点)
./tests/test_perf

# 连接测试 (需要两个节点)
# 服务器端:
./tests/test_connection server
# 客户端:
./tests/test_connection client <server_ip>
```

## 🎯 开发计划

### 当前状态 ✅
- TCPX API 层已完成并测试
- 基础连接功能已验证

### 下一步 🔄
1. **性能测试** - 验证 TCPX send/recv 性能
2. **NIXL 插件** - 创建类似 mooncake 的后端插件
3. **集成测试** - 让 benchmark_nixl.py 使用 tcpx 后端

## 📚 参考

- `mooncake/` - NIXL 后端插件参考实现
- `p2p/uccl_engine.h` - 引擎接口参考
- `nccl-plugin-gpudirecttcpx/src/net_tcpx.h` - TCPX API 定义
