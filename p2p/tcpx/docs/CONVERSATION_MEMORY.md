# TCPX NIXL 后端开发记录

## 🎯 项目目标

### 核心任务
为 `benchmark_nixl.py` 创建 TCPX 后端，实现类似 mooncake 插件的胶水层。

**参考架构**:
- `mooncake/` - NIXL 后端插件参考实现
- `p2p/uccl_engine.h` - 引擎接口参考

**关键原则**:
- ✅ **正确方向**: 创建 NIXL 胶水层，直接调用 TCPX API
- ✅ **参考 mooncake**: 使用相同的插件架构模式
- ✅ **渐进开发**: 一小步一小步测试

## 🔧 技术要点

### 网络环境
- **服务器**: 10.0.0.238 (gcp5-h100-1-64)
- **客户端**: 10.0.0.107 (gcp5-h100-2-65)
- **环境**: Google Cloud H100 集群，无 RDMA 设备，只能使用 TCPX

### TCPX API 状态
- ✅ **插件加载**: 成功加载 `/usr/local/tcpx/lib64/libnccl-net-tcpx.so`
- ✅ **设备发现**: 发现 4 个 TCPX 设备 (eth1-eth4)
- ✅ **API 一致性**: 统一使用 v5 API
- ✅ **内存注册**: 支持 `NCCL_PTR_HOST` 主机内存类型

## 🏗️ 开发架构

### 目标架构 (参考 mooncake)
```
benchmark_nixl.py → NIXL API → TCPX Backend Plugin → TCPX API → GPU Memory
```

### 开发步骤
1. **第一步**: 清理已有代码，保留经过测试的 TCPX API 层
2. **第二步**: 测试 TCPX send/recv 性能，验证 API 可用性
3. **第三步**: 创建 NIXL 后端插件 (参考 mooncake 实现)
4. **第四步**: 集成到 benchmark_nixl.py

## 📊 当前状态

### ✅ 已完成 (经过测试)
- **TCPX API 层**: `tcpx_impl.cc` 和 `tcpx_interface.h`
- **基础测试**: `test_connection.cc` 和 `test_device_discovery.cc`
- **构建系统**: `Makefile`

### 🔄 当前任务
1. **清理代码** - 删除过时文档，保留核心功能
2. **性能测试** - 验证 TCPX send/recv 性能
3. **创建 NIXL 插件** - 参考 mooncake 实现

## 🔧 核心文件

### 保留的核心代码
```
p2p/tcpx/
├── tcpx_interface.h          # TCPX API 定义
├── tcpx_impl.cc              # TCPX 插件集成实现 (经过测试)
├── tests/
│   ├── test_device_discovery.cc  # 设备发现测试
│   └── test_connection.cc         # 连接测试
├── Makefile                  # 构建系统
└── CONVERSATION_MEMORY.md    # 项目记录
```

### 下一步文件
- `tcpx_backend.h/cpp` - NIXL 后端插件 (参考 mooncake)
- `tcpx_plugin.cpp` - 插件入口点

## 🚨 运行环境

### 编译和测试
```bash
# 工作目录
cd /mnt/user_storage/uccl/p2p/tcpx

# 编译
make clean && make test_connection

# 运行测试
# 服务器端 (10.0.0.238):
export UCCL_TCPX_DEBUG=1
./tests/test_connection server

# 客户端 (10.0.0.107):
export UCCL_TCPX_DEBUG=1
./tests/test_connection client 10.0.0.238
```

## 🎯 最终目标

让 `benchmark_nixl.py` 能够选择 "tcpx" 作为后端：
```python
python benchmark_nixl.py --backend tcpx --role server
python benchmark_nixl.py --backend tcpx --role client --remote-ip 10.0.0.238
```
