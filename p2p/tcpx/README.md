# TCPX Engine for UCCL

基于 Google NCCL GPUDirect TCPX 插件的 UCCL 引擎实现，用于云端 GPU 间的高性能通信。

## 📋 完整文件架构对应

### 核心引擎文件
| RDMA 原版 | TCPX 版本 | 状态 | 主要差异 |
|-----------|-----------|------|----------|
| `p2p/engine.h` | `p2p/tcpx/engine.h` | ✅ 已修改 | `uccl::RDMAEndpoint` → `tcpx::TcpxEndpoint` |
| `p2p/engine.cc` | `p2p/tcpx/engine.cc` | ❌ 缺失 | 需要复制并修改 RDMA 调用 |
| `p2p/pybind_engine.cc` | `p2p/tcpx/pybind_engine.cc` | ❌ 缺失 | 需要复制 Python 绑定 |

### 传输层文件
| RDMA 原版 | TCPX 版本 | 状态 | 主要差异 |
|-----------|-----------|------|----------|
| `rdma/transport.h` | `p2p/tcpx/tcpx_interface.h` | ✅ 已创建 | RDMA verbs → TCPX 插件接口 |
| `rdma/transport.cc` | `p2p/tcpx/tcpx_transport.cc` | ❌ 缺失 | 需要实现 TCPX 传输层 |

### 现有文件说明
- `engine.h/cc` - 主引擎类，与 RDMA 版本接口相同
- `tcpx_interface.h` - 简化的 TCPX 插件接口定义
- `tcpx_transport.cc` - TCPX 传输层实现 (TcpxEndpoint 类)
- `pybind_engine.cc` - Python 绑定，与 RDMA 版本相同
- `uccl_engine_tcpx.h/cc` - UCCL 引擎 C API 实现
- `test_tcpx_write.py` - 基本功能测试
- `Makefile` - 编译配置

### 旧文件 (可能需要清理)
- `tcpx_endpoint.h/cc` - 旧版实现，已被 tcpx_transport.cc 替代
- `tcpx_engine.cc` - 旧版 C API，已被 uccl_engine_tcpx.cc 替代
- `tcpx_pybind.cc` - 旧版绑定，已被 pybind_engine.cc 替代

## 🏗️ 完整架构对比

### 层次结构
```
Python 应用层: collective.py, transfer.py (相同)
     ↓
Python 绑定层: pybind_engine.cc (相同接口)
     ↓
C++ 引擎层: engine.h/cc (相同接口)
     ↓
传输抽象层: RDMA: rdma/transport.cc ←→ TCPX: tcpx_transport.cc
     ↓
底层传输: InfiniBand verbs ←→ NCCL TCPX Plugin
```

### 关键差异

| 组件 | RDMA 版本 | TCPX 版本 | 差异说明 |
|------|-----------|-----------|----------|
| **传输类** | `uccl::RDMAEndpoint` | `tcpx::TcpxEndpoint` | 接口相同，底层不同 |
| **连接建立** | `uccl_connect()` | `tcpx_connect()` | RDMA verbs vs TCPX plugin |
| **内存注册** | `ibv_reg_mr()` | `tcpxRegMr()` | 支持 GPU 内存 |
| **数据传输** | `ibv_post_send()` | `tcpxIsend()` | 异步操作 |
| **设备管理** | `uccl::RDMAFactory` | `tcpx::TcpxFactory` | 设备发现和管理 |

## 编译和测试

```bash
# 编译 TCPX 引擎库
make clean && make

# 运行基本测试
python test_tcpx_write.py
```

## 环境变量

```bash
export UCCL_TCPX_PLUGIN_PATH=/path/to/libnccl-net-tcpx.so
export UCCL_TCPX_DEV=0
```

## 使用方法

1. 设置环境变量
2. 编译引擎库
3. 运行测试验证功能
4. 集成到现有的 UCCL 系统中

## 特性

- ✅ 基本引擎创建和销毁
- ✅ 元数据生成和解析
- ✅ 连接建立（connect/accept）
- ✅ 内存注册（reg/dereg）
- 🚧 数据传输（write/recv）- 简化实现
- 🚧 异步操作支持

## 注意事项

- 当前版本跳过了 NCCL 插件的初始化以避免段错误
- 数据传输功能使用简化的占位符实现
- 需要在有 TCPX 支持的云端环境中运行
