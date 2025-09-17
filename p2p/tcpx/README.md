# TCPX Engine for UCCL

基于 Google NCCL GPUDirect TCPX 插件的 UCCL 引擎实现，用于云端 GPU 间的高性能通信。

## 架构

采用与原版 RDMA 引擎相同的架构：
- `TcpxEndpoint` 类：封装 TCPX 插件的复杂性
- `uccl_engine_tcpx.cc`：提供 C API 包装
- 兼容现有的 Python 绑定和测试

## 文件说明

- `tcpx_endpoint.h/cc` - TCPX 端点类，封装 NCCL TCPX 插件
- `uccl_engine_tcpx.cc` - UCCL 引擎 C API 实现
- `test_tcpx_write.py` - 基本功能测试
- `Makefile` - 编译配置

## 编译和测试

```bash
# 编译 TCPX 引擎库
make

# 运行基本测试
make test
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
