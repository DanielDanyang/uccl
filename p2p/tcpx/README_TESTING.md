# TCPX Testing Guide

## 测试目标
逐步验证TCPX功能，从基础设备发现开始，最终实现两个节点间的连接。

## 测试阶段

### 阶段1: TCPX设备发现测试 ✅ (当前阶段)

**目标**: 验证TCPX插件能够正确加载并发现设备

**测试文件**:
- `test_tcpx.cc` - 基础插件加载测试
- `test_device_discovery.cc` - 详细的设备发现测试

**运行方法**:
```bash
# 编译测试
make -f Makefile.simple test_device_discovery

# 运行测试 (需要设置环境变量)
export UCCL_TCPX_DEBUG=1
export UCCL_TCPX_PLUGIN_PATH=/path/to/libnccl-net-tcpx.so
./test_device_discovery
```

**预期结果**:
- 成功加载TCPX插件
- 发现至少1个TCPX设备 (如果硬件支持)
- 或者优雅地报告没有设备但插件加载成功

### 阶段2: Endpoint初始化测试 (下一阶段)

**目标**: 验证Endpoint类能够使用TCPX进行初始化

**当前状态**: 
- ✅ 已添加TCPX设备发现到构造函数
- ⚠️ 需要注释掉更多RDMA依赖代码
- ⚠️ 需要解决编译依赖问题

**下一步**:
1. 修复tcpx_endpoint.cc中的编译错误
2. 注释掉所有RDMA相关的初始化代码
3. 创建简化的Endpoint测试

### 阶段3: 连接建立测试 (未来)

**目标**: 实现两个节点间的TCPX连接

**计划**:
1. 实现TCPX版本的connect()函数
2. 实现TCPX版本的accept()函数  
3. 创建双节点连接测试

### 阶段4: 数据传输测试 (未来)

**目标**: 验证连接建立后的数据传输

## 当前问题和解决方案

### 编译问题
- **问题**: tcpx_endpoint.cc依赖大量RDMA头文件
- **解决**: 逐步注释RDMA代码，添加TCPX替代实现

### 依赖问题  
- **问题**: 缺少GPU运行时、Python绑定等依赖
- **解决**: 创建独立的测试，不依赖完整的Endpoint类

### 环境问题
- **问题**: 云端环境无法编译执行
- **解决**: 提供详细的测试代码和预期行为说明

## 测试策略

1. **渐进式测试**: 从最简单的功能开始，逐步增加复杂性
2. **独立测试**: 每个测试尽可能独立，减少依赖
3. **详细日志**: 使用UCCL_TCPX_DEBUG=1获取详细调试信息
4. **错误处理**: 提供清晰的错误信息和故障排除指导

## 下一步行动

1. **立即**: 运行test_device_discovery测试，验证TCPX插件加载
2. **短期**: 修复tcpx_endpoint.cc编译问题，创建Endpoint初始化测试
3. **中期**: 实现connect/accept函数的TCPX版本
4. **长期**: 完整的双节点连接和数据传输测试
