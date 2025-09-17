# TCPX Engine 完整架构文档

## 📋 文件对应关系总表

### ✅ 已完成的核心文件
| RDMA 原版 | TCPX 版本 | 状态 | 主要差异 |
|-----------|-----------|------|----------|
| `p2p/engine.h` | `p2p/tcpx/engine.h` | ✅ 完成 | `uccl::RDMAEndpoint` → `tcpx::TcpxEndpoint` |
| `p2p/engine.cc` | `p2p/tcpx/engine.cc` | ✅ 完成 | RDMA 调用 → TCPX 调用 |
| `p2p/pybind_engine.cc` | `p2p/tcpx/pybind_engine.cc` | ✅ 完成 | Python 绑定，接口相同 |
| `rdma/transport.h` | `p2p/tcpx/tcpx_interface.h` | ✅ 完成 | RDMA verbs → TCPX 插件接口 |
| `rdma/transport.cc` | `p2p/tcpx/tcpx_transport.cc` | ✅ 完成 | TCPX 传输层实现 |
| `p2p/uccl_engine.h` | `p2p/tcpx/uccl_engine_tcpx.h` | ✅ 完成 | C API 头文件 |
| `p2p/uccl_engine.cc` | `p2p/tcpx/uccl_engine_tcpx.cc` | ✅ 已存在 | C API 实现 |

### 🧹 需要清理的旧文件
| 旧文件 | 状态 | 说明 |
|--------|------|------|
| `p2p/tcpx/tcpx_endpoint.h/cc` | 🗑️ 可删除 | 被 tcpx_transport.cc 替代 |
| `p2p/tcpx/tcpx_engine.cc` | 🗑️ 可删除 | 被 uccl_engine_tcpx.cc 替代 |
| `p2p/tcpx/tcpx_pybind.cc` | 🗑️ 可删除 | 被 pybind_engine.cc 替代 |

### 📁 测试和构建文件
| RDMA 原版 | TCPX 版本 | 状态 | 说明 |
|-----------|-----------|------|------|
| `p2p/Makefile` | `p2p/tcpx/Makefile` | ✅ 已修改 | 编译新的文件组合 |
| `p2p/tests/test_*.py` | `p2p/tcpx/test_tcpx_*.py` | 🚧 部分存在 | 需要更多测试 |

## 🏗️ 架构层次对比

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 应用层                            │
│  collective.py, transfer.py, benchmarks/ (完全相同)         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Python 绑定层                              │
│  RDMA: pybind_engine.cc ←→ TCPX: pybind_engine.cc          │
│  (接口完全相同，可以无缝切换)                                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   C++ 引擎层                                │
│  RDMA: engine.h/cc ←→ TCPX: engine.h/cc                   │
│  (Endpoint 类接口相同，内部调用不同传输层)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   传输抽象层                                 │
│  RDMA: rdma/transport.cc ←→ TCPX: tcpx_transport.cc       │
│  (RDMAEndpoint vs TcpxEndpoint)                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   底层传输层                                 │
│  RDMA: InfiniBand verbs ←→ TCPX: NCCL TCPX Plugin         │
│  (硬件 RDMA vs 软件优化的 TCP/IP)                           │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 关键实现差异

### 1. 传输层初始化
**RDMA 版本:**
```cpp
// p2p/engine.cc:70
ep_ = new uccl::RDMAEndpoint(num_cpus_);
```

**TCPX 版本:**
```cpp
// p2p/tcpx/engine.cc:70
ep_ = new tcpx::TcpxEndpoint(num_cpus_);
```

### 2. 设备工厂
**RDMA 版本:**
```cpp
// p2p/engine.cc:83
numa_node_ = uccl::RDMAFactory::get_factory_dev(gpu_to_dev[local_gpu_idx_])->numa_node;
```

**TCPX 版本:**
```cpp
// p2p/tcpx/engine.cc:83
numa_node_ = tcpx::TcpxFactory::get_factory_dev(gpu_to_dev[local_gpu_idx_])->numa_node;
```

### 3. 连接建立
**RDMA 版本:**
```cpp
// p2p/engine.cc:163
return ep_->uccl_connect(gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
                        gpu_to_dev[remote_gpu_idx], remote_gpu_idx,
                        ip_addr, remote_port);
```

**TCPX 版本:**
```cpp
// p2p/tcpx/engine.cc:163
return ep_->tcpx_connect(gpu_to_dev[local_gpu_idx_], local_gpu_idx_,
                        gpu_to_dev[remote_gpu_idx], remote_gpu_idx,
                        ip_addr, remote_port);
```

## 🎯 实现状态

### Phase 1: 架构完成 ✅
- [x] 文件结构对应
- [x] 接口定义完成
- [x] 基本类实现
- [x] 编译配置更新

### Phase 2: 功能实现 🚧
- [x] TCPX 插件加载框架
- [x] 设备发现和管理
- [x] 简化的连接管理
- [ ] 真实的 TCPX 插件调用
- [ ] 内存注册实现
- [ ] 数据传输实现

### Phase 3: 测试和优化 ❌
- [ ] 完整的测试套件
- [ ] 性能优化
- [ ] 错误处理完善
- [ ] 文档完善

## 🚀 下一步行动

### 立即可做 (编译测试)
```bash
cd p2p/tcpx
make clean
make
python test_tcpx_write.py
```

### 短期目标 (1-2天)
1. **修复编译问题** - 解决头文件和链接问题
2. **基本功能测试** - 确保引擎创建和元数据生成工作
3. **真实插件调用** - 逐步添加真实的 TCPX 插件功能

### 中期目标 (1周)
1. **完整连接建立** - 实现 listen/connect/accept
2. **内存注册** - 实现 regMr/deregMr
3. **数据传输** - 实现 isend/irecv/test

### 长期目标 (2-4周)
1. **性能优化** - 调优传输性能
2. **完整测试** - 端到端功能测试
3. **生产就绪** - 错误处理和稳定性

## 💡 架构优势

### 1. 接口兼容性
- Python 应用无需修改
- C++ 引擎接口相同
- 可以通过环境变量切换传输层

### 2. 模块化设计
- 传输层完全解耦
- 易于添加新的传输机制
- 便于测试和调试

### 3. 渐进式实现
- 可以先用简化实现验证架构
- 逐步添加真实功能
- 降低开发风险

这个架构确保了 TCPX 版本与 RDMA 版本的完全兼容性，同时提供了清晰的实现路径。
