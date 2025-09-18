# TCPX P2P 项目目标与进度总结

## 🎯 项目目标

### 核心需求
用户有两台 H100 16卡机器，但**没有 RDMA 设备**，只能使用 TCPX。目标是让现有的 P2P KV 缓存系统能够工作，通过 TCPX 替代 RDMA 进行 GPU 间通信。

### 关键约束
- **环境**: Google Cloud H100 集群
- **网络**: 10.0.1.25 和 10.0.0.226，每台 8 张 H100
- **传输**: 只有 TCPX 可用 (`/usr/local/tcpx/lib64/libnccl-net-tcpx.so`)
- **兼容性**: Python 应用无需修改

## 📋 用户的重要提醒

### 1. 实现策略
- ✅ **"时刻记住要抄p2p原有实现"** - 参考现有 RDMA 代码模式
- ✅ **"要用tcpx专属接口api"** - 使用真实的 TCPX 插件 API
- ✅ **"请在已有代码基础上修改，不要创建新文件"** - 修改现有文件而非重写
- ✅ **"如果代码没有用请直接清除"** - 删除无用的模拟代码

### 2. 技术要求
- ❌ **"像这种创建虚拟设备显然不对，我是有真实设备的啊"** - 需要真实设备发现
- ❌ **"你的transport很有必要照着之前已有的代码进行修改"** - 参考 RDMA 实现
- ✅ **"我觉得写个东西替代它不现实"** - 不要重写整个传输层

### 3. 测试验证
- ✅ **"不过你的测试能确保两个nodes直接的connection正常吗？"** - 需要真实跨节点测试
- ✅ **"毕竟我tcpx的目的是代替rdma"** - 验证 TCPX 作为 RDMA 替代品

## 📊 当前进度

### ✅ 已完成
1. **插件接口发现** - 成功加载 NCCL TCPX 插件，验证所有 11 个函数指针可访问
2. **基础架构搭建** - 创建完整的测试框架和文档结构
3. **编译系统** - 成功编译 `libuccl_tcpx_engine.so` Python 模块
4. **基本功能测试** - 引擎创建、销毁、元数据生成等基础功能正常
5. **连接测试** - 模拟连接功能测试通过

### 🚧 当前问题
1. **虚拟设备问题** - 当前实现创建虚拟设备而非发现真实 TCPX 设备
2. **模拟连接问题** - 连接函数返回 `127.0.0.1` 而非真实客户端 IP
3. **缺少真实 API 调用** - 没有调用真正的 TCPX 插件函数

### 🔄 正在进行
- **真实设备发现** - 修改 `TcpxFactory::initialize()` 使用真实 TCPX 插件
- **真实网络连接** - 实现真正的 `tcpxConnect_v5()` 和 `tcpxAccept_v5()` 调用

## 🎯 修正后的实现策略

### 核心洞察
用户指出了关键问题：**不需要重写整个传输层**！

现有的 `p2p/engine.cc` 只是调用 `ep_` (RDMAEndpoint) 的方法：
- `ep_->uccl_connect()`
- `ep_->uccl_regmr()`
- `ep_->uccl_send_async()`
- 等等...

### 正确方案
1. **保持现有架构** - 不修改 `p2p/engine.cc` 的业务逻辑
2. **替换底层实现** - 只需要让 TCPX 版本的 endpoint 提供相同接口
3. **真实插件调用** - 修改现有的 `tcpx_transport_minimal.cc` 调用真实 TCPX API

## 🧹 代码整理完成

### 删除的冗余文件
- ❌ `nccl_plugin_interface.h` - 与 `tcpx_interface.h` 重复
- ❌ `engine.h/cc` - 过于复杂的引擎实现
- ❌ `pybind_engine.cc` - Python 绑定（暂时不需要）
- ❌ `uccl_engine_tcpx.h/cc` - C API 包装（暂时不需要）
- ❌ 多个测试文件 - 过于复杂的测试
- ❌ 文档目录 - 重复的文档

### 保留的核心文件
- ✅ `tcpx_interface.h` - 最简化的接口定义
- ✅ `tcpx_impl.cc` - 基础 TCPX 插件加载实现
- ✅ `test_tcpx.cc` - 简单的测试程序
- ✅ `Makefile.simple` - 简化的编译配置
- ✅ `PROJECT_GOALS_AND_PROGRESS.md` - 项目文档

## 🔧 当前实现

### 最简化的方法
根据你的提醒，我们现在采用最简单的方法：

1. **只保留核心功能** - 插件加载和设备发现
2. **直接调用 TCPX 插件** - 不创建虚拟设备
3. **简单的 C 接口** - 避免复杂的 C++ 类层次

### 验证步骤
```bash
# 编译测试程序
make -f Makefile.simple test_tcpx

# 运行测试
./test_tcpx
```

### 预期结果
- ✅ 加载真实的 TCPX 插件
- ✅ 发现真实的网络设备数量
- ✅ 不再显示虚拟设备

## 📝 技术细节

### 参考代码模式
- **设备发现**: 参考 `p2p/engine.cc` 中的 `gpu_to_dev` 映射
- **插件加载**: 参考 `nccl-plugin-gpudirecttcpx/src/net_tcpx.cc`
- **接口调用**: 参考 `p2p/tcpx/tcpx_simple.cc` 中的插件函数调用

### 关键 API 函数
```c
tcpxResult_t tcpxInit(tcpxDebugLogger_t logFunction);
tcpxResult_t tcpxDevices(int* ndev);
tcpxResult_t tcpxGetProperties(int dev, tcpxNetProperties_t* props);
tcpxResult_t tcpxListen(int dev, void* oHandle, void** listenComm);
tcpxResult_t tcpxConnect_v5(int dev, void* oHandle, void** sendComm, devNetDeviceHandle** sendDevHandle);
tcpxResult_t tcpxAccept_v5(void* listenComm, void** recvComm, devNetDeviceHandle** recvDevHandle);
```

## 🎉 成功标准

项目成功的标志是：
1. **真实设备发现** - 不再显示虚拟设备，而是真实的 TCPX 网络接口
2. **真实网络连接** - 连接显示真实的客户端 IP (10.0.0.226)，而非 127.0.0.1
3. **跨节点通信** - 两台 H100 机器之间成功建立 TCPX 连接
4. **性能验证** - TCPX 能够作为 RDMA 的有效替代品

---

**文档创建时间**: 2025-09-18  
**当前状态**: 基础架构完成，正在实现真实 TCPX 插件集成  
**下一步**: 修复设备发现和网络连接的真实实现
