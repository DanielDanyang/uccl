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

目前 p2p 这一套是围绕 RDMA 写的，一个“连接”请求会从 Python 层一路调用到 uccl::RDMAEndpoint。想让它在 TCPX-only 环境下跑通，不能只改几行 glue code，至少要把下列“类/文件”换成 TCPX 版本，或提供能被它们调用的等价实现：

1. p2p/engine.cc → uccl::RDMAEndpoint
构造/析构、connect/accept、reg/recv/send 全都直接调用 RDMA 资源（libibverbs、QP、CQ）
要支持 TCPX，必须写一个新的 C++ Endpoint（或同名类的 TCPX 变种），内部完成：
TCPX 插件的 listen/connect/accept 握手（包括 metadata OOB、socket 交互）
连接状态管理（发送/接收 Comm、socket、线程等）
内存注册 / 发送 / 接收的封装
然后让现有 Endpoint* endpoint 成员指向 TCPX 版，而不是 uccl::RDMAEndpoint
2. p2p/uccl_engine.cc
这里只是把 uccl_engine_* C API 转发给 Endpoint
一旦上面的 Endpoint 换成 TCPX 实现，这里的函数基本 “照抄” 也能用；否则它仍会继续调到 RDMA 逻辑
你可以先复制一份 TCPX 版本（如 uccl_engine_tcpx.cc），确保其中 connect/accept/reg/send/... 都调用 TCPX Endpoint
3. p2p/pybind_engine.cc
Python 的 p2p.Endpoint 绑定指向的是 RDMA Endpoint
要让 Python 调用 TCPX，需要：
要么修改这个 pybind 文件，让它根据编译开关/环境变量绑定 TCPX 版 Endpoint
要么像你正在做的那样，在 p2p/tcpx/pybind_tcpx.cc 暴露一个独立的 TCPX 模块，并让上层脚本 import 它

实际落地建议
先实现 TCPX Endpoint（类似你在 p2p/tcpx/ 目录尝试的 minimal 版本，但要支持多个连接、metadata 编码/解析、send/recv 管理等）
封装成 C API/pybind：可参考 uccl_engine_cc + pybind_engine.cc 的接口
再考虑和 RDMA 分支的切换：例如编译时选项、环境变量切换，或在 uccl 包里同时暴露 Endpoint（RDMA）和 TcpxEndpoint


推荐做法（并行 TCPX 栈）

新建/使用 p2p/tcpx/ 目录，不动 RDMA 原实现。
复制并改出三套对应文件（命名保持清晰，避免符号冲突）：
Endpoint 层
从 p2p/engine.cc 拆一套接口一致的 TCPX 版本，例如 p2p/tcpx/engine.cc/engine.h 中的 TcpxEndpoint（你已有雏形）。
公开的方法签名尽量与 RDMA Endpoint 相同：构造/析构、connect/accept/get_metadata/parse_metadata/reg/dereg/send/recv/send_async/recv_async/poll_async 等，便于上层胶水直接替换。
C API 层
从 p2p/uccl_engine.cc 复制为 p2p/tcpx/uccl_engine_tcpx.cc（和必要的 .h），所有 uccl_engine_* 直接转发到 TcpxEndpoint。
注意：不要与 RDMA 版 .so 同时导出同名符号（或在独立 .so 中导出，或通过不同 module 名/构建目标隔离）。
PyBind 层
从 p2p/pybind_engine.cc 复制为 p2p/tcpx/pybind_tcpx.cc，将 Endpoint 绑定到 TcpxEndpoint。模块名可以用 p2p_tcpx 或放在 uccl.tcpx 命名空间，避免覆盖现有 RDMA 模块。
构建
在 p2p/tcpx/Makefile 或 CMake 新增目标，生成独立的 pybind 模块与（如需要）C API .so，链接 -ldl。
运行时通过 PYTHONPATH 或 import 路径选择 TCPX 模块，不影响 RDMA 栈。
这些 .h 的改动原则

RDMA 的 .h 不必大改；TCPX 版本单独提供自己的 .h，保持与 RDMA 版的类/函数签名一致（名字可以不同，如 TcpxEndpoint，但对上层胶水保持兼容）。
uccl_engine.h 作为 C API 的“契约”可以不变（如果你在 p2p/tcpx/ 下导出相同的 uccl_engine_* 簇，放在独立 .so 即可）；或复制一份 uccl_engine_tcpx.h 供 TCPX 构建目标 include。
哪些“include 的文件”不能直接用 net_tcpx.h，需要你自己封装

不要把 nccl-plugin-gpudirecttcpx/src/net_tcpx.h 直接暴露在对外 .h 或 pybind 里（它是 NCCL 插件内部 API），否则会把插件内部符号/类型泄露到公共接口，且升级/兼容性风险大。
正确姿势是写一个薄封装（你已在 p2p/tcpx/tcpx_impl.cc 做了）：
用 dlopen + dlsym 解析 ncclNetPlugin_v8/v7 函数表；
在实现文件里（.cc）调用 init/devices/listen/connect/accept/regMr/isend/irecv/test/close...；
对上层（TcpxEndpoint）只暴露你自己的 minimal 接口（如 tcpx_get_device_count()、tcpx_listen/connect/accept 封装等）。
不要在公共 .h 直接 include net_tcpx.h；把它限制在 .cc 内部，外部通过你定义的 wrapper 函数交互（例如你现在的 tcpx_interface.h 就是对的）。
从 RDMA 复制后必须改的核心点

Endpoint 构造/析构：去掉 RDMA 资源（verbs、QP、CQ、代理线程），换成 TCPX 的监听/连接状态与必要的线程/锁。
OOB metadata：返回“真实 IP + 监听端口 + GPU index”。不要用 127.0.0.1；复用 RDMA OOB 获取 IP 的逻辑或自行实现。
connect/accept：通过 socket+TCPX 插件 handle 交换接入信息，建立 sendComm/recvComm。
reg/send/recv：用 ncclNet 的 regMr/isend/irecv/test/irecvConsumed/deregMr 流程；支持 host/device 指针判别（CUDA attrs）。
poll：把 uccl_engine_xfer_status 等转为 net->test 轮询，并在完成后释放注册的 MR/request。
错误映射与资源释放：保证每次失败路径都 closeSend/closeRecv/closeListen/deregMr/close(sock)。
测试建议

先在 p2p/tcpx/ 下用独立 pybind 模块/脚本调通连接（get_metadata/accept/connect），再逐步填充 send/recv。
用环境变量（如 UCCL_TCPX_PLUGIN_PATH/UCCL_TCPX_DEV）控制插件路径与设备选择；可加 UCCL_TCPX_DEBUG 打印内部日志。
总之，复制这三层（Endpoint/C API/PyBind）到 p2p/tcpx/ 并替换底层实现，是最快能跑通 TCPX 连接的路径；三处 .h 的改动保持谨慎（尽量不碰公共契约，或在 tcpx/ 下提供等价 .h），对外暴露接口保持与 RDMA 版兼容，上层脚本才能无感切换。