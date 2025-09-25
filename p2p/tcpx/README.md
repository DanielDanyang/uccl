# TCPX NIXL Backend

为 `benchmark_nixl.py` 创建 TCPX 后端插件，参考 mooncake 实现。

## 📁 项目结构

```
p2p/tcpx/
├── tcpx_interface.h          # TCPX API 定义
├── tcpx_impl.cc              # TCPX 插件集成实现 (经过测试)
├── tests/
│   ├── test_device_discovery.cc  # 设备发现测试
│   ├── test_connection.cc         # 连接测试 (主机内存)
│   ├── test_tcpx_transfer.cc      # ⭐ GPU-to-GPU传输测试 (核心功能)
│   ├── test_tcpx.cc              # 基础功能测试
│   └── test_performance.cc       # 真实性能测试
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
make test_tcpx_transfer  # GPU传输测试 (推荐)
make test_tcpx_transfer
make test_tcpx
```

### 运行测试
```bash
# 基础功能测试
export UCCL_TCPX_DEBUG=1
./tests/test_tcpx

# 设备发现测试
./tests/test_device_discovery

# 真实性能测试 (需要两个节点)
# 服务器端:
./tests/test_performance server
# 客户端:
./tests/test_performance client <server_ip>

# 连接测试 (仅握手)
./tests/test_connection server
./tests/test_connection client <server_ip>

# ⭐ GPU-to-GPU传输测试 (核心功能测试)
# 这是最重要的测试，验证TCPX的GPU直接内存传输能力
./tests/test_tcpx_transfer server
./tests/test_tcpx_transfer client <server_ip>

# 使用脚本运行 (推荐)
./run_tcpx_test.sh transfer server      # 服务器端
./run_tcpx_test.sh transfer <server_ip> # 客户端
```

## 🎯 **test_tcpx_transfer.cc 详细说明**

### 📋 **测试目标**
`test_tcpx_transfer.cc` 是验证TCPX GPU-to-GPU直接传输的核心测试，专门测试：
- GPU设备内存的分配和4KB对齐
- CUDA内存注册到TCPX (NCCL_PTR_CUDA)
- GPU-to-GPU的直接数据传输
- 数据完整性验证

### 🔧 **测试流程**
1. **服务器端**：
   - 初始化TCPX设备
   - 创建监听连接并生成句柄
   - 通过bootstrap TCP连接发送句柄给客户端
   - 接受TCPX连接
   - 分配4KB对齐的GPU内存
   - 注册GPU内存到TCPX
   - 等待接收数据并验证内容

2. **客户端**：
   - 初始化TCPX设备
   - 通过bootstrap TCP连接获取服务器句柄
   - 建立TCPX连接
   - 分配4KB对齐的GPU内存
   - 将测试消息复制到GPU内存
   - 注册GPU内存到TCPX
   - 发送数据到服务器

### ✅ **成功标志**
- 服务器接收到完整数据
- 数据内容与发送的测试消息完全匹配
- 无CUDA错误或TCPX传输错误

### 🚨 **常见问题**
- **内存对齐**：GPU内存必须4KB对齐
- **gpumemd依赖**：需要gpumemd服务支持GPU DMA-BUF
- **环境变量**：需要正确设置TCPX相关环境变量

## 🎯 开发计划

### 当前状态 ✅
- TCPX API 层已完成并测试
- 基础连接功能已验证
- GPU-to-GPU传输测试已实现

### 下一步 🔄
1. **验证transfer测试** - 确保GPU传输功能正常
2. **NIXL 插件** - 创建类似 mooncake 的后端插件
3. **集成测试** - 让 benchmark_nixl.py 使用 tcpx 后端

## 📚 参考

- `mooncake/` - NIXL 后端插件参考实现
- `p2p/uccl_engine.h` - 引擎接口参考
- `nccl-plugin-gpudirecttcpx/src/net_tcpx.h` - TCPX API 定义
- `docs/tcpx_transfer.md` - GPU 传输测试流程与注意事项

## 官方推荐路径（GPUDirect TCPX）

Google 发布的 TCPX 插件默认走 GPU DMA-BUF / `gpumemd` 服务链路：

- `kUseDmaBuf` 默认开启，意味着插件期望直接对 GPU 内存做 DMA 映射，而不是落回 host bounce buffer【nccl-plugin-gpudirecttcpx/src/flags.cc:32】。
- 当调用 `tcpx_reg_mr(..., NCCL_PTR_CUDA, ...)` 时，代码会强制要求 4 KB 对齐并通过 `gpu_tx_reg_mr()` 向 `gpumemd` 请求 DMA-BUF FD，若未能拿到则直接返回 `tcpxInternalError`【nccl-plugin-gpudirecttcpx/src/net_tcpx.cc:792-809】【nccl-plugin-gpudirecttcpx/src/gpu/cuda_wrapper.cu:226-246】。
- 接收端同样通过 `gpumem_import()` / `GpumemImport()` 走 UNIX 域 socket 与 gpumemd 协议，期望在 `/tmp/nvdma-<GPU PCI>` 和 `<prefix>/get_gpu_fd_*` 提供共享句柄【nccl-plugin-gpudirecttcpx/src/gpu/rx_pool.cu:31-124】。

因此，官方推荐路径要保证：

1. **gpumemd 服务运行在两台节点上**（通常由 Google 提供的 systemd 单元或容器部署），负责在每块 GPU 暴露 `/tmp/nvdma-<pci>` 文件及 `unix://<prefix>/get_gpu_fd_*` 控制通道。
2. **应用使用 CUDA 设备内存** 作为收发缓冲区，确保指针按 4KB 对齐（可以通过 `cudaMalloc`/`cudaMallocAsync` 或在注册前手动对齐）。
3. **先建立 CUDA 上下文**（`cudaSetDevice` 或 `cudaFree(0)`）再初始化 TCPX，使 `gpu_current_dev` / `cuCtxSetCurrent` 调用能够成功。
4. **保持 DMA-BUF 相关环境变量为默认值**，不要人为关闭 GPU 内存导入；只有在调试 fallback 时才改。

## 环境变量（官方配置，双方一致）

```bash
# 控制面网卡（Bootstrap / 控制通道）
export NCCL_SOCKET_IFNAME=eth0

# 数据面 NIC 列表（按实际拓扑排列，需与 gpumemd/network 绑核设置一致）
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"

# 启用 gpumemd + GPUDirect RX (默认值为 1，显式写出避免被其他脚本覆盖)
export NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=1
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"

# DMA-BUF、流表等保持官方默认
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE=1
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0

# 调试项（可选）
export UCCL_TCPX_DEBUG=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

> 说明：如果需要暂时退回 host bounce 模式，再把 `NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT` 设为 0；但那属于降级手段，并不符合官方推荐流程。

## 推荐测试流程（GPU DMA-BUF 路径）

```
            ┌─────────────────────┐                   ┌─────────────────────┐
            │       Server        │                   │       Client        │
            └─────────┬───────────┘                   └─────────┬───────────┘
                      │                                           │
          tcpx_get_device_count ✅                    tcpx_get_device_count ✅
                      │                                           │
              tcpx_listen ✅                              接收 bootstrap 句柄 ✅
                      │                                           │
         发送 128B NCCL 句柄 ✅                        tcpx_connect_v5 ✅
                      │                                           │
              tcpx_accept_v5 ✅                           获得 send_comm ✅
                      │                                           │
   ┌─────────────── GPU 数据面 ───────────────┐        注册 CUDA 缓冲区 (对齐) ✅
   │  tcpx_reg_mr(NCCL_PTR_CUDA) ✅         │        tcpx_reg_mr(NCCL_PTR_CUDA) ✅
   │  gpumem_import / gpumemd handshake ✅   │        gpumemd: get_gpu_fd_* ✅
   │  tcpx_irecv ✅                          │        tcpx_isend ✅
   │  tcpx_test → done=1 ✅                  │<───── DMA-BUF 输送数据 ───┘
   │  解包数据 & 校验 ✅                     │
   └────────────────────────────────────────┘
                      │
          打印 "Hello from TCPX client!" ✅
```

## 当前阻塞点

1. **gpumemd 服务状态未知**：需要在两台 H100 节点确认 `systemctl status gpumemd`（或云端提供的等效命令），确保上述 UNIX socket、`/run/tcpx/get_gpu_fd_*` 可用。
2. **TCPX 测试仍在 host fallback**：尽管 CUDA runtime 已确认可用（NCCL-test 已跑通），日志显示 `tcpx_reg_mr` 仍然落在 host path，表明 DMA-BUF 注册被拒绝。建议在 `test_connection.cc` 调试输出 `rc_reg` 的详细错误码，并在 `tcpx_reg_mr` 失败时打印 `errno`/`ret`。
3. **需要恢复默认环境**：本地 README 之前为了调试 EFAULT 手动设置了 `NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=0` 与 `UCCL_TCPX_FORCE_HOST_RECV=1`，这些变量会绕过官方路径，现已移除，云端环境也需要同步更新。
4. **CUDA 上下文初始化**：确保 server/client 在 TCPX 初始化前调用 `cudaSetDevice(dev_id);`（或 `cudaFree(0);`），避免 `gpu_current_dev`/`cuCtxSetCurrent` 返回错误，导致 gpumemd 交互失败。

