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

# 连接测试 (需要两个节点)
# 服务器端 (gke-character-k8s-gcp5-h100-3: 10.0.1.46):
./tests/test_connection server
# 客户端 (gcp5-h100-2: 10.0.1.170):
./tests/test_connection client 10.0.0.250
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

## 环境变量建议（两端都需要设置）

```bash
# 控制面网卡（TCPX 控制连接）
export NCCL_SOCKET_IFNAME=eth0

# 数据面网卡列表（逗号分隔，按实际环境调整）
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"

# 如未启用/部署 gpumemd，或仅需先验证 TCP 传输路径，关闭 CUDA IPC 接收内存导入：
export NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=0

# 如需缩短流表等待时间（可选）
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
```

            ┌─────────────────────┐
            │       Server        │
            └─────────┬───────────┘
                      │
        Step 1: 初始化 TCPX (tcpx_get_device_count)
                      │
        Step 2: 监听设备 (tcpx_listen)
            生成 NCCL handle (128B)
                      │
        Step 3: 建立 bootstrap TCP socket
                      │
        发送 handle 给 client (send)
                      │
        Step 4: 等待连接
    ┌─────────────────┴─────────────────┐
    │                                   │
tcpx_accept_v5                  connect_to_bootstrap_server
 分配 recv_dev_handle buffer            │
 得到 recv_comm                        │
    │                                   │
    ▼                                   ▼
注册接收缓冲区 (tcpx_reg_mr)     Step 3: 接收 handle
发起接收请求 (tcpx_irecv)          用 handle 调用 tcpx_connect_v5
轮询完成 (tcpx_test)               得到 send_comm
    │                                   │
    ▼                                   │
 Step 4: 等待接收数据                 Step 4: 准备发送数据
 如果完成：打印 "Hello..."             注册发送缓冲区 (tcpx_reg_mr)
 否则超时                              调用 tcpx_isend
                                       轮询完成 (tcpx_test)
    │                                   │
    └─────────────────┬─────────────────┘
                      │
           === 测试完成 (COMPLETED) ===

                 ┌─────────────────────┐
                 │        Server       │
                 └──────────┬──────────┘
                            │
        Step 1: 初始化 TCPX (tcpx_get_device_count) ✅
                            │
        Step 2: 监听设备 (tcpx_listen) ✅
            生成 NCCL handle (128B)
                            │
        Step 3: 建立 bootstrap TCP socket ✅
                            │
        发送 handle 给 client (send) ✅
                            │
        Step 4: 等待连接 ✅
        ┌──────────────────┴──────────────────┐
        │                                     │
 tcpx_accept_v5 ✅                     connect_to_bootstrap_server ✅
 分配 recv_dev_handle buffer           Step 3: 接收 handle ✅
 得到 recv_comm ✅                      用 handle 调用 tcpx_connect_v5 ✅
        │                                得到 send_comm ✅
        ▼
 注册接收缓冲区 (tcpx_reg_mr) ✅        Step 4: 准备发送数据
 发起接收请求 (tcpx_irecv) ✅           注册发送缓冲区 (tcpx_reg_mr) ✅
        │                                调用 tcpx_isend ✅
        ▼                                轮询完成 (tcpx_test) ❌
 轮询完成 (tcpx_test) ❌
   │   recvfrom(fd=53, buf, 24) = EFAULT
   │   → 插件 host-mem data-socket 路径出错
   │
   ▼
 Step 4: 等待接收数据 ❌
 打印 "Hello..." ← 未成功
 （TIMEOUT: no data）

        │
        └──────────────────┬──────────────────┘
                           │
              === 测试未完成 (卡在数据传输阶段) ===

