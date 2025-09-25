# TCPX Device-to-Device Transfer Flow

这份文档对应 `tests/test_tcpx_transfer.cc`，帮助在两节点环境中验证 GPUDirect TCPX 的 GPU 端到端数据通路。

## ASCII 流程图

```
            ┌────────────────────────────┐                    ┌────────────────────────────┐
            │            Server          │                    │            Client          │
            └──────────────┬─────────────┘                    └──────────────┬─────────────┘
                           │                                               │
                 tcpx_get_device_count ✅                         tcpx_get_device_count ✅
                           │                                               │
                    tcpx_listen(dev0) ✅                                   │
                           │                                               │
         ┌─ bootstrap socket (TCP 12345) ───────────────┬──────────────────┘
         │ exchange 128B ncclNet handle                 │
         └──────────────────────────────────────────────┘
                           │                                               │
       async accept_v5 → recv_comm ready ✅                      tcpx_connect_v5 ✅
                           │                                               │
           allocate & align CUDA buffer (4KB)                allocate & align CUDA buffer (4KB)
                           │                                               │
          tcpx_reg_mr(ptr, 4096, NCCL_PTR_CUDA) ✅        tcpx_reg_mr(ptr, 4096, NCCL_PTR_CUDA) ✅
                           │                                               │
                tcpx_irecv(ptr, tag=42) ✅               填充 "Hello from TCPX client!"
                           │                                               │
                         tcpx_test ◄───────────── tcpx_isend(ptr, size=24, tag=42)
                           │                      tcpx_test + 延迟 500ms (等待消费)
                cuMemcpyDtoH -> Hex dump + 校验                     │
                           │                                               │
         tcpx_dereg_mr / close_recv / close_listen        tcpx_dereg_mr / close_send
```

关键要点：
* 双方都注册 4KB 对齐的 CUDA 内存，符合 GPUDirect TCPX DMA-BUF 要求（`kRegisteredBytes = 4096`）。
* 客户端在 send 完成后额外等待 500ms，避免服务端尚未完成 `tcpx_test` 时连接被关闭。
* 服务端将收到的 GPU 缓冲区拷回宿主并输出十六进制预览，`memcmp` 校验字符串是否匹配。

## 运行步骤

1. **服务器节点**

   ```bash
   # 假设位于 10.64.147.221
   ./tests/test_tcpx_transfer server
   ```

2. **客户端节点**

   ```bash
   # 替换为服务器 IP
   ./tests/test_tcpx_transfer client 10.64.147.221
   ```

若想自动化，可使用 `run_tcpx_test.sh`：

```bash
./run_tcpx_test.sh transfer server
./run_tcpx_test.sh transfer 10.64.147.221
```

## 依赖条件

- gpumemd 服务运行正常（`/run/tcpx/get_gpu_fd_*` 可访问）。
- 环境变量遵循官方建议，例如：

  ```bash
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"
  export NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=1
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"
  ```

- CUDA Runtime 可用，两端能成功运行 `nvidia-smi`、`nccl-tests`。

## 常见问题排查

- **`tcpx_reg_mr` 返回 3**：注册内存未对齐或 gpumemd 未提供 DMA-BUF。检查 `/run/tcpx/` 以及 CUDA 指针是否 4KB 对齐。
- **`tcpx_test` 返回 2 / "Connection closed by remote peer"**：确保客户端在 `tcpx_test` 成功后保留连接（本测试通过 500ms 延迟解决）。
- **Payload mismatch**：查看服务端日志中的十六进制预览，若全为零，可能是发送侧提前关闭或 `cuMemcpyHtoD` 未成功。

测试通过后，日志应看到 `SUCCESS: Payload matches expected string`，且 GPUDirectTCPX 的 `tcpxClose` 输出 `All bytes: 24`。
