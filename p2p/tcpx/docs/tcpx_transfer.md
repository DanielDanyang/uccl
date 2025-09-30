# TCPX Device-to-Device Transfer Flow

This document describes the end-to-end GPU-to-GPU transfer flow implemented in `tests/test_tcpx_transfer.cc`.

## Transfer Flow Diagram

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
                tcpx_irecv(ptr, tag=42) ✅               Write "Hello from TCPX client!"
                           │                                               │
                         tcpx_test ◄───────────── tcpx_isend(ptr, size=24, tag=42)
                           │                      tcpx_test + wait 500ms
                cuMemcpyDtoH -> Hex dump + verify                  │
                           │                                               │
         tcpx_dereg_mr / close_recv / close_listen        tcpx_dereg_mr / close_send
```

## Key Points

- Both sides register 4KB aligned CUDA memory (GPUDirect TCPX DMA-BUF requirement)
- Client waits 500ms after send completion to avoid premature connection close
- Server copies received GPU buffer to host for hex dump and validation
- Payload: "Hello from TCPX client!" (23 bytes)

## Running the Test

**Server:**
```bash
./tests/test_tcpx_transfer server
```

**Client:**
```bash
./tests/test_tcpx_transfer client <server_ip>
```

## Expected Output

**Server:**
```
[DEBUG] === TCPX GPU-to-GPU transfer test ===
[DEBUG] Running in SERVER mode
[DEBUG] Listening on device 0
[DEBUG] Connection accepted; recv_comm=0x...
[DEBUG] Registered server receive buffer ptr=0x..., bytes=4096
[DEBUG] Waiting for client data, expected bytes=23
[DEBUG] Request metadata: frag_count=1
[DEBUG] descriptor[0] src_off=... len=23 dst_off=0
[DEBUG] Launching device unpack (D2D copies), total_bytes=23
[DEBUG] Device unpack completed successfully
[DEBUG] Received data matches expected payload
[DEBUG] ✓ Test PASSED
```

**Client:**
```
[DEBUG] === TCPX GPU-to-GPU transfer test ===
[DEBUG] Running in CLIENT mode
[DEBUG] Connecting to server at <server_ip>
[DEBUG] Connection established; send_comm=0x...
[DEBUG] Registered client send buffer ptr=0x..., bytes=4096
[DEBUG] Sending 23 bytes
[DEBUG] ✓ Test PASSED
```

## Requirements

- gpumemd service running (`/run/tcpx/get_gpu_fd_*` accessible)
- Environment variables configured:
  ```bash
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME="eth1,eth2,eth3,eth4"
  export NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=1
  export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"
  ```
- CUDA Runtime available on both nodes

## Troubleshooting

- **`tcpx_reg_mr` returns 3**: Memory not aligned or gpumemd not providing DMA-BUF. Check `/run/tcpx/` and ensure CUDA pointer is 4KB aligned.
- **`tcpx_test` returns 2 / "Connection closed by remote peer"**: Ensure client keeps connection alive after send (test uses 500ms delay).
- **Payload mismatch**: Check server hex dump. If all zeros, sender may have closed early or `cuMemcpyHtoD` failed.
