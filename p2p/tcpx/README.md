# TCPX GPU-to-GPU Transfer

TCPX-based GPU-to-GPU data transfer implementation using [nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx).

## Features

- TCPX connection management (listen, accept, connect)
- CUDA device buffer registration
- Async send/receive operations
- RX metadata parsing (CMSG scatter-gather lists)
- Multiple unpack implementations:
  - **D2D**: Device-to-device memcpy (default)
  - **Host**: Host-mediated gather (fallback)
  - **Kernel**: CUDA kernel-based unpack (experimental)

## Quick Start

### Build

```bash
cd p2p/tcpx
make clean
make
```

### Run Test

**Option 1: Using test script (recommended)**

```bash
# Server node
./run_tcpx_test.sh transfer server

# Client node
./run_tcpx_test.sh transfer <server_ip>

# Connection test only
./run_tcpx_test.sh connection server
./run_tcpx_test.sh connection <server_ip>
```

The script automatically sets all required TCPX environment variables.

**Option 2: Manual execution**

```bash
# Server
./tests/test_tcpx_transfer server

# Client
./tests/test_tcpx_transfer client <server_ip>
```

### Select Unpack Implementation

```bash
export UCCL_TCPX_UNPACK_IMPL=d2d    # Default
export UCCL_TCPX_UNPACK_IMPL=host   # Fallback
export UCCL_TCPX_UNPACK_IMPL=kernel # Experimental
```

## Architecture

```
Client                          Server
  │                               │
  ├─ cudaMalloc(send_buf)        ├─ cudaMalloc(recv_buf)
  ├─ tcpx_reg_mr(send_buf)       ├─ tcpx_reg_mr(recv_buf)
  ├─ Write payload to GPU        │
  ├─ tcpx_isend() ──────────────>├─ tcpx_irecv()
  │                               ├─ Poll tcpx_test()
  │                               ├─ Parse RX metadata (CMSG)
  │                               ├─ Build UnpackDescriptorBlock
  │                               ├─ Execute unpack (D2D/host/kernel)
  │                               ├─ Validate received data
  └─ Cleanup                      └─ Cleanup
```
## Components

### RX Metadata Parsing (`rx/`)
- `rx_cmsg_parser.{h,cc}`: Parse control messages with scatter-gather lists
- `rx_descriptor.{h,cc}`: Build unpack descriptor blocks

### Device Unpack (`device/`)
- `unpack_kernels.cu`: CUDA kernels for GPU-side unpack
- `unpack_launch.{h,cu}`: Kernel launcher and configuration

### Tests (`tests/`)
- `test_tcpx_transfer.cc`: End-to-end GPU-to-GPU transfer
- `test_connection.cc`: Connection handshake only
- `test_rx_cmsg_parser.cc`: CMSG parser unit test
- `test_rx_descriptor.cc`: Descriptor builder unit test

## Requirements

- CUDA Toolkit 12.x
- nccl-plugin-gpudirecttcpx at `/usr/local/tcpx/`
- Linux kernel with devmem-tcp support
- H100 or compatible GPU

## File Structure

```
p2p/tcpx/
├── README.md
├── Makefile
├── tcpx_interface.h
├── tcpx_impl.cc
├── include/
│   └── tcpx_structs.h
├── rx/
│   ├── rx_cmsg_parser.{h,cc}
│   └── rx_descriptor.{h,cc}
├── device/
│   ├── unpack_kernels.cu
│   └── unpack_launch.{h,cu}
├── tests/
│   ├── test_tcpx_transfer.cc
│   ├── test_connection.cc
│   ├── test_rx_cmsg_parser.cc
│   └── test_rx_descriptor.cc
└── docs/
    ├── TCPX_LOGIC_MAPPING.md
    └── tcpx_transfer.md
```

## Implementation Details

### TCPX Plugin Integration

Uses NCCL plugin v7 APIs:
- `ncclNetPlugin_v7`: Main plugin interface
- `ncclNet_v7_t`: Network operations (init, listen, accept, connect)
- Device handle management for GPU memory registration

### RX Metadata Format

TCPX delivers fragments with metadata via control messages:
```c
struct loadMeta {
  uint32_t src_off;  // Offset in bounce buffer
  uint32_t len;      // Fragment length
  uint32_t dst_off;  // Offset in destination buffer
  uint32_t _pad;
};
```

### Memory Management

- **Bounce buffers**: TCPX plugin managed (devmem-tcp mapped GPU memory)
- **Destination buffers**: Application allocated via `cudaMalloc`
- **Descriptor blocks**: Host-side structures for unpack operations

## Known Issues

1. **Kernel unpack**: Direct kernel access to devmem-tcp bounce buffers not working
   - Workaround: D2D copy to staging buffer (adds overhead)
   - Status: Under investigation

2. **Multi-fragment transfers**: D2D issues one copy per fragment
   - Future: Kernel-based vectorized unpack

## References

- [nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx)
- [NCCL Plugin API v7](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h)

## License

See LICENSE.txt for license information.

