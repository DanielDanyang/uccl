# TCPX NIXL Plugin

This directory hosts the TCPX-backed C API that mirrors `p2p/uccl_engine.h`. It exposes the NCCL GPUDirectTCPX transport through the same surface area used by the RDMA engine (`p2p/uccl_engine.cc`), together with a lightweight smoke test.

## Layout
- `uccl_engine_tcpx.cc` 每 production TCPX engine that talks to the NCCL GPUDirectTCPX plugin via the `ncclNet` v7 interface.
- `uccl_engine_tcpx_nixl.cc` 每 legacy minimal example kept for reference.
- `test_tcpx_write.py` 每 Python smoke test that exercises the TCPX engine through ctypes.
- `Makefile` 每 helper to build the shared libraries consumed by the smoke test.
- `TESTING.md` 每 step-by-step validation guide for cloud hosts.

## Prerequisites
- CUDA + PyTorch available (the smoke test moves CUDA tensors).
- NCCL GPUDirectTCPX plugin built as a shared object (default name `libnccl-net.so`).
- Environment variables exported before importing `uccl`:
  ```bash
  export UCCL_TCPX_PLUGIN_PATH=/abs/path/to/libnccl-net.so
  export UCCL_TCPX_DEV=0
  export UCCL_RCMODE=1
  ```

## Quick Start
1. Build the TCPX glue:
   ```bash
   cd p2p/tcpx
   make
   ```
   This produces `libuccl_tcpx_engine.so` (used by the smoke test) and the legacy sample plugin.
2. From the repository root, run the TCPX smoke test:
   ```bash
   python p2p/tcpx/test_tcpx_write.py
   ```
3. A successful run prints `Local TCPX write test passed` after both processes finish.

For multi-node procedures and troubleshooting, see `TESTING.md`.

## Next Steps
- Extend `uccl_engine_tcpx.cc` with additional operations (e.g. read/FIFO helpers) in sync with `uccl_engine.h`.
- Integrate the TCPX backend with the higher level NIXL benchmarks (`p2p/benchmarks`).
