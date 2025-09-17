# TCPX Testing Playbook

Minimal checklist for validating the TCPX-backed UCCL engine on a host without RDMA support.

## 1. Prerequisites
- CUDA-visible GPU with PyTorch installed.
- NCCL GPUDirectTCPX plugin shared object (built from [google/nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx)).
- `python -c "from uccl import p2p"` succeeds (UCCL installed in the environment).

## 2. Environment
```bash
export UCCL_TCPX_PLUGIN_PATH=/abs/path/to/libnccl-net.so
export UCCL_TCPX_DEV=0              # choose the TCPX NIC index
export UCCL_RCMODE=1                # enable one-sided ops in the test harness
```
Verify the plugin path exists (`ls $UCCL_TCPX_PLUGIN_PATH`).

## 3. Build the glue
```bash
cd /path/to/uccl/p2p/tcpx
make
```
This creates `libuccl_tcpx_engine.so`, which the smoke test loads via ctypes.

## 4. Smoke test
From the repository root:
```bash
python p2p/tcpx/test_tcpx_write.py
```
Expected output:
- Engine metadata exchange shows the TCPX plugin/device being used.
- Both server and client print tensor samples.
- The script finishes with `Local TCPX write test passed`.

## 5. Troubleshooting
- **Import errors** 每 reinstall UCCL (`python -m build` and `pip install dist/*.whl` or `pip install -e .`).
- **Plugin not found** 每 double check `UCCL_TCPX_PLUGIN_PATH` and file permissions.
- **No TCPX device** 每 adjust `UCCL_TCPX_DEV` based on the NIC index reported by the plugin.
- **CUDA errors** 每 confirm `nvidia-smi` output and `python -c "import torch; print(torch.cuda.is_available())"` are healthy.

## 6. Going further
Run the NIXL benchmarks (`p2p/benchmarks/benchmark_nixl.py`) on paired hosts with the same TCPX environment variables to stress more complex transfer flows.
