# TCPX P2P Benchmark

> **UPDATE (2025-10-08)**: Multi-channel configuration!
> **Key insight**: Each channel = 1 TCPX connection. Use multiple channels for pipeline parallelism.
>
> **Quick Start (Recommended - 4 connections per GPU)**:
> ```bash
> # Single GPU pair (GPU 0 on both nodes)
> # Server
> ./run_p2p_fullmesh.sh server 0
>
> # Client
> ./run_p2p_fullmesh.sh client <SERVER_IP> 0
>
> # Full-mesh (all 8 GPUs)
> # Server
> ./run_p2p_fullmesh.sh server
>
> # Client
> ./run_p2p_fullmesh.sh client <SERVER_IP>
> ```
>
> **Architecture**:
> - Each GPU: 2 channels × 2 sockets = 4 sockets total (UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=2)
> - GPU → NIC mapping: {0,1}→eth1, {2,3}→eth2, {4,5}→eth3, {6,7}→eth4
> - 2 GPUs share 1 NIC → 8 sockets per NIC (MAX_SOCKETS=8)
>
> **Primary docs**:
> - [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 实施总结（中文）
> - [docs/AI_HANDOFF_PROMPT.md](docs/AI_HANDOFF_PROMPT.md) - AI 交接文档
> - [HANDOFF_README.md](HANDOFF_README.md) - 项目交接
> - [DEBUG_GUIDE.md](DEBUG_GUIDE.md) - 调试指南

GPU-to-GPU P2P communication using Google's TCPX (GPUDirect over TCP) for GCP A3-high instances.

## Quick Start

```bash
# Node 0 (Server)
./run_p2p_fullmesh.sh server

# Node 1 (Client)
./run_p2p_fullmesh.sh client <NODE0_IP>

# Check results
grep "PERF.*Avg.*BW:" logs/fullmesh_*.log
```

## Current Status (2025-10-07)

| Metric | Value |
|--------|-------|
| **Working** | Single-NIC P2P: 2.75 GB/s (server), 1.17 GB/s (client) per GPU |
| **Reference** | NCCL AllReduce: 19.176 GB/s bus bandwidth |
| **Investigation** | ✅ Complete - IRQ affinity NOT the bottleneck |
| **Next** | Single-process architecture refactor to enable multi-NIC |

**Key Finding**: Performance gap due to process architecture (8 processes vs 1), not IRQ affinity.

## Documentation

**Read in order**:
1. **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current status and next steps
2. **[docs/DIAGNOSTICS_SUMMARY.md](docs/DIAGNOSTICS_SUMMARY.md)** - IRQ investigation results
3. **[docs/SINGLE_PROCESS_PLAN.md](docs/SINGLE_PROCESS_PLAN.md)** - Refactor plan
4. **[docs/AI_HANDOFF_PROMPT.md](docs/AI_HANDOFF_PROMPT.md)** - Context for new developers

**Archive**: `docs/archive/` (historical debug docs, see archive/README.md)

## Architecture

**Current (Multi-Process)**:
- 8 processes/node, 1 GPU each, 1 NIC each
- **Limitation**: Cannot share NICs (devmem conflicts)

**NCCL (Single-Process)**:
- 1 process/node, 8 GPUs, all 4 NICs visible
- **Advantage**: NIC sharing enables multi-NIC per GPU

**Next**: Refactor to single-process architecture (see SINGLE_PROCESS_PLAN.md)

## Environment

- **Platform**: GCP A3-high (2 nodes)
- **GPUs**: 8× H100 per node
- **NICs**: 4× gVNIC (eth1-4, 200 Gbps each)
- **Network**: TCPX (devmem-tcp kernel API)

## Key Files

| File | Purpose |
|------|---------|
| `run_p2p_fullmesh.sh` | Launcher (8 processes) |
| `tests/test_tcpx_perf_multi.cc` | P2P benchmark |
| `src/channel_manager.cc` | Channel management |
| `scripts/node_ips/tcpx.txt` | Node IPs |

## Build

```bash
make clean && make
```

## Troubleshooting

**"rx no cmsg" error**: Run on separate nodes (not loopback)
**Low performance**: Check `docs/DIAGNOSTICS_ANALYSIS.md`
**Connection failures**: Verify NICs are up, dp-manager running

---

**Last Updated**: 2025-10-07
**Status**: IRQ investigation complete, planning single-process refactor

