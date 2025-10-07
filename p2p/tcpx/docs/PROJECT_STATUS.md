# TCPX P2P Project Status

**Last Updated**: 2025-10-07  
**Project**: NIXL-TCPX Plugin for GCP A3-high  
**Partners**: Anyscale, Character AI

---

## Current Status

| Metric | Value |
|--------|-------|
| **Working** | Single-NIC P2P: 2.75 GB/s (server), 1.17 GB/s (client) per GPU |
| **Reference** | NCCL AllReduce: 19.176 GB/s bus bandwidth |
| **Investigation** | ✅ Complete - IRQ affinity NOT the bottleneck |
| **Next** | Single-process architecture refactor |

---

## What's Working ✅

1. **Single-NIC P2P**: All 4 NICs (eth1-4) work with TCPX devmem
2. **NCCL Reference**: 19.176 GB/s bus BW, proves environment works
3. **Infrastructure**: TCPX plugin, dp-manager, bootstrap all operational

## Known Issues ❌

1. **Multi-NIC per GPU**: Devmem conflicts in multi-process architecture
2. **Multi-channel degradation**: 4 channels on 1 NIC worse than 1 channel
3. **Performance gap**: ~7x vs NCCL (due to architecture, not IRQ)

---

## Investigation Results (2025-10-07)

### Key Findings

**IRQ Affinity** ❌ NOT the bottleneck
- NCCL uses default IRQ distribution (no custom IRQ bindings)
- eth3/eth4 have NUMA mismatch (IRQs on NUMA 0, threads on NUMA 1)
- Despite mismatch, NCCL achieves 19.176 GB/s

**Thread CPU Affinity** ✅ Important
- NCCL pins TX/RX threads to NUMA-local cores
- P2P doesn't set any thread affinity
- But limited benefit in multi-process architecture

**Multi-NIC Parallelism** ✅ Critical
- NCCL uses all 4 NICs simultaneously (~128 GB each)
- P2P uses only 1 NIC per GPU
- Requires single-process architecture

**Process Architecture** ✅ Root Cause
- NCCL: 1 process/node, 8 GPUs, all 4 NICs visible
- P2P: 8 processes/node, 1 GPU each, 1 NIC each
- Multi-process prevents NIC sharing (devmem conflicts)

### Performance Comparison

| Metric | P2P | NCCL | Gap |
|--------|-----|------|-----|
| Bandwidth | 2.75 GB/s/GPU | 19.176 GB/s bus BW | ~7x |
| NICs used | 1 | 4 | 4x |
| Channels/GPU | 1 | 8 | 8x |
| Thread affinity | None | NUMA-local | Different |
| **IRQ affinity** | Default | Default | **Same** |

---

## Project Timeline

### Phase 1: Initial Development (Complete)
- ✅ TCPX plugin integration
- ✅ Basic P2P communication working
- ✅ Single-NIC benchmark: 2.75 GB/s

### Phase 2: Multi-NIC Debugging (Complete)
- ✅ Identified "rx no cmsg" issue (loopback testing)
- ✅ Verified all 4 NICs working
- ✅ Documented devmem conflicts

### Phase 3: Multi-Channel Attempt (Complete)
- ✅ Tested multi-channel on single NIC (degraded)
- ✅ Tested multi-NIC per GPU (devmem conflicts)
- ✅ Concluded: requires architecture change

### Phase 4: IRQ Investigation (Complete 2025-10-07)
- ✅ Collected NCCL diagnostics
- ✅ Analyzed IRQ, CPU, NUMA, NIC data
- ✅ Concluded: IRQ affinity NOT the bottleneck
- ✅ Identified: Process architecture IS the bottleneck

### Phase 5: Single-Process Refactor (Current)
- ⏳ Status: Planning complete, ready to implement
- ⏳ Timeline: ~5-7 days
- ⏳ Target: >15 GB/s bus bandwidth

---

## Next Steps

### Immediate: Single-Process Refactor

**Goal**: Refactor from 8-process to 1-process architecture

**Why**: Enable multi-NIC per GPU (no devmem conflicts)

**Plan**: See `SINGLE_PROCESS_PLAN.md`

**Steps**:
1. Define architecture (0.5 day)
2. Refactor control plane (1 day)
3. Upgrade data plane (2 days)
4. Add thread affinity (0.5 day)
5. Instrumentation (0.5 day)
6. Validation (3 days)

**Timeline**: 7-8 days total

**Target**: >15 GB/s bus bandwidth (within 20% of NCCL)

---

## Documentation

**Core Docs** (read in order):
1. **PROJECT_STATUS.md** (this file) - Current status
2. **DIAGNOSTICS_SUMMARY.md** - IRQ investigation results
3. **SINGLE_PROCESS_PLAN.md** - Refactor plan
4. **AI_HANDOFF_PROMPT.md** - Context for new developers

**Archive**: `docs/archive/` - Historical debug docs

---

## Key Learnings

1. **Always test on separate nodes** - Loopback doesn't work with TCPX devmem
2. **ethtool rx_devmem_pkts is ground truth** - Authoritative devmem indicator
3. **NCCL proves environment works** - Don't waste time on system debugging
4. **Process architecture matters** - Multi-process limits NIC sharing
5. **IRQ affinity is NOT critical** - NCCL uses default and achieves high perf
6. **Thread CPU affinity IS important** - NCCL pins to NUMA-local cores
7. **Multi-NIC parallelism is key** - NCCL's 4 NICs provide 4x parallelism
8. **Measure everything** - CPU, IRQ, bandwidth before making changes
9. **Document negative results** - Knowing what doesn't work is valuable

---

## Environment

- **Platform**: GCP A3-high (2 nodes)
- **GPUs**: 8× H100 per node
- **NICs**: 4× gVNIC (eth1-4, 200 Gbps each)
- **Network**: TCPX (GPUDirect over TCP with devmem-tcp)
- **OS**: Ubuntu with custom kernel (devmem-tcp support)

### Hardware Topology

**CPU**:
- 208 CPUs (104 cores × 2 HT)
- NUMA 0: CPUs 0-51, 104-155
- NUMA 1: CPUs 52-103, 156-207

**NIC-to-NUMA**:
- eth1, eth2 → NUMA 0
- eth3, eth4 → NUMA 1

**GPU-to-NIC** (current P2P):
- GPU 0-1 → eth1
- GPU 2-3 → eth2
- GPU 4-5 → eth3
- GPU 6-7 → eth4

---

## Quick Start

### Run P2P Benchmark
```bash
# Node 0 (Server)
./run_p2p_fullmesh.sh server

# Node 1 (Client)
./run_p2p_fullmesh.sh client <NODE0_IP>

# Check results
grep "PERF.*Avg.*BW:" logs/fullmesh_*.log
```

### Run NCCL Reference
```bash
cd ../../collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_p2p_fullmesh.sh` | P2P launcher (8 processes) |
| `tests/test_tcpx_perf_multi.cc` | P2P benchmark program |
| `src/channel_manager.cc` | Channel management |
| `scripts/node_ips/tcpx.txt` | Node IPs |

---

## For New Developers

1. Read **DIAGNOSTICS_SUMMARY.md** for IRQ investigation results
2. Read **SINGLE_PROCESS_PLAN.md** for refactor roadmap
3. Read **AI_HANDOFF_PROMPT.md** for complete context
4. Run current P2P to verify environment

---

## Success Criteria

### Phase 5 (Single-Process Refactor)
- [ ] Single-process P2P works (1 GPU smoke test)
- [ ] No devmem conflicts (8 GPUs, multi-NIC)
- [ ] Multi-NIC works (4 NICs simultaneously)
- [ ] Multi-channel works (8 channels/GPU)
- [ ] Bandwidth >15 GB/s bus BW
- [ ] Within 20% of NCCL performance

### Overall Project
- [ ] P2P benchmark achieves >10 GB/s per GPU
- [ ] Understand root cause of any remaining gap
- [ ] Clear documentation of what works and what doesn't
- [ ] Production-ready NIXL-TCPX plugin

---

**Status**: Phase 5 planning complete, ready to implement  
**Next Milestone**: Single-process refactor (7-8 days)  
**Target**: >15 GB/s bus bandwidth

