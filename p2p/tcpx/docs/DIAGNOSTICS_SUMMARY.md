# NCCL Diagnostics Analysis - Summary

**Date**: 2025-10-07  
**Test**: NCCL AllReduce with TCPX (2 nodes, 8 GPUs/node)  
**Performance**: 19.176 GB/s bus bandwidth  

---

## TL;DR

**Question**: Why does NCCL achieve 19.176 GB/s while P2P achieves only 2.75 GB/s?

**Answer**:
- ❌ **NOT** IRQ affinity (NCCL uses default round-robin IRQ distribution)
- ✅ **YES** Thread CPU affinity (NCCL pins threads to NUMA-local cores)
- ✅ **YES** Multi-NIC parallelism (NCCL uses 4 NICs, P2P uses 1)
- ✅ **YES** Process architecture (NCCL: 1 process enables NIC sharing)

---

## Key Findings

### 1. IRQ Affinity is NOT the Bottleneck ❌

**Evidence**:
- NCCL does NOT set `NCCL_GPUDIRECTTCPX_TX/RX_IRQ_BINDINGS` (value = null)
- All gVNIC IRQs use default round-robin distribution (CPUs 1-64)
- **Critical**: eth3/eth4 IRQs on NUMA 0 CPUs (33-48), but NCCL threads on NUMA 1 CPUs (60-87)
- Despite NUMA mismatch, NCCL achieves 19.176 GB/s

**Conclusion**: IRQ affinity optimization is NOT necessary.

### 2. Thread CPU Affinity IS Important ✅

**Evidence**:
- NCCL sets `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `RX_BINDINGS`
- TX threads: cores 8-21, 112-125 (NUMA 0) and 60-73, 164-177 (NUMA 1)
- RX threads: cores 22-35, 126-139 (NUMA 0) and 74-87, 178-191 (NUMA 1)
- All cores are NUMA-local to their NICs
- P2P does NOT set any thread affinity

**Recommendation**: Add thread CPU affinity to P2P (but limited benefit in multi-process architecture).

### 3. Multi-NIC Parallelism IS Critical ✅

**Evidence**:
- NCCL uses all 4 NICs simultaneously (~128 GB transferred per NIC)
- P2P uses only 1 NIC per GPU
- NCCL's single-process architecture enables NIC sharing
- P2P's multi-process architecture causes devmem conflicts

**Recommendation**: Refactor to single-process architecture.

### 4. Process Architecture Matters ✅

| Architecture | NCCL | P2P |
|--------------|------|-----|
| Processes/node | 1 | 8 |
| GPUs/process | 8 | 1 |
| NICs visible | 4 | 1 |
| Channels/GPU | 8 | 1 |
| NIC sharing | ✅ Yes | ❌ No (devmem conflicts) |

---

## Performance Comparison

| Metric | P2P | NCCL | Gap |
|--------|-----|------|-----|
| Bandwidth | 2.75 GB/s/GPU | 19.176 GB/s bus BW | ~7x |
| NICs used | 1 | 4 | 4x |
| Channels | 1/GPU | 8/GPU | 8x |
| Thread affinity | None | NUMA-local | Different |
| **IRQ affinity** | Default | Default | **Same** |

---

## Hardware Topology

### CPU Configuration
- **Total CPUs**: 208 (104 cores × 2 HT)
- **NUMA nodes**: 2
  - NUMA 0: CPUs 0-51, 104-155
  - NUMA 1: CPUs 52-103, 156-207

### NIC-to-NUMA Mapping
| NIC | NUMA | IRQ CPUs | NCCL TX Cores | NCCL RX Cores |
|-----|------|----------|---------------|---------------|
| eth1 | 0 | 1-16 | 8-21, 112-125 | 22-35, 126-139 |
| eth2 | 0 | 17-32 | 8-21, 112-125 | 22-35, 126-139 |
| eth3 | 1 | 33-48 ⚠️ | 60-73, 164-177 | 74-87, 178-191 |
| eth4 | 1 | 49-64 ⚠️ | 60-73, 164-177 | 74-87, 178-191 |

⚠️ **NUMA mismatch**: eth3/eth4 IRQs on NUMA 0, threads on NUMA 1 (but still works!)

---

## NIC Activity During Test

| NIC | RX Devmem Packets | RX Bytes | Status |
|-----|-------------------|----------|--------|
| eth1 | +15,622,248 | +128.93 GB | ✅ Active |
| eth2 | +15,611,897 | +128.90 GB | ✅ Active |
| eth3 | +15,608,838 | +128.89 GB | ✅ Active |
| eth4 | +15,608,838 | +128.89 GB | ✅ Active |
| **Total** | **+62,451,821** | **+515.61 GB** | ✅ Balanced |

**Key**: All 4 NICs used equally, zero dropped packets.

---

## Recommendations

### ❌ Do NOT Do
- IRQ affinity tuning (NCCL doesn't use it)
- IRQ-to-thread alignment (not necessary)
- Multi-NIC with current architecture (known to fail)

### ✅ Do This
1. **Refactor to single-process architecture** (enables multi-NIC)
   - 1 process/node, 8 GPUs/process
   - All 4 NICs visible
   - Expected gain: 2-4x (multi-NIC parallelism)

2. **Add thread CPU affinity** (after refactor)
   - Export `NCCL_GPUDIRECTTCPX_TX/RX_BINDINGS`
   - Expected gain: 10-30%

3. **Implement multi-channel** (after refactor)
   - 4-8 channels/GPU distributed across NICs
   - Expected gain: 1.5-2x

---

## Next Steps

See **SINGLE_PROCESS_PLAN.md** for detailed refactor plan.

**Timeline**: ~5-7 days  
**Target**: >15 GB/s bus bandwidth (within 20% of NCCL)

---

## Raw Data

Full diagnostics data: `/home/daniel/uccl/diagnostics/`

Key files:
- `SUMMARY.txt` - Test summary
- `env_vars_*.txt` - NCCL environment variables
- `irq_affinity_before_*.txt` - IRQ affinity masks
- `nic_stats_after_*.txt` - NIC traffic statistics
- `nccl_test_output.log` - NCCL performance results

**Full analysis**: See `docs/archive/DIAGNOSTICS_ANALYSIS.md` (300+ lines)

