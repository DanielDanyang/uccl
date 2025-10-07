# NCCL Diagnostics Analysis - IRQ Binding Investigation

**Date**: 2025-10-07  
**Test**: NCCL AllReduce with TCPX (2 nodes, 8 GPUs per node)  
**Performance**: 19.176 GB/s average bus bandwidth  
**Hostname**: k-dd4e3d8ed5be80000 (Node 0)

---

## Executive Summary

✅ **Diagnostics collection successful** - Comprehensive IRQ, CPU, and NIC data captured during NCCL test  
⚠️ **Key finding**: NCCL does NOT set IRQ affinity - all gVNIC IRQs use default round-robin distribution  
✅ **NCCL uses thread CPU bindings** - TX/RX threads pinned to NUMA-local cores via environment variables  
📊 **All 4 NICs actively used** - ~128 GB transferred per NIC during test (~390 GB total)

**Conclusion**: The performance gap between P2P (2.75 GB/s) and NCCL (19.176 GB/s bus BW) is **NOT** primarily due to IRQ affinity. NCCL's advantage comes from:
1. **Thread CPU affinity** (TX/RX threads pinned to NUMA-local cores)
2. **Process architecture** (1 process with all GPUs/NICs vs 8 separate processes)
3. **Algorithm efficiency** (Ring algorithm with bidirectional communication)
4. **Multi-NIC parallelism** (4 NICs × 8 channels = 32 parallel paths)

---

## 1. Hardware Topology Confirmed

### CPU Configuration
- **Total CPUs**: 208 (104 physical cores × 2 hyperthreads)
- **Sockets**: 2
- **Cores per socket**: 52
- **Threads per core**: 2
- **NUMA nodes**: 2
  - **NUMA 0**: CPUs 0-51, 104-155
  - **NUMA 1**: CPUs 52-103, 156-207

### NIC-to-NUMA Mapping
| NIC  | PCI Address   | NUMA Node | Status |
|------|---------------|-----------|--------|
| eth1 | 0000:06:00.0  | 0         | ✅ Active |
| eth2 | 0000:0c:00.0  | 0         | ✅ Active |
| eth3 | 0000:86:00.0  | 1         | ✅ Active |
| eth4 | 0000:8c:00.0  | 1         | ✅ Active |

---

## 2. NCCL Configuration Analysis

### Environment Variables (from diagnostics)
```bash
# Thread CPU bindings (NCCL respects these)
NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# IRQ bindings (NOT SET - null)
NCCL_GPUDIRECTTCPX_TX_IRQ_BINDINGS=(null)
NCCL_GPUDIRECTTCPX_RX_IRQ_BINDINGS=(null)

# Other key settings
NCCL_NSOCKS_PERTHREAD=4
NCCL_SOCKET_NTHREADS=1
NCCL_DYNAMIC_CHUNK_SIZE=524288
NCCL_MAX_NCHANNELS=16
NCCL_MIN_NCHANNELS=16
NCCL_IGNORE_CPU_AFFINITY=1  # Disables NCCL's default CPU affinity
```

### CPU Binding Pattern
| NIC Group | NUMA | TX Cores | RX Cores | Total Cores |
|-----------|------|----------|----------|-------------|
| eth1, eth2 | 0 | 8-21, 112-125 (28 cores) | 22-35, 126-139 (28 cores) | 56 cores |
| eth3, eth4 | 1 | 60-73, 164-177 (28 cores) | 74-87, 178-191 (28 cores) | 56 cores |

**Key observations**:
- TX and RX cores are **separate** (no overlap)
- All cores are **NUMA-local** to their NICs
- Each NIC group gets **28 TX + 28 RX = 56 cores**
- Cores include both physical cores and hyperthreads (e.g., 8-21 physical, 112-125 HT)

---

## 3. IRQ Affinity Analysis

### Default IRQ Distribution (Before Test)

**Finding**: All gVNIC IRQs use **default round-robin** affinity, NOT custom bindings.

#### eth1 IRQs (Sample - IRQ 79-94, 16 queues)
```
IRQ 79:  CPU 1   (cores=1)
IRQ 80:  CPU 2   (cores=2)
IRQ 81:  CPU 3   (cores=3)
IRQ 82:  CPU 4   (cores=4)
IRQ 83:  CPU 5   (cores=5)
IRQ 84:  CPU 6   (cores=6)
IRQ 85:  CPU 7   (cores=7)
IRQ 86:  CPU 8   (cores=8)
IRQ 87:  CPU 9   (cores=9)
IRQ 88:  CPU 10  (cores=10)
IRQ 89:  CPU 11  (cores=11)
IRQ 90:  CPU 12  (cores=12)
IRQ 91:  CPU 13  (cores=13)
IRQ 92:  CPU 14  (cores=14)
IRQ 93:  CPU 15  (cores=15)
IRQ 94:  CPU 16  (cores=16)
```

**Pattern**: Each IRQ pinned to a **single CPU** in sequential order (1, 2, 3, ..., 16).

#### eth2 IRQs (IRQ 112-127, 16 queues)
```
IRQ 112: CPU 17  (cores=17)
IRQ 113: CPU 18  (cores=18)
...
IRQ 127: CPU 32  (cores=32)
```

#### eth3 IRQs (IRQ 145-160, 16 queues)
```
IRQ 145: CPU 33  (cores=33)
IRQ 146: CPU 34  (cores=34)
...
IRQ 160: CPU 48  (cores=48)
```

#### eth4 IRQs (IRQ 178-193, 16 queues)
```
IRQ 178: CPU 49  (cores=49)
IRQ 179: CPU 50  (cores=50)
...
IRQ 193: CPU 64  (cores=64) [estimated]
```

### IRQ Affinity vs NCCL Bindings Comparison

| NIC  | IRQ CPUs (Actual) | NCCL TX Cores | NCCL RX Cores | Overlap? |
|------|-------------------|---------------|---------------|----------|
| eth1 | 1-16              | 8-21, 112-125 | 22-35, 126-139 | ✅ Partial (8-16 overlap with TX) |
| eth2 | 17-32             | 8-21, 112-125 | 22-35, 126-139 | ✅ Partial (17-21 TX, 22-32 RX) |
| eth3 | 33-48             | 60-73, 164-177 | 74-87, 178-191 | ❌ No overlap (IRQs on NUMA 0, threads on NUMA 1) |
| eth4 | 49-64             | 60-73, 164-177 | 74-87, 178-191 | ✅ Partial (60-64 overlap with TX) |

**Critical finding**: 
- **eth3 IRQs are on NUMA 0 CPUs (33-48)** but **NCCL threads are on NUMA 1 CPUs (60-87, 164-191)**
- This is a **NUMA mismatch** but NCCL still achieves high performance
- Suggests IRQ affinity is **NOT critical** for TCPX performance

---

## 4. NIC Activity During Test

### Traffic Volume (Before → After)

| NIC  | RX Devmem Pkts (Before) | RX Devmem Pkts (After) | Delta (Packets) | RX Bytes (Before) | RX Bytes (After) | Delta (GB) |
|------|-------------------------|------------------------|-----------------|-------------------|------------------|------------|
| eth1 | 32,229,881              | 47,852,129             | 15,622,248      | 265.99 GB         | 394.92 GB        | **128.93 GB** |
| eth2 | 32,209,057              | 47,820,954             | 15,611,897      | 265.95 GB         | 394.85 GB        | **128.90 GB** |
| eth3 | 32,202,822              | 47,811,660             | 15,608,838      | 265.92 GB         | 394.81 GB        | **128.89 GB** |
| eth4 | 32,202,891              | 47,811,729             | 15,608,838      | 265.92 GB         | 394.81 GB        | **128.89 GB** |
| **Total** | | | **62,451,821** | | | **515.61 GB** |

**Key observations**:
- All 4 NICs used **equally** (~128 GB each)
- **Zero dropped packets** (`rx_devmem_dropped: 0` on all NICs)
- Devmem path confirmed active (rx_devmem_pkts increased significantly)
- Balanced load distribution across NICs

---

## 5. Performance Metrics

### NCCL AllReduce Performance
```
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 19.176 GB/s
```

**Peak performance** (1 GB transfer):
- **Algorithm bandwidth**: 41.79 GB/s
- **Bus bandwidth**: 78.35 GB/s

### Comparison to P2P Benchmark

| Metric | P2P (Single-NIC) | NCCL (4-NIC) | Ratio |
|--------|------------------|--------------|-------|
| Bandwidth per GPU | 2.75 GB/s (server) | ~19.176 GB/s bus BW | ~7x |
| NICs used | 1 | 4 | 4x |
| Channels per GPU | 1 | 8 | 8x |
| Process architecture | 8 processes, 1 GPU each | 1 process, 8 GPUs | Different |
| IRQ affinity | Default | Default | Same |
| Thread CPU affinity | None | NUMA-local | **Different** |

---

## 6. Key Findings

### ✅ What NCCL Does
1. **Thread CPU affinity**: Pins TX/RX threads to NUMA-local cores
2. **Multi-NIC usage**: Uses all 4 NICs in parallel
3. **Multi-channel**: 16 channels total (8 per GPU)
4. **Ring algorithm**: Efficient collective communication pattern
5. **Single process**: All GPUs and NICs visible to one process

### ❌ What NCCL Does NOT Do
1. **IRQ affinity**: Does NOT set custom IRQ affinity (uses system default)
2. **IRQ-to-thread alignment**: IRQs and threads are NOT on same cores
3. **NUMA-strict IRQs**: eth3/eth4 IRQs are on NUMA 0, not NUMA 1

### 🔍 Implications for P2P Benchmark

**IRQ affinity is NOT the bottleneck**. The performance gap is due to:

1. **Thread CPU affinity** (P2P doesn't set this)
   - **Action**: Add `NCCL_GPUDIRECTTCPX_TX/RX_BINDINGS` to P2P benchmark
   - **Expected impact**: Moderate (10-30% improvement)

2. **Process architecture** (P2P uses 8 processes, NCCL uses 1)
   - **Issue**: Multi-process limits NIC sharing (devmem conflicts)
   - **Action**: Consider refactoring to single-process architecture
   - **Expected impact**: High (enables multi-NIC per GPU)

3. **Multi-NIC parallelism** (P2P uses 1 NIC, NCCL uses 4)
   - **Issue**: Currently blocked by devmem conflicts
   - **Action**: Solve devmem sharing or change architecture
   - **Expected impact**: Very high (4x theoretical)

4. **Algorithm efficiency** (P2P is simple P2P, NCCL is Ring)
   - **Issue**: Ring algorithm has better bandwidth utilization
   - **Action**: Not applicable to P2P use case
   - **Expected impact**: N/A

---

## 7. Recommendations

### Immediate (Low-hanging fruit)
1. **Add thread CPU affinity to P2P benchmark**
   - Export `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `RX_BINDINGS`
   - Match NCCL's NUMA-local core allocation
   - **Effort**: 1-2 hours
   - **Expected gain**: 10-30% bandwidth improvement

2. **Test with different window sizes**
   - Increase `UCCL_TCPX_WINDOW_SIZE` from 16 to 32 or 64
   - More outstanding requests → better pipeline utilization
   - **Effort**: 30 minutes
   - **Expected gain**: 5-15% bandwidth improvement

### Medium-term (Architecture change)
3. **Investigate single-process architecture**
   - Refactor P2P benchmark to use 1 process with 8 GPUs
   - Enables multi-NIC per GPU (no devmem conflicts)
   - **Effort**: 1-2 days
   - **Expected gain**: 2-4x bandwidth improvement (multi-NIC)

4. **Profile with perf**
   - Identify CPU bottlenecks (syscalls, unpack kernel, etc.)
   - **Effort**: 2-3 hours
   - **Expected gain**: Insights for further optimization

### Long-term (If needed)
5. **Implement multi-channel with single-process**
   - After architecture change, add multi-channel support
   - Distribute channels across NICs
   - **Effort**: 2-3 days
   - **Expected gain**: Approach NCCL performance

---

## 8. Next Steps

### Step 1: Implement Thread CPU Affinity (Recommended)
1. Modify `run_p2p_fullmesh.sh` to export binding environment variables
2. Test with same configuration as baseline (1 NIC, 1 channel)
3. Measure performance improvement
4. **Decision point**: If >20% improvement, continue optimization; if <10%, pivot to architecture change

### Step 2: Test Window Size Tuning
1. Run test matrix with `UCCL_TCPX_WINDOW_SIZE` = 16, 32, 64, 128
2. Measure bandwidth and CPU usage
3. Find optimal window size

### Step 3: Evaluate Architecture Change
1. Prototype single-process P2P benchmark
2. Test multi-NIC capability (no devmem conflicts expected)
3. Measure performance vs current 8-process approach
4. **Decision point**: If multi-NIC works, proceed with full refactor

---

## 9. Conclusion

**IRQ affinity is NOT the performance bottleneck**. NCCL achieves high performance with default IRQ distribution.

The key differences are:
1. ✅ **Thread CPU affinity** (easy to add to P2P)
2. ✅ **Multi-NIC parallelism** (requires architecture change)
3. ✅ **Single-process architecture** (enables NIC sharing)

**Recommended path forward**:
1. Add thread CPU affinity (quick win)
2. Tune window size (quick win)
3. Evaluate single-process architecture (bigger effort, bigger gain)

**Do NOT spend time on**:
- IRQ affinity tuning (NCCL doesn't use it)
- IRQ-to-thread alignment (not necessary)
- NUMA-strict IRQ placement (eth3/eth4 work fine on NUMA 0 IRQs)

---

## Appendix: Raw Data Files

All diagnostics data available in `/home/daniel/uccl/diagnostics/`:
- `SUMMARY.txt` - Test summary
- `irq_snapshot_before_*.txt` - IRQ state before test
- `irq_snapshot_after_*.txt` - IRQ state after test
- `irq_delta_*.txt` - IRQ changes during test
- `irq_affinity_before_*.txt` - IRQ affinity masks
- `cpu_topology_*.txt` - CPU/NUMA topology
- `nic_info_before_*.txt` - NIC configuration
- `nic_stats_after_*.txt` - NIC traffic statistics
- `env_vars_*.txt` - NCCL environment variables
- `nccl_test_output.log` - NCCL performance results

