# TCPX P2P Benchmark - Project Status

**Last Updated**: 2025-10-07  
**Project**: NIXL-TCPX Plugin Development  
**Partners**: Anyscale, Character AI  
**Environment**: GCP A3-high (2 nodes, 8x H100 GPUs, 4x gVNIC per node)

---

## 🎯 Project Goal

Implement a high-performance NIXL-TCPX plugin using Google's nccl-plugin-gpudirecttcpx APIs for GPU-to-GPU communication over TCPX (GPUDirect over TCP with devmem-tcp kernel API).

---

## 📊 Current Status

### ✅ What's Working

1. **Single-NIC P2P Benchmark**
   - All 4 NICs (eth1-4) work correctly with TCPX devmem
   - Performance: ~2.75 GB/s (server), ~1.17 GB/s (client) per GPU
   - Verified with ethtool rx_devmem_pkts counters
   - Stable and reproducible

2. **NCCL Reference**
   - AllReduce achieves ~18.7 GB/s bus bandwidth
   - Uses all 4 NICs simultaneously
   - 8 channels per GPU
   - Proves environment is fully functional

3. **Infrastructure**
   - TCPX plugin (v3.1.6) loaded and working
   - Flow steering (dp-manager) operational
   - Bootstrap and connection setup working
   - Multi-GPU (8 GPUs per node) working

### ❌ Known Issues

1. **Multi-NIC per GPU Process**
   - When multiple GPU processes try to use the same NIC, devmem conflicts occur
   - Symptom: "rx no cmsg" errors on the second channel
   - Root cause: devmem registration appears to be process-exclusive per NIC
   - Status: Deferred (requires architecture change or plugin modification)

2. **Multi-Channel Performance Degradation**
   - 4 channels on 1 NIC: 0.43 GB/s (worse than 1 channel: 1.17 GB/s)
   - Root cause: Channels compete for same NIC bandwidth without parallelism
   - Status: Expected behavior, not a bug

3. **Performance Gap vs NCCL**
   - P2P: ~2.75 GB/s per GPU
   - NCCL: ~18.7 GB/s bus bandwidth (effective per-GPU much higher)
   - Gap: ~7x difference
   - Status: Under investigation (IRQ binding hypothesis)

### ✅ Investigation Complete: IRQ Binding Analysis

1. **IRQ Affinity Investigation** (COMPLETED 2025-10-07)
   - ❌ **IRQ affinity is NOT the bottleneck** - NCCL uses default IRQ distribution
   - ✅ **Thread CPU affinity IS important** - NCCL pins threads to NUMA-local cores
   - ✅ **Multi-NIC parallelism IS critical** - NCCL uses all 4 NICs simultaneously
   - ✅ **Process architecture matters** - Single process enables NIC sharing
   - Status: Analysis complete, action plan ready
   - See: `DIAGNOSTICS_ANALYSIS.md` and `NEXT_STEPS_ACTION_PLAN.md`

---

## 📁 Key Files and Documentation

### Documentation
- **DIAGNOSTICS_ANALYSIS.md** - Complete IRQ/CPU/NIC analysis from NCCL test (READ THIS!)
- **NEXT_STEPS_ACTION_PLAN.md** - Recommended optimization path (3 phases)
- **AI_HANDOFF_PROMPT.md** - Handoff guide for new developers/AI assistants
- **IRQ_BINDING_INVESTIGATION_PLAN.md** - Original investigation plan (now obsolete)
- **DEBUG_ETH2_RX_NO_CMSG.md** - Historical debug report (resolved: was loopback issue)
- **PROJECT_STATUS.md** - This file

### Code
- **run_p2p_fullmesh.sh** - Main P2P benchmark launcher (8 processes, 1 GPU each)
- **tests/test_tcpx_perf_multi.cc** - Multi-channel P2P test program
- **src/channel_manager.cc** - TCPX channel management and NIC selection
- **src/tcpx_wrapper.cc** - TCPX plugin wrapper

### Reference
- **collective/rdma/run_nccl_test_tcpx.sh** - NCCL reference with diagnostics collection

### Logs
- **logs/fullmesh_*.log** - P2P benchmark logs (per GPU, per run)
- **diagnostics/nccl_*/** - NCCL diagnostics data (IRQ, CPU, NUMA)

---

## 🚀 Quick Start

### Run Current Working P2P Benchmark
```bash
# Node 0 (Server)
cd /home/daniel/uccl/p2p/tcpx
./run_p2p_fullmesh.sh server

# Node 1 (Client)
./run_p2p_fullmesh.sh client <NODE0_ETH0_IP>

# Expected: ~2.75 GB/s server, ~1.17 GB/s client per GPU
```

### Run NCCL Reference Test
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 0

# Expected: ~18.7 GB/s bus bandwidth
```

### Collect NCCL Diagnostics (for IRQ investigation)
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1  # Last param=1 enables diagnostics

# Results in: diagnostics/nccl_<timestamp>/
```

---

## 🔬 Investigation History

### Phase 1: Initial Development (Completed)
- ✅ TCPX plugin integration
- ✅ Basic P2P communication working
- ✅ Single-GPU tests passing

### Phase 2: Multi-NIC Debugging (Completed)
- ✅ Identified "rx no cmsg" issue
- ✅ Verified environment with NCCL
- ✅ Discovered root cause: loopback testing
- ✅ Fixed by running on separate nodes
- ✅ All 4 NICs verified working

### Phase 3: Multi-Channel Attempt (Completed, Failed)
- ✅ Implemented multi-channel support
- ✅ Attempted 2 NICs per GPU (eth1+eth2 for GPU 0-3)
- ❌ Failed: devmem conflicts when multiple processes use same NIC
- ❌ Failed: 4 channels on 1 NIC degraded performance
- 📝 Learned: Current architecture (8 processes) limits multi-NIC usage

### Phase 4: IRQ Binding Investigation (COMPLETED 2025-10-07)
- ✅ Diagnostics collected from NCCL test
- ✅ IRQ, CPU, NUMA, and NIC data analyzed
- ✅ Key finding: IRQ affinity is NOT the bottleneck
- ✅ Thread CPU affinity IS important (NCCL uses NUMA-local cores)
- ✅ Multi-NIC parallelism IS critical (NCCL uses all 4 NICs)
- ✅ Action plan created for optimization

### Phase 5: Optimization (Current)
- ⏳ Status: Ready to implement thread CPU affinity
- ⏳ Next: Phase 1 - Quick wins (CPU affinity, window/chunk tuning)
- ⏳ Then: Phase 2 - Single-process architecture evaluation
- ⏳ Goal: Approach NCCL performance (>15 GB/s bus BW)

---

## 📈 Performance Summary

| Configuration | Server BW | Client BW | Total BW | Notes |
|---------------|-----------|-----------|----------|-------|
| P2P (1 NIC, 1 ch) | 2.75 GB/s/GPU | 1.17 GB/s/GPU | ~22 GB/s | Current working config |
| P2P (1 NIC, 4 ch) | 2.59 GB/s/GPU | 0.43 GB/s/GPU | ~17 GB/s | Degraded performance |
| P2P (2 NIC, 2 ch) | N/A | N/A | Failed | Devmem conflict |
| NCCL AllReduce | N/A | N/A | 18.7 GB/s bus BW | Reference (different metric) |

**Note**: NCCL's "bus bandwidth" is not directly comparable to P2P's unidirectional bandwidth. NCCL's effective per-GPU throughput is much higher due to Ring algorithm and bidirectional communication.

---

## 🎯 Next Steps

### Immediate: Phase 1 - Quick Wins (1-2 days)
1. **Add thread CPU affinity** (~2-3 hours)
   - Export `NCCL_GPUDIRECTTCPX_TX/RX_BINDINGS` in run_p2p_fullmesh.sh
   - Match NCCL's NUMA-local core allocation
   - Expected gain: 10-30% bandwidth improvement

2. **Tune window size** (~1 hour)
   - Test `UCCL_TCPX_WINDOW_SIZE` = 16, 32, 64, 128
   - Find optimal value
   - Expected gain: 5-15% bandwidth improvement

3. **Tune chunk size** (~1 hour)
   - Test `UCCL_TCPX_CHUNK_BYTES` = 256KB, 512KB, 1MB, 2MB
   - Find optimal value
   - Expected gain: 5-10% bandwidth improvement

**Target**: >3.5 GB/s (server) or >1.5 GB/s (client) per GPU

### Medium-term: Phase 2 - Architecture Evaluation (2-3 days)
1. **Prototype single-process P2P** (~1 day)
   - Create test program with 1 process, 8 GPUs
   - Test multi-NIC capability (check for devmem conflicts)
   - Measure performance

2. **Compare architectures** (~4 hours)
   - Evaluate single-process vs multi-process
   - Decide on path forward

**Decision point**: If single-process enables multi-NIC, proceed to Phase 3

### Long-term: Phase 3 - Full Optimization (3-5 days)
1. **Implement multi-NIC P2P** (~2 days)
   - Modify ChannelManager for multiple NICs per GPU
   - Test with 2, 3, 4 NICs
   - Expected gain: 2-4x bandwidth improvement

2. **Implement multi-channel** (~1 day)
   - Increase channels to 4-8 per GPU
   - Distribute across NICs
   - Expected gain: 1.5-2x bandwidth improvement

3. **Final tuning** (~1 day)
   - Optimize all parameters
   - Document winning configuration

**Target**: >15 GB/s bus bandwidth (within 20% of NCCL)

---

## 🤝 Handoff Information

### For New Developers
1. Read **DIAGNOSTICS_ANALYSIS.md** for IRQ/CPU investigation results
2. Read **NEXT_STEPS_ACTION_PLAN.md** for optimization roadmap
3. Read **AI_HANDOFF_PROMPT.md** for complete context
4. Review recent logs in `p2p/tcpx/logs/`

### For AI Assistants
Use the context injection prompt in **AI_HANDOFF_PROMPT.md** to get up to speed immediately.

### Key Contacts
- **Project Partners**: Anyscale, Character AI
- **Environment**: GCP A3-high instances
- **Workspace**: `/home/daniel/uccl`

---

## 📚 Technical Background

### TCPX GPUDirect
- Zero-copy GPU-to-GPU communication over TCP
- Uses devmem-tcp kernel API for DMA
- Requires cmsg (control messages) with buffer descriptors
- Flow steering via dp-manager (external service)

### Hardware Topology
- 2 nodes, 8 H100 GPUs per node
- 4 gVNIC interfaces per node (100 Gbps each)
- NUMA 0: GPU 0-3, eth1-2
- NUMA 1: GPU 4-7, eth3-4
- 208 CPU cores per node (104 physical × 2 HT)

### NCCL Configuration
- 1 process per node, 8 GPUs per process
- All 4 NICs visible to each process
- 8 channels per GPU
- Explicit TX/RX CPU bindings to NUMA-local cores
- Ring/Tree algorithms for collective operations

### P2P Configuration
- 8 processes per node, 1 GPU per process
- 1 NIC per process (NUMA-matched)
- 1 channel per GPU (currently)
- No explicit CPU bindings (yet)
- Simple point-to-point transfers

---

## 🔑 Key Learnings

1. **Always test on separate nodes** - Loopback doesn't work with TCPX devmem
2. **ethtool rx_devmem_pkts is ground truth** - Authoritative indicator of devmem activity
3. **NCCL proves environment works** - Don't waste time on system-level debugging
4. **Process architecture matters** - Multi-process limits NIC sharing due to devmem conflicts
5. **IRQ affinity is NOT critical** - NCCL uses default IRQ distribution and still achieves high performance
6. **Thread CPU affinity IS important** - NCCL pins threads to NUMA-local cores
7. **Multi-NIC parallelism is key** - NCCL's 4 NICs provide 4x parallelism
8. **Measure everything** - CPU, IRQ, bandwidth - before making changes
9. **Document negative results** - Knowing what doesn't work is valuable (e.g., IRQ affinity)

---

## 📞 Support

For questions or issues:
1. Check documentation in `p2p/tcpx/docs/`
2. Review recent logs in `p2p/tcpx/logs/`
3. Compare with NCCL reference in `collective/rdma/`
4. Use AI_HANDOFF_PROMPT.md for AI assistance

---

**Status**: Active development, optimization phase (Phase 1 ready to start)
**Last Test**: 2025-10-07, single-NIC P2P working at 2.75 GB/s
**Investigation**: IRQ binding analysis complete (IRQ affinity NOT the bottleneck)
**Next Milestone**: Implement thread CPU affinity, evaluate single-process architecture

