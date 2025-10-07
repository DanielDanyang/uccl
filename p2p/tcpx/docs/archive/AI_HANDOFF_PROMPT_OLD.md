# AI Assistant Handoff Prompt

**Last Updated**: 2025-10-07
**Status**: Multi-channel investigation phase, IRQ binding analysis planned
**Purpose**: This prompt allows a new AI assistant (without conversation history) to immediately continue the TCPX P2P performance optimization work.

---

## Context Injection Prompt

Copy and paste this entire section to a new AI assistant:

```
I'm working on a NIXL-TCPX plugin for GCP A3-high instances (2 nodes, 8x H100 GPUs per node, 4x gVNIC per node). The project uses Google's nccl-plugin-gpudirecttcpx APIs to implement GPU-to-GPU P2P communication over TCPX (GPUDirect over TCP with devmem-tcp kernel API).

CURRENT STATUS:
- Single-NIC, single-channel P2P benchmark: WORKING (2.75 GB/s server, 1.17 GB/s client per GPU)
- NCCL AllReduce reference: WORKING (18.7 GB/s bus bandwidth)
- Multi-NIC attempt: FAILED due to devmem resource conflicts between GPU processes
- Next investigation: IRQ binding and CPU affinity optimization

KEY FINDINGS:
1. ✅ Environment fully verified - all 4 NICs (eth1-4) work with TCPX devmem
2. ✅ Single-NIC P2P works reliably on all NICs (eth1, eth2, eth3, eth4)
3. ❌ Multi-NIC per GPU fails: when multiple GPU processes try to use the same NIC, "rx no cmsg" errors occur
4. ❌ Multi-channel on single NIC degrades performance (4 channels on 1 NIC: 0.43 GB/s vs 1 channel: 1.17 GB/s)
5. 🔍 NCCL uses different architecture: 1 process with 8 GPUs, all 4 NICs visible, 8 channels per GPU
6. 🔍 NCCL sets explicit IRQ/CPU bindings via NCCL_GPUDIRECTTCPX_TX/RX_BINDINGS

HYPOTHESIS: Performance gap may be due to:
- CPU/IRQ affinity differences (NCCL pins threads to specific cores)
- Process architecture (NCCL: 1 process/8 GPUs vs P2P: 8 processes/1 GPU each)
- Channel distribution strategy (NCCL distributes 8 channels across 4 NICs)

WORKSPACE: /home/daniel/uccl
KEY FILES:
- IRQ investigation plan: p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md (READ THIS FIRST)
- Current P2P benchmark: p2p/tcpx/run_p2p_fullmesh.sh
- Multi-channel test: p2p/tcpx/tests/test_tcpx_perf_multi.cc
- Channel manager: p2p/tcpx/src/channel_manager.cc
- NCCL reference (with diagnostics): collective/rdma/run_nccl_test_tcpx.sh
- Previous debug reports: p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md (historical context)

IMMEDIATE TASK:
1. Run NCCL diagnostics collection to understand IRQ/CPU binding behavior
2. Analyze the data to see if IRQ affinity explains the performance gap
3. Decide whether to implement NCCL-style bindings in P2P benchmark

START BY: Reading p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md for the systematic investigation plan.
```

---

## Quick Start Commands for New AI

After injecting the context above, the AI can immediately run:

### 1. Read the investigation plan
```
view p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md
```

### 2. Run NCCL diagnostics collection
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1  # Last param=1 enables diagnostics
```

### 3. Check diagnostics results
```bash
# Find latest diagnostics directory
ls -lt /home/daniel/uccl/diagnostics/ | head -5

# View summary
cat /home/daniel/uccl/diagnostics/nccl_*/SUMMARY.txt

# View key metrics
cat /home/daniel/uccl/diagnostics/nccl_*/nccl_metrics_summary.txt
```

### 4. Examine current P2P implementation
```
view p2p/tcpx/run_p2p_fullmesh.sh
view p2p/tcpx/tests/test_tcpx_perf_multi.cc
view p2p/tcpx/src/channel_manager.cc
```

### 5. Check recent P2P logs
```bash
ls -lt p2p/tcpx/logs/fullmesh_* | head -10
view p2p/tcpx/logs/fullmesh_server_gpu0_<latest>.log
```

### 6. Review historical context (if needed)
```
view p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md
```

---

## Expected AI Workflow

### Phase 1: Data Collection (Step 1 of investigation plan)
1. **Read IRQ_BINDING_INVESTIGATION_PLAN.md** - Understand the systematic approach
2. **Run NCCL diagnostics** - Collect IRQ, CPU, NUMA data during NCCL test
3. **Analyze diagnostics** - Identify NCCL's actual CPU/IRQ binding behavior
4. **Document findings** - Fill in the tables from Step 2 of the plan

### Phase 2: Analysis (Steps 2-3)
1. **Compare NCCL vs P2P** - CPU usage patterns, IRQ distribution
2. **Design binding policy** - Match NCCL's approach or adapt it
3. **Implement bindings** - Modify run_p2p_fullmesh.sh with IRQ affinity setup

### Phase 3: Testing (Steps 4-5)
1. **Run test matrix** - Baseline vs various binding configurations
2. **Measure impact** - Bandwidth, CPU usage, IRQ distribution
3. **Iterate** - Refine based on results

### Phase 4: Decision (Step 6)
1. **Document results** - Create IRQ_BINDING_RESULTS.md
2. **Decide next steps** - Continue optimization or pivot to other approaches

---

## Key Investigation Principles

- **NCCL is the reference**: It achieves 18.7 GB/s bus BW, we need to understand why
- **Systematic approach**: Follow the 6-step plan, don't skip data collection
- **Measure everything**: CPU usage, IRQ counts, bandwidth - before and after
- **ethtool rx_devmem_pkts is ground truth**: Confirms devmem path is active
- **One variable at a time**: Test IRQ binding separately from other changes

---

## Common Pitfalls to Avoid

❌ **Don't** skip the diagnostics collection step (need data to make decisions)
❌ **Don't** assume IRQ binding is the only factor (may be architecture difference)
❌ **Don't** try multi-NIC again without solving devmem conflict (known issue)
❌ **Don't** ignore the process architecture difference (1 proc/8 GPUs vs 8 proc/1 GPU)

✅ **Do** collect comprehensive diagnostics first (IRQ, CPU, NUMA)
✅ **Do** compare NCCL's actual behavior vs documentation
✅ **Do** test incrementally (baseline → IRQ affinity → env vars → combined)
✅ **Do** document negative results (important to know what doesn't work)

---

## Success Criteria

### Current Phase (IRQ Binding Investigation)
1. ✅ Diagnostics collected for NCCL test (IRQ, CPU, NUMA data)
2. ⏳ NCCL's actual CPU/IRQ binding behavior documented
3. ⏳ Binding policy designed for P2P benchmark
4. ⏳ Bindings implemented and tested
5. ⏳ Performance impact measured (target: >5 GB/s per GPU, 2x improvement)

### Overall Project Success
1. P2P benchmark achieves competitive performance (>10 GB/s per GPU)
2. Understand root cause of performance gap vs NCCL
3. Clear documentation of what works and what doesn't
4. Decision made on next steps (continue optimization vs alternative approaches)

---

## Additional Context

### Project Background
- **Goal**: Implement NIXL-TCPX plugin using Google's nccl-plugin-gpudirecttcpx APIs
- **Partners**: Anyscale, Character AI
- **Environment**: GCP A3-high (H100), TCPX-only (no RDMA)
- **Constraint**: Treat net_tcpx.h as black box, use send/recv APIs directly

### Technical Details
- **GPUDirect TCPX**: Zero-copy GPU-to-GPU over TCP via devmem-tcp kernel API
- **devmem-tcp**: Kernel provides cmsg with scattered buffer descriptors for DMA
- **Unpack kernel**: CUDA kernel copies scattered devmem buffers to contiguous GPU memory
- **Flow steering**: UNIX socket-based traffic steering via dp-manager (external service)

### Network Topology
- **Node 0**: eth0 (ctrl), eth1/eth2 (NUMA0/GPU0-3), eth3/eth4 (NUMA1/GPU4-7)
- **Node 1**: Same layout
- **IPs**: See scripts/node_ips/tcpx.txt

---

## Reproduction Commands

### Current Working P2P Benchmark (Single-NIC)
```bash
# Terminal 1 (Node 0 - Server):
cd /home/daniel/uccl/p2p/tcpx
./run_p2p_fullmesh.sh server

# Terminal 2 (Node 1 - Client):
./run_p2p_fullmesh.sh client <NODE0_ETH0_IP>

# Expected: ~2.75 GB/s server, ~1.17 GB/s client per GPU
# Logs in: p2p/tcpx/logs/fullmesh_*.log
```

### NCCL Reference Test
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 0  # Without diagnostics
# Expected: ~18.7 GB/s bus bandwidth
```

### NCCL with Diagnostics Collection
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1  # Last param=1 enables diagnostics
# Results in: diagnostics/nccl_<timestamp>/
```

### Monitor NIC Activity
```bash
# Before test
ethtool -S eth1 | grep rx_devmem_pkts
ethtool -S eth2 | grep rx_devmem_pkts

# During test (in another terminal)
watch -n 1 'ethtool -S eth1 | grep rx_devmem_pkts; ethtool -S eth2 | grep rx_devmem_pkts'

# After test - verify counters increased
```

---

## File Structure Reference

```
/home/daniel/uccl/
├── p2p/tcpx/
│   ├── run_p2p_fullmesh.sh           # Current P2P benchmark launcher (8 processes)
│   ├── tests/
│   │   ├── test_tcpx_perf_multi.cc   # Multi-channel P2P test program
│   │   └── test_tcpx_perf.cc         # Original single-channel test (legacy)
│   ├── src/
│   │   ├── channel_manager.cc        # Manages TCPX channels and NIC selection
│   │   └── tcpx_wrapper.cc           # TCPX plugin wrapper
│   ├── include/
│   │   ├── channel_manager.h         # Channel manager API
│   │   └── tcpx_wrapper.h            # TCPX wrapper API
│   ├── logs/                         # Test logs
│   │   ├── fullmesh_server_gpu*.log  # Server logs per GPU
│   │   ├── fullmesh_client_gpu*.log  # Client logs per GPU
│   │   └── (older logs from previous tests)
│   └── docs/
│       ├── IRQ_BINDING_INVESTIGATION_PLAN.md  # Current investigation plan (READ FIRST)
│       ├── AI_HANDOFF_PROMPT.md               # This file
│       └── DEBUG_ETH2_RX_NO_CMSG.md           # Historical debug report (resolved)
├── collective/rdma/
│   └── run_nccl_test_tcpx.sh         # NCCL reference with diagnostics collection
├── diagnostics/                      # Diagnostics output directory
│   └── nccl_<timestamp>/             # Per-run diagnostics data
└── scripts/node_ips/
    └── tcpx.txt                      # Network configuration (2 nodes)
```

---

## Environment Variables Reference

### TCPX Configuration (from run_p2p_fullmesh.sh and run_nccl_test_tcpx.sh)
```bash
# NIC selection
NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4  # Data NICs
NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0                      # Control NIC

# Flow steering
NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=/run/tcpx
NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000

# CPU/IRQ bindings (NCCL uses these, P2P currently doesn't)
NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# Performance tuning
NCCL_NSOCKS_PERTHREAD=4
NCCL_SOCKET_NTHREADS=1
NCCL_DYNAMIC_CHUNK_SIZE=524288
NCCL_MIN_ZCOPY_SIZE=4096
```

### P2P Benchmark Configuration
```bash
# Channel and window settings
UCCL_TCPX_NUM_CHANNELS=1           # Number of channels per GPU (currently 1)
UCCL_TCPX_WINDOW_SIZE=16           # Outstanding requests per channel
UCCL_TCPX_CHUNK_BYTES=524288       # Chunk size (512 KB)
UCCL_TCPX_PERF_SIZE=67108864       # Total transfer size (64 MB)

# Bootstrap configuration
UCCL_TCPX_BOOTSTRAP_PORT=20000     # Base port for bootstrap
```

### Debug Logging
```bash
NCCL_DEBUG=INFO                    # INFO, WARN, TRACE
NCCL_DEBUG_SUBSYS=ENV,NET,INIT     # Subsystems to log
```

---

## Last Known State

- **Date**: 2025-10-07
- **Status**: Single-NIC P2P working, multi-NIC failed, IRQ binding investigation planned
- **Current Performance**: 2.75 GB/s (server) / 1.17 GB/s (client) per GPU
- **Target Performance**: >10 GB/s per GPU (approaching NCCL's 18.7 GB/s bus BW)
- **Next Step**: Run NCCL diagnostics collection, analyze IRQ/CPU binding data
- **Blocker**: Need to understand if IRQ affinity explains performance gap

---

## Questions to Guide Investigation

### IRQ Binding Questions (Current Focus)
1. What CPU cores does NCCL actually use for TX/RX threads? (check diagnostics)
2. Which CPUs handle gVNIC IRQs during NCCL test? (check /proc/interrupts delta)
3. Does NCCL's binding configuration match the actual CPU usage? (compare env vars vs reality)
4. Can we replicate NCCL's binding in P2P benchmark? (implement and test)
5. What performance improvement (if any) does IRQ binding provide? (measure)

### Architecture Questions (Secondary)
1. Is the 1-process-8-GPUs vs 8-processes-1-GPU architecture difference significant?
2. Can we safely use multiple NICs per GPU without devmem conflicts? (previous attempt failed)
3. Should we increase channel count after fixing IRQ binding? (previous attempt degraded performance)

### Multi-NIC Questions (Known Issue, Deferred)
1. Why do multiple GPU processes conflict when using the same NIC? (devmem registration issue)
2. Is there a way to share NIC devmem resources across processes? (may require plugin changes)
3. Should we switch to NCCL's 1-process architecture to enable multi-NIC? (major refactor)

---

## Expected Time to Resolution

### IRQ Binding Investigation (Current Phase)
- **Data collection**: 1-2 hours (run diagnostics, analyze results)
- **Implementation**: 2-3 hours (add IRQ affinity setup to P2P benchmark)
- **Testing**: 2-3 hours (run test matrix, measure impact)
- **Total estimate**: 5-8 hours

### If IRQ Binding Helps
- **Optimization**: +2-4 hours (tune core ranges, test variations)
- **Documentation**: +1 hour (write results report)

### If IRQ Binding Doesn't Help
- **Pivot decision**: +1-2 hours (analyze why, decide next approach)
- **Alternative investigation**: TBD (may need to revisit architecture)

---

## Progress Checklist

### Phase 1: Data Collection ⏳
- [ ] NCCL diagnostics collected (IRQ, CPU, NUMA)
- [ ] IRQ-to-NIC-to-CPU mapping table created
- [ ] NCCL's actual CPU usage documented
- [ ] P2P baseline metrics recorded

### Phase 2: Implementation ⏳
- [ ] IRQ affinity setup script created
- [ ] run_p2p_fullmesh.sh modified with bindings
- [ ] Binding environment variables exported
- [ ] Pre-flight verification added

### Phase 3: Testing ⏳
- [ ] Test matrix executed (5 configurations)
- [ ] Performance metrics collected
- [ ] CPU/IRQ usage verified
- [ ] Results compared to baseline

### Phase 4: Decision ⏳
- [ ] Results documented in IRQ_BINDING_RESULTS.md
- [ ] Performance improvement quantified (or lack thereof)
- [ ] Next steps decided
- [ ] Handoff documentation updated

---

## Final Notes

### Current Understanding
- ✅ Single-NIC P2P works reliably (all 4 NICs tested)
- ✅ Environment fully verified (NCCL proves it)
- ❌ Multi-NIC per GPU fails (devmem conflict)
- ❌ Multi-channel on single NIC degrades performance
- 🔍 IRQ/CPU binding may explain NCCL's superior performance

### Investigation Strategy
Follow the **systematic 6-step plan** in IRQ_BINDING_INVESTIGATION_PLAN.md:
1. Gather data (don't skip this!)
2. Design policy (based on data, not assumptions)
3. Implement (incrementally)
4. Test (measure everything)
5. Iterate (refine based on results)
6. Document (success or failure)

### Key Insight
NCCL's performance advantage may come from:
- **CPU affinity**: Pinning threads to NUMA-local cores
- **IRQ affinity**: Directing interrupts to specific cores
- **Architecture**: 1 process managing all GPUs and NICs
- **Algorithm**: Ring/Tree collective patterns vs simple P2P

We're investigating the first two factors. If they don't explain the gap, we may need to reconsider the architecture.

Good luck! 🚀

