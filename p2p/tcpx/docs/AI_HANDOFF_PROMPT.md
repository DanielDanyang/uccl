# AI Assistant Handoff Prompt

**Last Updated**: 2025-10-07  
**Status**: IRQ investigation complete, single-process refactor planned  
**Purpose**: Quick context injection for new AI assistants

---

## Context Injection (Copy This)

```
I'm working on a NIXL-TCPX plugin for GCP A3-high instances (2 nodes, 8× H100 GPUs, 4× gVNIC per node). 
The project uses Google's nccl-plugin-gpudirecttcpx APIs for GPU-to-GPU P2P over TCPX (GPUDirect over TCP).

CURRENT STATUS (2025-10-07):
- Single-NIC P2P: WORKING (2.75 GB/s server, 1.17 GB/s client per GPU)
- NCCL Reference: WORKING (19.176 GB/s bus bandwidth)
- IRQ Investigation: COMPLETE - IRQ affinity NOT the bottleneck
- Next: Single-process architecture refactor to enable multi-NIC

KEY FINDINGS:
1. ✅ IRQ affinity is NOT the bottleneck (NCCL uses default IRQ distribution)
2. ✅ Thread CPU affinity IS important (NCCL pins to NUMA-local cores)
3. ✅ Multi-NIC parallelism IS critical (NCCL uses 4 NICs, P2P uses 1)
4. ✅ Process architecture matters (NCCL: 1 proc/8 GPUs, P2P: 8 procs/1 GPU)
5. ❌ Multi-NIC per GPU fails in current architecture (devmem conflicts)

ROOT CAUSE: Performance gap (~7x) due to process architecture, not IRQ affinity.
- NCCL: 1 process/node → can share NICs across GPUs → multi-NIC per GPU works
- P2P: 8 processes/node → cannot share NICs → devmem conflicts

WORKSPACE: /home/daniel/uccl

KEY DOCS (read in order):
1. p2p/tcpx/docs/PROJECT_STATUS.md - Current status and timeline
2. p2p/tcpx/docs/DIAGNOSTICS_SUMMARY.md - IRQ investigation results
3. p2p/tcpx/docs/SINGLE_PROCESS_PLAN.md - Refactor plan
4. p2p/tcpx/README.md - Quick start

KEY FILES:
- p2p/tcpx/run_p2p_fullmesh.sh - Current P2P launcher (8 processes)
- p2p/tcpx/tests/test_tcpx_perf_multi.cc - P2P benchmark program
- p2p/tcpx/src/channel_manager.cc - Channel management
- collective/rdma/run_nccl_test_tcpx.sh - NCCL reference

IMMEDIATE TASK:
Implement single-process architecture refactor (see SINGLE_PROCESS_PLAN.md)
- Goal: Enable multi-NIC per GPU (no devmem conflicts)
- Timeline: ~5-7 days
- Target: >15 GB/s bus bandwidth

START BY: Reading p2p/tcpx/docs/PROJECT_STATUS.md and SINGLE_PROCESS_PLAN.md
```

---

## Quick Start for New AI

### 1. Read Core Documentation
```bash
# Project status
view p2p/tcpx/docs/PROJECT_STATUS.md

# IRQ investigation results
view p2p/tcpx/docs/DIAGNOSTICS_SUMMARY.md

# Refactor plan
view p2p/tcpx/docs/SINGLE_PROCESS_PLAN.md
```

### 2. Verify Current P2P Works
```bash
cd /home/daniel/uccl/p2p/tcpx

# Node 0 (Server)
./run_p2p_fullmesh.sh server

# Node 1 (Client)
./run_p2p_fullmesh.sh client <NODE0_IP>

# Check results
grep "PERF.*Avg.*BW:" logs/fullmesh_*.log
# Expected: ~2.75 GB/s (server), ~1.17 GB/s (client)
```

### 3. Review NCCL Reference
```bash
cd /home/daniel/uccl/collective/rdma

# Run NCCL test
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1

# Check results
cat diagnostics/nccl_*/nccl_metrics_summary.txt
# Expected: ~19 GB/s bus bandwidth
```

### 4. Start Refactor (Step 1)
```bash
cd /home/daniel/uccl/p2p/tcpx

# Read the plan
view docs/SINGLE_PROCESS_PLAN.md

# Create prototype launcher
cp run_p2p_fullmesh.sh run_p2p_singleproc.sh
vim run_p2p_singleproc.sh  # Remove per-GPU forking

# Create orchestrator skeleton
cp tests/test_tcpx_perf_multi.cc tests/test_tcpx_perf_orchestrator.cc
vim tests/test_tcpx_perf_orchestrator.cc  # Add multi-threaded worker model
```

---

## Expected AI Workflow

1. **Understand Context** (30 min)
   - Read PROJECT_STATUS.md
   - Read DIAGNOSTICS_SUMMARY.md
   - Read SINGLE_PROCESS_PLAN.md

2. **Verify Environment** (15 min)
   - Run current P2P benchmark
   - Confirm 2.75 GB/s baseline

3. **Plan Implementation** (1 hour)
   - Review SINGLE_PROCESS_PLAN.md steps
   - Ask clarifying questions if needed
   - Confirm approach with user

4. **Implement Step-by-Step** (5-7 days)
   - Follow SINGLE_PROCESS_PLAN.md timeline
   - Test after each step
   - Document progress

5. **Validate and Measure** (2-3 days)
   - Run validation tests (Step 6 in plan)
   - Compare to NCCL baseline
   - Document results

---

## Common Pitfalls to Avoid

1. **Don't tune IRQ affinity** - NCCL doesn't use it, not necessary
2. **Don't try multi-NIC with current architecture** - Known to fail (devmem conflicts)
3. **Don't increase channels on single NIC** - Known to degrade performance
4. **Don't test on loopback** - TCPX devmem requires separate nodes
5. **Don't skip validation steps** - Each step builds on previous

---

## Success Criteria

### Functional
- [ ] Single-process P2P works (1 GPU smoke test)
- [ ] No devmem conflicts (8 GPUs, multi-NIC)
- [ ] Multi-NIC works (4 NICs simultaneously)
- [ ] Multi-channel works (8 channels/GPU)

### Performance
- [ ] Bandwidth >10 GB/s/GPU (4 NICs × multi-channel)
- [ ] Bandwidth >15 GB/s bus BW (target)
- [ ] Within 20% of NCCL (19.176 GB/s)

### Quality
- [ ] Stable and reproducible
- [ ] Thread affinity verified
- [ ] Well-documented
- [ ] Fallback to multi-process works

---

## Key Technical Concepts

**TCPX**: GPUDirect over TCP using devmem-tcp kernel API (zero-copy GPU-to-GPU)

**devmem-tcp**: Kernel API providing cmsg with scattered buffer descriptors for DMA

**Unpack kernel**: CUDA kernel copying scattered devmem buffers to contiguous GPU memory

**Flow steering**: Traffic steering via dp-manager (external UNIX socket service)

**Process architecture**:
- **Single-process** (NCCL): 1 proc/node, all GPUs/NICs visible, enables NIC sharing
- **Multi-process** (P2P): 8 procs/node, 1 GPU each, devmem conflicts when sharing NICs

**NUMA topology**:
- NUMA 0: GPUs 0-3, eth1-2, CPUs 0-51, 104-155
- NUMA 1: GPUs 4-7, eth3-4, CPUs 52-103, 156-207

**Thread affinity**: Pinning threads to NUMA-local cores (important for performance)

**IRQ affinity**: Pinning interrupt handlers to specific CPUs (NOT important - NCCL uses default)

---

## Environment Details

**Hardware**:
- 2 nodes, 8× H100 GPUs per node
- 4× gVNIC per node (eth1-4, 200 Gbps each)
- 208 CPUs per node (104 cores × 2 HT)

**Software**:
- Ubuntu with custom kernel (devmem-tcp support)
- TCPX plugin v3.1.6
- NCCL 2.x with TCPX support
- dp-manager for flow steering

**Network Config**:
- Node IPs: `scripts/node_ips/tcpx.txt`
- Bootstrap port: 20000 (base)
- NICs: eth1-4 (200 Gbps each)

---

## Useful Commands

### Check NIC Status
```bash
# Verify devmem is working
ethtool -S eth1 | grep rx_devmem_pkts  # Should increase during test

# Check NIC info
ip addr show eth1
```

### Check CPU/IRQ
```bash
# CPU topology
lscpu

# IRQ affinity
cat /proc/irq/*/smp_affinity_list | head -20

# CPU usage
mpstat -P ALL 1
```

### Build and Test
```bash
# Build
cd /home/daniel/uccl/p2p/tcpx
make clean && make

# Run test
./run_p2p_fullmesh.sh server  # Node 0
./run_p2p_fullmesh.sh client <NODE0_IP>  # Node 1

# Check logs
grep "PERF.*Avg.*BW:" logs/fullmesh_*.log
```

---

## Documentation Structure

```
p2p/tcpx/docs/
├── PROJECT_STATUS.md          # Current status (READ FIRST)
├── DIAGNOSTICS_SUMMARY.md     # IRQ investigation results
├── SINGLE_PROCESS_PLAN.md     # Refactor plan
├── AI_HANDOFF_PROMPT.md       # This file
└── archive/                   # Historical docs
    ├── DIAGNOSTICS_ANALYSIS.md (detailed version)
    ├── IRQ_BINDING_INVESTIGATION_PLAN.md (obsolete)
    └── ... (other historical docs)
```

---

## Contact and Support

- **Logs**: `p2p/tcpx/logs/fullmesh_*.log`
- **Diagnostics**: `/home/daniel/uccl/diagnostics/`
- **Reference**: NCCL tests in `collective/rdma/`

---

**Last Updated**: 2025-10-07  
**Next Action**: Implement single-process refactor (see SINGLE_PROCESS_PLAN.md)

