# AI Assistant Handoff Prompt

**Last Updated**: 2025-10-07
**Status**: Step 2 (Control Plane) COMPLETE, Step 3 (Data Plane) ready to start
**Purpose**: Quick context injection for new AI assistants

---

## Context Injection (Copy This)

```
I'm working on a NIXL-TCPX plugin for GCP A3-high instances (2 nodes, 8× H100 GPUs, 4× gVNIC per node).
The project uses Google's nccl-plugin-gpudirecttcpx APIs for GPU-to-GPU P2P over TCPX (GPUDirect over TCP).

CURRENT STATUS (2025-10-07):
- ✅ Step 2.5: Devmem validation COMPLETE (single-process can use 4 channels on same NIC)
- ✅ Step 2: Control plane refactor COMPLETE (orchestrator created, 64 channels working)
- ⏳ Step 3: Data plane upgrade READY TO START (add actual data transfer)
- Target: >15 GB/s bus bandwidth (NCCL baseline: 19.176 GB/s)

CRITICAL PROGRESS:
1. ✅ Devmem conflicts RESOLVED - single-process architecture works!
2. ✅ Control plane working - 1 process manages all 8 GPUs, 64 channels total
3. ✅ All 4 NICs available to all GPUs (no more devmem conflicts)
4. ✅ Bootstrap strategy implemented (per-GPU ports: 20000-20007)
5. ✅ Bug fixes: duplicate listen, missing error checks, CUDA context leaks, accept retry logic

WORKSPACE: /home/daniel/uccl

KEY DOCS (read in order):
1. p2p/tcpx/STEP2_COMPLETE.md - Step 2 completion status (READ FIRST)
2. p2p/tcpx/BUGFIXES_ORCHESTRATOR.md - Recent bug fixes
3. p2p/tcpx/docs/SINGLE_PROCESS_PLAN.md - Overall refactor plan
4. p2p/tcpx/docs/PROJECT_STATUS.md - Historical context

KEY FILES:
- p2p/tcpx/run_p2p_singleproc.sh - NEW single-process launcher
- p2p/tcpx/tests/test_tcpx_perf_orchestrator.cc - NEW orchestrator (Step 2 complete)
- p2p/tcpx/tests/test_devmem_validation.cc - Devmem validation test
- p2p/tcpx/src/channel_manager.cc - Channel management (recently fixed accept retry)
- p2p/tcpx/run_p2p_fullmesh.sh - OLD multi-process launcher (working baseline)

IMMEDIATE TASK:
Implement Step 3: Data Plane Upgrade (see SINGLE_PROCESS_PLAN.md)
- Add actual data transfer to orchestrator
- Implement round-robin channel selection
- Add sliding window flow control
- Integrate unpack kernel
- Timeline: 2-3 days
- Expected: >10 GB/s bandwidth

START BY: Reading p2p/tcpx/STEP2_COMPLETE.md and logs/singleproc_*.log
```

---

## Quick Start for New AI

### 1. Read Recent Progress
```bash
# Step 2 completion status (READ FIRST)
view p2p/tcpx/STEP2_COMPLETE.md

# Recent bug fixes
view p2p/tcpx/BUGFIXES_ORCHESTRATOR.md

# Overall refactor plan
view p2p/tcpx/docs/SINGLE_PROCESS_PLAN.md

# Check latest test logs
view p2p/tcpx/logs/singleproc_server_*.log
view p2p/tcpx/logs/singleproc_client_*.log
```

### 2. Verify Step 2 Works (Control Plane)
```bash
cd /home/daniel/uccl/p2p/tcpx

# Build orchestrator
make test_tcpx_perf_orchestrator

# Node 0 (Server)
./run_p2p_singleproc.sh server

# Node 1 (Client)
./run_p2p_singleproc.sh client <NODE0_IP>

# Check results
tail -50 logs/singleproc_server_*.log
tail -50 logs/singleproc_client_*.log

# Expected output:
# - All 8 GPUs initialized
# - All 64 channels created (8 GPUs × 8 channels)
# - All channels accepted connections
# - All channels registered memory
# - "=== ALL GPUs READY ===" message
```

### 3. Understand Current Architecture
```bash
# Single-process orchestrator (Step 2 complete)
view p2p/tcpx/tests/test_tcpx_perf_orchestrator.cc

# Channel manager (recently fixed accept retry)
view p2p/tcpx/src/channel_manager.cc

# Compare with old multi-process version
view p2p/tcpx/tests/test_tcpx_perf_multi.cc
```

### 4. Start Step 3 (Data Plane)
```bash
cd /home/daniel/uccl/p2p/tcpx

# Read Step 3 requirements in plan
view docs/SINGLE_PROCESS_PLAN.md  # Look for "Step 3: Data Plane Upgrade"

# Study existing data transfer logic
view tests/test_tcpx_perf_multi.cc  # Lines 600-800 (send/recv loops)

# Plan modifications to orchestrator
# TODO: Add per-GPU worker threads
# TODO: Implement round-robin channel selection
# TODO: Add sliding window flow control
# TODO: Integrate unpack kernel
```



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

