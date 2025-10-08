# AI Assistant Handoff Prompt (Updated)

Last Updated: 2025-10-08
Status: Step 3 (Data plane) IN PROGRESS — sliding window + continuous progress implemented; awaiting on-hardware verification and Google feedback
Purpose: One-page, current, actionable context for next developer/AI. Legacy content below is outdated; start here.

---

> PIVOT (2025-10-08): Orchestrator path is deprecated for now. Proceed with the multi-process baseline and make each GPU open 4 TCPX connections (per vendor guidance: one channel ≈ one TCPX connection; prefer single 200Gbps NIC + ~8 connections for per-NIC max; GPU should stick to NUMA-local NIC).

### New Working Plan (Multi-Process + Multi-Channel per GPU)
- Target: multi-process benchmark (tests/test_tcpx_perf_multi.cc)
- **CURRENT STATE**: Each GPU process opens 1 channel = 1 TCPX connection (insufficient pipeline)
- **GOAL**: Each GPU process should open 4 channels = 4 TCPX connections
  - Result: 2 GPUs sharing 1 NIC = 2×4 = 8 connections (hits MAX_SOCKETS=8, enables real pipeline)
- **TASK**: Modify test_tcpx_perf_multi.cc to actually USE all 4 channels in parallel (not just create them)
- Measurement: do not expect symmetric send/recv GB/s; focus on scaling with connection count
- Orchestrator docs below are historical; keep for reference only

## Copy-paste this block to brief the next AI (start here)
```
ROLE: You are the next AI developer taking over the NIXL-TCPX P2P benchmark.

CONTEXT:
- Platform: GCP A3-high, 2 nodes, 8× H100 per node, 4× gVNIC (eth1-4, 200Gbps each)
- Current code: tests/test_tcpx_perf_multi.cc is multi-process baseline (one process per GPU, one channel per process)
- Vendor guidance (Google, 2025-10-08):
  * One channel ≈ one TCPX connection
  * Prefer single 200Gbps NIC + ~8 TCPX connections for per-NIC max (~21.26 GB/s ceiling)
  * MAX_SOCKETS=8 per process (TCPX plugin limit)
  * GPU should stick to NUMA-local NIC (GPU0-3→eth1/eth2; GPU4-7→eth3/eth4)
  * vLLM usage: one process per GPU, each process operates a single NIC

PROBLEM (DIAGNOSED):
- Previous misunderstanding: We thought "1 channel = 1 TCP connection"
- **REALITY**: In TCPX, 1 channel (comm) can have MULTIPLE sockets (TCP connections)
- The number of sockets per comm is controlled by: NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS
- **Root cause of slow performance**: Default config creates channels with only 1 socket each
- Example: UCCL_TCPX_NUM_CHANNELS=4 with default settings → 4 channels × 1 socket = 4 connections (not 8!)

SOLUTION (IMPLEMENTED):
- **Strategy A (Recommended)**: Use 1 channel with 8 sockets
  - Set: UCCL_TCPX_NUM_CHANNELS=1, NCCL_NSOCKS_PERTHREAD=8, NCCL_SOCKET_NTHREADS=1
  - Result: 1 channel × 8 sockets = 8 TCP connections per GPU

- **Strategy B (Alternative)**: Use 2 channels with 4 sockets each
  - Set: UCCL_TCPX_NUM_CHANNELS=2, NCCL_NSOCKS_PERTHREAD=4, NCCL_SOCKET_NTHREADS=1
  - Result: 2 channels × 4 sockets = 8 TCP connections per GPU

- **Auto-configuration**: test_tcpx_perf_multi.cc now automatically sets NCCL_NSOCKS_PERTHREAD
  based on UCCL_TCPX_NUM_CHANNELS to target ~8 total sockets

YOUR TASK:
1) READ AND UNDERSTAND the current codebase thoroughly:
   - tests/test_tcpx_perf_multi.cc (main test, has detailed Chinese comments)
   - src/channel_manager.{h,cc} (how channels are created/managed)
   - src/bootstrap.{h,cc} (how handles are exchanged)
   - Understand: how does UCCL_TCPX_NUM_CHANNELS currently work? Where is it read? How are channels created?

2) ANALYZE the gap:
   - Current: ChannelManager creates N channels, but test only uses channel 0 (or round-robins poorly)
   - Current: Each process binds to one GPU and one IFNAME (NIC)
   - Current: Sliding window is per-channel, but we need to actually USE multiple channels in parallel
   - Question: Does the current code already support multi-channel per process, or is it hardcoded to 1?

3) DESIGN the solution (write a plan BEFORE coding):
   - How to make each process actually post sends/recvs across all 4 channels in parallel?
   - How to round-robin chunks across channels (like the orchestrator did)?
   - How to manage 4 separate sliding windows per GPU process?
   - How to ensure progress is driven on all 4 channels (not just channel 0)?
   - How to verify 2 GPUs × 4 channels = 8 total TCPX connections on one NIC?

4) IMPLEMENT incrementally:
   - Start with UCCL_TCPX_NUM_CHANNELS=2, verify 2 connections work
   - Scale to 4, verify 8 connections total (2 GPUs on same NIC)
   - Measure bandwidth improvement vs single-channel baseline

5) VALIDATE:
   - Check logs: "created 4 channels" per GPU process
   - Check TRACE: 8 distinct TCPX connections active on the NIC
   - Check bandwidth: should approach ~21.26 GB/s for single-NIC max (vendor target)
   - Check stability: no deadlocks, windows drain properly, iterations complete

CONSTRAINTS:
- Do not install new system packages without permission
- Keep the multi-process model (one process per GPU)
- Keep NUMA/NIC mapping as-is for now (can hardcode later if needed)
- Preserve the existing progress/window/FIFO semantics (they are correct)

SUCCESS CRITERIA:
- Each GPU process opens 4 TCPX connections (verified in logs/TRACE)
- 2 GPUs sharing 1 NIC = 8 total connections (≤ MAX_SOCKETS=8)
- Bandwidth scales with connection count (4 conns > 1 conn)
- Iterations complete without deadlock; windows drain as expected

KEY FILES TO READ FIRST:
- tests/test_tcpx_perf_multi.cc (current test, ~1100 lines, heavy Chinese comments)
- src/channel_manager.cc (channel creation logic)
- include/channel_manager.h (ChannelManager interface)
- ../HANDOFF_README.md (runbook), ../DEBUG_GUIDE.md (TRACE/troubleshooting)
- ../REPORT_EXEC_SUMMARY_CN.md (中文项目概览)

START BY:
1. Reading tests/test_tcpx_perf_multi.cc completely (understand current flow)
2. Tracing how UCCL_TCPX_NUM_CHANNELS is used (grep for it)
3. Understanding the current channel selection logic (is it round-robin? single-channel?)
4. Writing a detailed plan with code locations and proposed changes
5. Discussing the plan with the user before implementing
```



## Context (TL;DR)
- Project: NIXL-TCPX plugin P2P benchmark on GCP A3-high (2 nodes, 8× H100, 4× gVNIC)
- Interface: Google nccl-plugin-gpudirecttcpx (TCPX / GPUDirect over TCP)
- Architecture: Multi-process baseline (one process per GPU). Orchestrator is deprecated for now.

## Current Status & Problem
- Working path: multi-process test (tests/test_tcpx_perf_multi.cc)
- **PROBLEM**: Current code only uses 1 channel per GPU process = 1 TCPX connection (no real pipeline)
- **GOAL**: Make each GPU process use 4 channels = 4 TCPX connections in parallel
  - This enables: 2 GPUs × 4 channels = 8 connections on one NIC (hits MAX_SOCKETS=8, vendor target)
- Data plane semantics are correct: async + sliding window + continuous progress; FIFO-only tcpx_test; recv calls consumed
- Windows: recv per-comm=16; send per-comm=12
- Debugging: TRACE instructions ready; rc/done/size logs in tests
- External: Guidance from Google applied (prefer ~8 conns per NIC for max throughput ~21.26 GB/s)

## How to Run (current baseline - single channel per GPU)
```bash
# Build (on both nodes)
cd p2p/tcpx
make -j

# Server (Node 0) - currently only uses 1 channel even if you set UCCL_TCPX_NUM_CHANNELS=4
./tests/test_tcpx_perf_multi server 0

# Client (Node 1)
./tests/test_tcpx_perf_multi client <SERVER_IP> 0

# NOTE: Setting UCCL_TCPX_NUM_CHANNELS=4 creates 4 channels but test doesn't USE them in parallel yet
# This is what needs to be fixed!

# Optional diagnostics
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
```

## What Good Looks Like (after fix)
- Logs show: "created 4 channels" per GPU process
- TRACE shows: 8 distinct TCPX connections active when 2 GPUs share 1 NIC
- Server: recvs posted across all 4 channels; windows drain on all channels; Iteration 0 completes
- Client: sends posted across all 4 channels; bandwidth scales with connection count (4 conns > 1 conn)
- Bandwidth approaches ~21.26 GB/s for single-NIC max (vendor target with ~8 connections)

## If It Stalls (after implementing multi-channel)
- Symptom: Only channel 0 shows activity; channels 1-3 idle
  - Action: Verify round-robin logic actually distributes chunks across all channels
  - Action: Check that progress is driven on ALL channels, not just channel 0
- Symptom: Deadlock when using multiple channels
  - Action: Verify each channel has its own sliding window (not shared)
  - Action: Verify opportunistic drain covers all channels
  - Action: Check that FIFO-only testing is per-channel (not global)
- Symptom: Server window stuck at 16/16; tcpx_test logs show rc=0, done=0 repeatedly
  - Action: Confirm progress_channel() is called non-blocking after each post (current + all other channels)
  - Action: Confirm window-full path uses progress_channel(blocking=true)
  - Action: Enable TRACE; check next_transmitting changes and queue shrinking
- Symptom: Only a few "opportunistically released" on client, then nothing
  - Action: Same as above; verify send side also drives progress
- Symptom: done=1 but no release
  - Action: Verify tcpx_irecv_consumed() called after done=1 (recv); memcpy path doesn’t require kernel but must respect done

## Key Files (start here)
- tests/test_tcpx_perf_multi.cc — multi-process benchmark (primary)
- include/sliding_window.h, src/sliding_window.cc — sliding window with 0/1/-1 semantics
- src/channel_manager.{h,cc}, src/bootstrap.{h,cc} — control-plane
- tests/test_tcpx_perf_orchestrator.cc — historical reference only (deprecated)

## Open Questions to Google (sent)
- Channel→socket multiplexing with MAX_SOCKETS=8 and NCHANNELS=8; NUMA mapping guidance
- Progress cadence: Does tcpx_test(rc=0, done=0) advance state? recommended call frequency in single-process
- Recv lifecycle: correct consumed timing; whether memcpy path needs unpack flags
- Multi-NIC scheduling/binding patterns in plugin/NCCL code; pointers to key functions


## New Guidance from Google (2025-10-08)
- One channel ≈ one TCPX connection in the NCCL plugin; in our NIXL plugin, treat connections directly and don’t over-index on the channel abstraction
- Preferred per-NIC max measurement: single 200Gbps NIC with ~8 TCPX connections (~21.26 GB/s ceiling)
- NUMA: keep each GPU on its NUMA-local NIC(s); OK to hardcode a static GPU→NIC map for now (GPU0,1→NIC0; GPU2,3→NIC1; ...)
- vLLM mode: one process per GPU; each process operates a single NIC
- Keep NCCL’s threading mechanism for now; no app-level ACK required (simple send/recv is sufficient)

## Next Steps (checklist for next AI)
- [ ] READ tests/test_tcpx_perf_multi.cc completely and understand current flow
- [ ] TRACE how UCCL_TCPX_NUM_CHANNELS is used (grep for it in all files)
- [ ] UNDERSTAND current channel selection logic (is it round-robin? single-channel only?)
- [ ] WRITE a detailed implementation plan with code locations and proposed changes
- [ ] DISCUSS the plan with user before implementing
- [ ] IMPLEMENT incrementally: start with 2 channels, verify, then scale to 4
- [ ] VALIDATE: check logs show "4 channels created", TRACE shows 8 connections (2 GPUs × 4), bandwidth scales

## Pointers to Detail
- Handoff overview: p2p/tcpx/HANDOFF_README.md
- Executive summary (CN): p2p/tcpx/REPORT_EXEC_SUMMARY_CN.md
- Debug guide (how to read TRACE): p2p/tcpx/DEBUG_GUIDE.md
- Archive index (historical docs only): p2p/tcpx/docs/archive/README.md

---

Below is legacy content (outdated); retained for historical context. Start from the updated section above.

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

