# Handoff Checklist

**Date**: 2025-10-07  
**Status**: Ready for handoff  
**Next Person**: Developer or AI assistant continuing IRQ binding investigation

---

## ✅ Pre-Handoff Verification (Completed)

- [x] **Working configuration verified**
  - Single-NIC P2P benchmark working
  - Configuration: 1 channel per GPU, 1 NIC per GPU
  - Performance: 2.75 GB/s server, 1.17 GB/s client per GPU

- [x] **Code reverted to stable state**
  - `run_p2p_fullmesh.sh` using single-NIC config
  - `CHANNELS=1` (not 2 or 4)
  - GPU-to-NIC mapping: 0-1→eth1, 2-3→eth2, 4-5→eth3, 6-7→eth4

- [x] **Documentation complete**
  - All 6 documentation files created/updated
  - Investigation plan written (6 steps, detailed)
  - Handoff prompt updated for current status
  - Project status documented

- [x] **Diagnostics tool ready**
  - `run_nccl_test_tcpx.sh` enhanced with collection capability
  - Usage: `./run_nccl_test_tcpx.sh nccl 2 8 0 1 1`
  - Output: `diagnostics/nccl_<timestamp>/` with 13+ files

---

## 📋 For Next Person: First Steps

### Step 1: Read Documentation (30 minutes)

Read in this order:

1. **Quick overview**
   ```bash
   cat /home/daniel/uccl/p2p/tcpx/README.md
   ```

2. **Complete status**
   ```bash
   cat /home/daniel/uccl/p2p/tcpx/docs/PROJECT_STATUS.md
   ```

3. **Handoff context**
   ```bash
   cat /home/daniel/uccl/p2p/tcpx/docs/AI_HANDOFF_PROMPT.md
   ```

4. **Investigation plan**
   ```bash
   cat /home/daniel/uccl/p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md
   ```

### Step 2: Verify Working Configuration (10 minutes)

```bash
cd /home/daniel/uccl/p2p/tcpx

# Check configuration
grep "CHANNELS=" run_p2p_fullmesh.sh
# Should show: CHANNELS=${UCCL_TCPX_NUM_CHANNELS:-1}

# Check NIC mapping
grep -A 8 "map_gpu_to_ifaces()" run_p2p_fullmesh.sh
# Should show single NIC per GPU pair
```

### Step 3: Run NCCL Diagnostics (30 minutes)

```bash
cd /home/daniel/uccl/collective/rdma

# Run with diagnostics collection enabled
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1

# Wait for completion (~2-3 minutes)

# Verify diagnostics collected
ls -lh /home/daniel/uccl/diagnostics/nccl_*/

# View summary
cat /home/daniel/uccl/diagnostics/nccl_*/SUMMARY.txt
```

### Step 4: Begin Investigation (Follow Plan)

```bash
# Open the investigation plan
cat /home/daniel/uccl/p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md

# Follow Step 1.1 - 1.4 for data collection
# Then proceed to Step 2 for analysis
```

---

## 📁 File Inventory

### Documentation Created/Updated

1. **p2p/tcpx/README.md** (NEW)
   - Quick start guide
   - 200 lines

2. **p2p/tcpx/docs/PROJECT_STATUS.md** (NEW)
   - Comprehensive project overview
   - Current status, history, next steps
   - 300 lines

3. **p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md** (NEW)
   - 6-step systematic investigation plan
   - Detailed commands, expected outputs
   - 300 lines

4. **p2p/tcpx/docs/AI_HANDOFF_PROMPT.md** (UPDATED)
   - Complete rewrite for current status
   - Context injection prompt
   - Quick start commands
   - 411 lines

5. **p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md** (UPDATED)
   - Added resolution summary
   - Marked as RESOLVED
   - Historical context preserved

6. **HANDOFF_SUMMARY.md** (NEW)
   - Summary of work done
   - Next steps
   - Key insights

7. **HANDOFF_CHECKLIST.md** (NEW, this file)
   - Verification checklist
   - First steps for next person

### Code Modified

1. **collective/rdma/run_nccl_test_tcpx.sh** (ENHANCED)
   - Added diagnostics collection (6th parameter)
   - Pre-test, during-test, post-test data collection
   - ~100 lines added

2. **p2p/tcpx/run_p2p_fullmesh.sh** (REVERTED)
   - Back to working single-NIC configuration
   - CHANNELS=1
   - Single NIC per GPU pair

---

## 🎯 Current State Summary

### What's Working ✅
- Single-NIC P2P: 2.75 GB/s server, 1.17 GB/s client per GPU
- All 4 NICs (eth1-4) verified working
- NCCL reference: 18.7 GB/s bus bandwidth
- Diagnostics collection tool ready

### What's Not Working ❌
- Multi-NIC per GPU (devmem conflicts)
- Multi-channel on single NIC (performance degradation)

### What's Next 🔍
- IRQ/CPU binding investigation
- Systematic 6-step plan ready
- Estimated time: 11-17 hours

---

## 🚀 Quick Commands Reference

### Run Working P2P Benchmark
```bash
# Node 0
cd /home/daniel/uccl/p2p/tcpx
./run_p2p_fullmesh.sh server

# Node 1
./run_p2p_fullmesh.sh client <NODE0_ETH0_IP>
```

### Run NCCL with Diagnostics
```bash
cd /home/daniel/uccl/collective/rdma
./run_nccl_test_tcpx.sh nccl 2 8 0 1 1
```

### Check Results
```bash
# P2P logs
ls -lt /home/daniel/uccl/p2p/tcpx/logs/fullmesh_* | head -10
grep "PERF.*Avg.*BW:" /home/daniel/uccl/p2p/tcpx/logs/fullmesh_*.log

# NCCL diagnostics
ls -lt /home/daniel/uccl/diagnostics/ | head -5
cat /home/daniel/uccl/diagnostics/nccl_*/SUMMARY.txt

# Verify devmem
ethtool -S eth1 | grep rx_devmem_pkts
```

---

## 📊 Expected Performance

### Current (Working)
- **P2P Server**: 2.75 GB/s per GPU
- **P2P Client**: 1.17 GB/s per GPU
- **Total**: ~22 GB/s aggregate (8 GPUs receiving)

### Target (After Optimization)
- **Goal**: >5 GB/s per GPU (2x improvement)
- **Stretch**: >10 GB/s per GPU (approaching NCCL)

### Reference (NCCL)
- **Bus Bandwidth**: 18.7 GB/s
- **Note**: Different metric, not directly comparable

---

## 🔑 Key Insights

1. **Loopback doesn't work** - Always test on separate nodes
2. **Single-NIC is stable** - Don't try multi-NIC without solving devmem conflicts
3. **NCCL uses different architecture** - 1 process vs 8 processes
4. **IRQ binding may help** - NCCL sets explicit CPU affinity
5. **Measure first** - Collect diagnostics before making changes

---

## ⚠️ Important Notes

### Don't Do This
- ❌ Run server and client on same node (loopback fails)
- ❌ Try multi-NIC per GPU (known to fail with devmem conflicts)
- ❌ Increase channels on single NIC (degrades performance)
- ❌ Skip diagnostics collection (need data to make decisions)

### Do This
- ✅ Run on separate nodes
- ✅ Use single-NIC configuration
- ✅ Collect NCCL diagnostics first
- ✅ Follow investigation plan systematically
- ✅ Document results (positive or negative)

---

## 📞 Support Resources

### Documentation
- **Quick Start**: `p2p/tcpx/README.md`
- **Full Status**: `p2p/tcpx/docs/PROJECT_STATUS.md`
- **Handoff Guide**: `p2p/tcpx/docs/AI_HANDOFF_PROMPT.md`
- **Investigation Plan**: `p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md`

### Logs
- **P2P Logs**: `p2p/tcpx/logs/fullmesh_*.log`
- **NCCL Diagnostics**: `diagnostics/nccl_*/`

### Reference
- **NCCL Script**: `collective/rdma/run_nccl_test_tcpx.sh`
- **Network Config**: `scripts/node_ips/tcpx.txt`

---

## ✅ Final Checklist

Before starting work:

- [ ] Read all documentation (30 min)
- [ ] Verify working P2P config (10 min)
- [ ] Run NCCL diagnostics (30 min)
- [ ] Review investigation plan (15 min)
- [ ] Understand current state and next steps (15 min)

**Total prep time**: ~1.5 hours

Then begin Step 1 of investigation plan.

---

## 🎓 Learning Resources

### Understanding TCPX
- GPUDirect TCPX: Zero-copy GPU-to-GPU over TCP
- devmem-tcp: Kernel API for DMA
- cmsg: Control messages with buffer descriptors
- Flow steering: Traffic routing via dp-manager

### Understanding the Gap
- NCCL: 1 process, 8 GPUs, 4 NICs, 8 channels, Ring algorithm
- P2P: 8 processes, 1 GPU each, 1 NIC each, 1 channel, simple P2P
- Key differences: Architecture, algorithm, CPU affinity

### Investigation Approach
- Systematic: Follow 6-step plan
- Data-driven: Collect before deciding
- Incremental: Test one variable at a time
- Documented: Record all results

---

**Handoff Complete**: 2025-10-07  
**Status**: Ready for next phase  
**Next Milestone**: Complete IRQ binding investigation

Good luck! 🚀

