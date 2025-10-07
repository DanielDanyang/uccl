# Phase 1: Round-Robin NIC Distribution Fix

**Date**: 2025-10-07  
**Status**: ✅ Implemented, ready for testing

---

## 🐛 Problem Summary

### Root Cause
The `ChannelManager` constructor had a critical flaw in NIC selection logic:

1. **Scoring Phase**: Correctly scored all 4 NICs based on PCIe proximity
2. **Selection Phase**: Only picked the **single best-scoring NIC** per GPU
3. **Replication Phase**: Replicated the same NIC to fill all channels

### Observed Behavior (8 channels/GPU)

**GPU 0-3 (NUMA 0)**:
- All 8 channels → eth1 (netDev 0) only
- eth2 completely unused

**GPU 4-5 (NUMA 1)**:
- All 8 channels → eth3 (netDev 2) only
- eth4 completely unused

**Result**:
- GPU 4: 8 channels on eth3 ✅ accepted
- GPU 5: 8 MORE channels on eth3 ❌ **stalled at channel 4**
- Total: 16 channels on eth3 exceeded TCPX plugin limits

### Error Symptoms
```
[ChannelManager] Channel 4: Accept not ready (rc=0), retrying...
[ChannelManager] Failed to accept connection for channel 4 after 100 retries
[ERROR] GPU 5: server_accept_all failed
```

---

## ✅ Phase 1 Fix: Round-Robin Distribution

### Implementation

**File**: `p2p/tcpx/src/channel_manager.cc`  
**Lines**: 266-290 (replaced 266-278)

**Key Changes**:
1. Build a **pool of ALL CUDA-supported NICs** (not just best-scoring ones)
2. Distribute channels via **round-robin** across the entire pool
3. Add logging to show distribution strategy

### Code Changes

**Before** (lines 266-278):
```cpp
if ((int)selected.size() < num_channels_) {
  size_t base = selected.size();
  if (base == 0) {
    selected.push_back(sorted.front());
    base = 1;
  }
  while ((int)selected.size() < num_channels_) {
    const Candidate& src = selected[selected.size() % base];  // ❌ Only replicates "selected" NICs
    selected.push_back(src);
  }
}
```

**After** (lines 266-290):
```cpp
if ((int)selected.size() < num_channels_) {
  // Phase 1 Fix: Round-robin across ALL available NICs to avoid saturating a single NIC.
  // This prevents accept stalls when multiple GPUs try to use the same NIC.
  // Build a pool of all CUDA-supported NICs for round-robin distribution.
  std::vector<Candidate> pool;
  for (const auto& cand : sorted) {
    if (cand.cuda_supported) {
      pool.push_back(cand);
    }
  }
  
  if (pool.empty()) {
    // Fallback: if no CUDA-supported NICs, use the first available NIC
    pool.push_back(sorted.front());
  }
  
  std::cout << "[ChannelManager] GPU " << gpu_id_ << ": Distributing " << num_channels_ 
            << " channels across " << pool.size() << " NICs (round-robin)" << std::endl;
  
  // Round-robin across all NICs in the pool
  while ((int)selected.size() < num_channels_) {
    const Candidate& src = pool[selected.size() % pool.size()];  // ✅ Round-robin all NICs
    selected.push_back(src);
  }
}
```

---

## 📊 Expected Results

### With 4 Channels/GPU (Phase 1 Test)

**Total Channels**: 8 GPUs × 4 channels = 32 channels  
**Distribution**: 32 channels ÷ 4 NICs = **8 channels per NIC**

**Expected Mapping**:
```
GPU 0: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 1: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 2: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 3: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 4: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 5: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 6: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
GPU 7: ch0→eth1, ch1→eth2, ch2→eth3, ch3→eth4
```

**NIC Load**:
- eth1 (netDev 0): 8 channels (2 per GPU × 4 GPUs)
- eth2 (netDev 1): 8 channels
- eth3 (netDev 2): 8 channels
- eth4 (netDev 3): 8 channels

### Success Criteria

✅ **All 8 GPUs complete `server_accept_all()` without stalling**  
✅ **All 4 NICs show usage in logs**  
✅ **No "Failed to accept connection" errors**  
✅ **Even distribution: ~8 channels per NIC**

---

## 🧪 Testing Instructions

### Step 1: Run Phase 1 Test (4 channels/GPU)

**Server (Node 0)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh server
```

**Client (Node 1)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh client <NODE0_IP>
```

### Step 2: Verify NIC Distribution

```bash
cd /home/daniel/uccl/p2p/tcpx
./verify_nic_distribution.sh
```

**Expected Output**:
```
=== NIC Usage Summary ===
netDev 0 (eth1): 8 channels
netDev 1 (eth2): 8 channels
netDev 2 (eth3): 8 channels
netDev 3 (eth4): 8 channels

=== Accept Status ===
✅ GPU 0: Accepted 4 connections
✅ GPU 1: Accepted 4 connections
✅ GPU 2: Accepted 4 connections
✅ GPU 3: Accepted 4 connections
✅ GPU 4: Accepted 4 connections
✅ GPU 5: Accepted 4 connections
✅ GPU 6: Accepted 4 connections
✅ GPU 7: Accepted 4 connections

=== Error Summary ===
✅ No errors detected
```

### Step 3: Scale to 8 Channels/GPU (If Step 2 Passes)

```bash
# Server
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh server

# Client
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh client <NODE0_IP>

# Verify
./verify_nic_distribution.sh
```

**Expected**: 16 channels per NIC (8 GPUs × 8 channels ÷ 4 NICs)

---

## ⚠️ Known Limitations

### 1. NUMA Affinity Not Optimal
**Issue**: Round-robin distributes across ALL 4 NICs, including cross-NUMA NICs  
**Impact**: GPU 0-3 (NUMA 0) will use eth3/eth4 (NUMA 1), causing cross-NUMA traffic  
**Mitigation**: Phase 2 will implement NUMA-aware selection

### 2. Not Production-Ready
**Issue**: This is a quick fix to unblock testing  
**Impact**: Performance may be suboptimal due to cross-NUMA traffic  
**Mitigation**: Phase 2 will optimize for NUMA locality

---

## 🎯 Next Steps

### If Phase 1 Test Passes:
1. ✅ Confirm all 8 GPUs complete accept
2. ✅ Verify all 4 NICs are used
3. 🔄 Scale to 8 channels/GPU
4. 📊 Measure bandwidth (expect >5 GB/s)
5. 🚀 Proceed to Phase 2 (NUMA-aware selection)

### If Phase 1 Test Fails:
1. ❌ Check if TCPX plugin has per-NIC limits
2. 🔍 Reduce to 2 channels/GPU for testing
3. 📞 Contact Google for plugin limitations
4. 🔄 Reconsider architecture

---

## 📝 Files Modified

### Modified Files
- `p2p/tcpx/src/channel_manager.cc` (lines 266-290)

### New Files
- `p2p/tcpx/test_phase1_4ch.sh` - Test script for 4 channels/GPU
- `p2p/tcpx/verify_nic_distribution.sh` - Log analysis script
- `p2p/tcpx/PHASE1_ROUNDROBIN_FIX.md` - This document

### Build Status
✅ **Compiled successfully**
```bash
$ make test_tcpx_perf_orchestrator
g++ -std=c++17 -fPIC -O2 -Wall -Iinclude -I. -I/usr/local/cuda/include -c src/channel_manager.cc -o src/channel_manager.o
Building test_tcpx_perf_orchestrator...
/usr/local/cuda/bin/nvcc ... -o tests/test_tcpx_perf_orchestrator ...
```

---

## 📈 Performance Expectations

### Phase 1 (Round-Robin, Cross-NUMA)
- **Target**: >5 GB/s bus bandwidth
- **Baseline**: 2.75 GB/s (single-NIC)
- **Expected**: 2x improvement from multi-NIC

### Phase 2 (NUMA-Aware, Future)
- **Target**: >15 GB/s bus bandwidth
- **NCCL Baseline**: 19.176 GB/s
- **Expected**: Within 20% of NCCL

---

**Status**: ✅ Phase 1 implemented, ready for GCP testing  
**Next Action**: Run `test_phase1_4ch.sh` on both nodes and verify results

