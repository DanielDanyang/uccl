# GPU-NIC Topology Analysis and Current Status

**Date**: 2025-10-06
**Status**: ⚠️ **ANALYSIS COMPLETE - AWAITING DECISION**

---

## Problem Analysis

### Initial Symptom
The 2-channel test failed with `fatal, rx no cmsg` error on eth2 (Channel 1).

### Root Cause Discovery

**Hardware Topology** (from logs):
```
GPU 0: 0000:04:00.0 on pci0000:01
eth1:  0000:06:00.0 on pci0000:01 (score=296)  ✅ Same PCIe root
eth2:  0000:0c:00.0 on pci0000:07 (score=-1000) ❌ Different PCIe root
eth3:  0000:86:00.0 on pci0000:81
eth4:  0000:8c:00.0 on pci0000:87
```

**Critical Finding**:
- eth2 is GPUDirect-capable (has `NCCL_PTR_CUDA` support)
- BUT eth2 **cannot access GPU 0's memory** because they are on different PCIe root complexes
- When TCPX tries to receive data via eth2 into GPU 0's memory, the **kernel driver cannot provide devmem control messages**
- Result: `fatal, rx no cmsg` → connection reset

**Incorrect Assumption**:
- We assumed "GPUDirect-capable" means the NIC can access any GPU's memory
- **Reality**: PCIe topology matters! Only devices on the same PCIe root complex can directly access each other's memory

---

## Actual Hardware Topology (GCE H100 Nodes)

### Per-Node Configuration
- **8 GPUs per node** (16 total across 2 nodes)
- **4 NICs per node**: eth1, eth2, eth3, eth4

### PCIe Topology (from logs and lspci)

```
PCIe Root Complex 1 (pci0000:01):
  ├─ GPU 0 (0000:04:00.0)
  ├─ GPU 1 (0000:05:00.0)
  └─ eth1  (0000:06:00.0)  ← Only eth1 is on this tree!

PCIe Root Complex 2 (pci0000:07):
  ├─ GPU 2 (0000:0a:00.0)
  ├─ GPU 3 (0000:0b:00.0)
  └─ eth2  (0000:0c:00.0)  ← Only eth2 is on this tree!

PCIe Root Complex 3 (pci0000:81):
  ├─ GPU 4 (0000:84:00.0)
  ├─ GPU 5 (0000:85:00.0)
  └─ eth3  (0000:86:00.0)  ← Only eth3 is on this tree!

PCIe Root Complex 4 (pci0000:87):
  ├─ GPU 6 (0000:8a:00.0)
  ├─ GPU 7 (0000:8b:00.0)
  └─ eth4  (0000:8c:00.0)  ← Only eth4 is on this tree!
```

### Correct GPU-NIC Mapping

| GPU | PCIe Root | Compatible NIC | Reason |
|-----|-----------|---------------|--------|
| GPU 0 | pci0000:01 | **eth1 only** | Same PCIe root complex |
| GPU 1 | pci0000:01 | **eth1 only** | Same PCIe root complex |
| GPU 2 | pci0000:07 | **eth2 only** | Same PCIe root complex |
| GPU 3 | pci0000:07 | **eth2 only** | Same PCIe root complex |
| GPU 4 | pci0000:81 | **eth3 only** | Same PCIe root complex |
| GPU 5 | pci0000:81 | **eth3 only** | Same PCIe root complex |
| GPU 6 | pci0000:87 | **eth4 only** | Same PCIe root complex |
| GPU 7 | pci0000:87 | **eth4 only** | Same PCIe root complex |

**Key Insight**: Each GPU can only use **ONE NIC**, not two!

---

## Why the "Two-Phase Selection" Fix Was Wrong

### What I Implemented
```cpp
// Phase 1: Select NICs with positive score (same PCIe tree)
// Phase 2: If not enough, add any GPUDirect-capable NICs (even with negative score)
```

### Why It Failed
- **Phase 2 is fundamentally flawed**: It assumes any GPUDirect-capable NIC can access any GPU's memory
- **Reality**: PCIe topology is a hard constraint, not a preference
- **Result**: eth2 (score=-1000) was selected for GPU 0, causing `rx no cmsg` fatal error

### The Correct Logic (Already in Original Code!)

The **original single-phase selection** was actually correct:

```cpp
// ORIGINAL (CORRECT):
for (const auto& cand : sorted) {
  if (!cand.cuda_supported) continue;
  if (!gpu_pci_segments.empty() && cand.score < 0) continue;  // ✅ Correctly rejects eth2!
  selected.push_back(cand);
  if ((int)selected.size() == num_channels_) break;
}
```

**Why it's correct**:
- `score < 0` means the NIC is on a **different PCIe root complex**
- Different PCIe root = **cannot access GPU's memory** (even if GPUDirect-capable)
- Rejecting `score < 0` is **not "too strict"**, it's **enforcing hardware constraints**

---

## Current Code Status

### What Was Changed (NEEDS REVERT!)

`p2p/tcpx/src/channel_manager.cc:149-193` was modified to use two-phase selection.

**This change is WRONG and must be reverted!**

### What Should Be Restored

The original single-phase selection logic:

```cpp
std::vector<Candidate> selected;
selected.reserve(num_channels_);
for (const auto& cand : sorted) {
  if (!cand.cuda_supported) continue;
  if (!gpu_pci_segments.empty() && cand.score < 0) continue;  // ✅ Keep this!
  selected.push_back(cand);
  if ((int)selected.size() == num_channels_) break;
}

if (selected.empty()) {
  std::cerr << "[ChannelManager] Warning: No GPU-direct capable NICs detected for GPU "
            << gpu_id_ << ". Falling back to first enumerated NIC." << std::endl;
  selected.push_back(sorted.front());
}

if ((int)selected.size() < num_channels_) {
  std::cerr << "[ChannelManager] Warning: Requested " << num_channels_
            << " channels but only " << selected.size()
            << " GPU-direct NICs matched this GPU. Reducing channel count." << std::endl;
  num_channels_ = selected.size();
}
```

---

## The Real Problem: Misunderstanding the Hardware

### What We Thought
- "GPU 0 should use 2 NICs (eth1 + eth2) for 2× bandwidth"
- "Each GPU paired with 2 NICs on nearby PCIe trees"

### What's Actually True
- **Each GPU can only use 1 NIC** (the one on its PCIe root complex)
- **GPU 0 → eth1 only** (both on pci0000:01)
- **GPU 2 → eth2 only** (both on pci0000:07)
- To use multiple NICs, you need **multiple GPUs**

### Correct Behavior (After Revert)

**Single-channel test** (GPU 0):
```
[ChannelManager] Channel 0 → netDev 0 (eth1, score=296)
[ChannelManager] Created 1 channel(s) for GPU 0
```
✅ Correct! GPU 0 can only use eth1.

**Two-channel test** (GPU 0):
```
[ChannelManager] Warning: Requested 2 channels but only 1 GPU-direct NICs matched this GPU.
[ChannelManager] Channel 0 → netDev 0 (eth1, score=296)
[ChannelManager] Created 1 channel(s) for GPU 0
```
✅ Correct! GPU 0 cannot use 2 channels because it only has access to 1 NIC.

---

## How to Actually Use Multiple NICs

### Option 1: Multi-GPU Single-Process (Current test_tcpx_perf_multi)

**Use 4 GPUs to utilize all 4 NICs**:

```bash
# Server (uses GPU 0, 2, 4, 6 → eth1, eth2, eth3, eth4)
UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
  ./tests/test_tcpx_perf_multi server 0,2,4,6

# Client
UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 \
  ./tests/test_tcpx_perf_multi client <server_ip> 0,2,4,6
```

**Expected**:
```
[ChannelManager] GPU 0: Channel 0 → eth1 (score=296)
[ChannelManager] GPU 2: Channel 1 → eth2 (score=296)
[ChannelManager] GPU 4: Channel 2 → eth3 (score=296)
[ChannelManager] GPU 6: Channel 3 → eth4 (score=296)
```

**Result**: 4× aggregate bandwidth (each GPU uses its local NIC)

---

### Option 2: Multi-Process (NCCL-Style, One Process Per GPU)

**Correct understanding of NCCL's approach**:
- NCCL runs **one process per GPU** (via mpirun -N 8)
- Each process uses `CUDA_VISIBLE_DEVICES=i` to see only GPU i
- NCCL's topology detection automatically selects the **one NIC** on that GPU's PCIe tree
- **NOT** "each GPU uses 2 NICs", but "each process uses 1 GPU → 1 NIC"

**Example launch script** (corrected):

```bash
#!/bin/bash
# Launch 8 processes (one per GPU)
# Each process uses CUDA_VISIBLE_DEVICES to see only its GPU
# ChannelManager automatically selects the matching NIC

for gpu in {0..7}; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export UCCL_TCPX_NUM_CHANNELS=1  # Each GPU uses 1 NIC!
    export UCCL_TCPX_PERF_SIZE=67108864

    log="logs/server_gpu${gpu}.log"
    ./tests/test_tcpx_perf_multi server 0 2>&1 | tee "$log"
  ) &
  sleep 1
done

wait
```

**Expected per-process output**:
```
# Process 0 (CUDA_VISIBLE_DEVICES=0):
[ChannelManager] GPU 0 → Channel 0 → eth1

# Process 2 (CUDA_VISIBLE_DEVICES=2):
[ChannelManager] GPU 0 (physical GPU 2) → Channel 0 → eth2

# Process 4 (CUDA_VISIBLE_DEVICES=4):
[ChannelManager] GPU 0 (physical GPU 4) → Channel 0 → eth3

# Process 6 (CUDA_VISIBLE_DEVICES=6):
[ChannelManager] GPU 0 (physical GPU 6) → Channel 0 → eth4
```

**Result**: 8 processes × 1 NIC each = full utilization of all 8 NICs

---

## Key Lessons Learned

### 1. PCIe Topology is a Hard Constraint
- **Not a preference**: You cannot use a NIC on a different PCIe root complex
- **Hardware limitation**: Kernel driver cannot provide devmem cmsg across PCIe roots
- **Result**: `fatal, rx no cmsg` → connection failure

### 2. "GPUDirect-Capable" ≠ "Can Access Any GPU"
- A NIC being GPUDirect-capable means it **supports the technology**
- It does **NOT** mean it can access any GPU's memory
- PCIe topology determines which GPU-NIC pairs can work together

### 3. The Original Code Was Correct
- Rejecting `score < 0` NICs was **not "too strict"**
- It was **correctly enforcing hardware constraints**
- The "fix" that relaxed this constraint was fundamentally flawed

### 4. How NCCL Actually Works
- **One process per GPU** (not "one process uses multiple GPUs")
- Each process uses **one NIC** (the one on its GPU's PCIe tree)
- Aggregate bandwidth comes from **multiple processes**, not multiple NICs per process

---

## Action Items

### Immediate (MUST DO)

1. **Revert the two-phase selection change** in `channel_manager.cc`
2. **Restore original single-phase logic** (reject `score < 0`)
3. **Test with single-channel mode** (GPU 0 → eth1 only)

### Next Steps (After Revert)

1. **Verify single-channel works**: GPU 0 → eth1, GPU 2 → eth2, etc.
2. **Test multi-GPU mode**: Use 4 GPUs (0,2,4,6) to utilize all 4 NICs
3. **Implement multi-process launcher**: One process per GPU (NCCL-style)

---

## Files to Update

### Code
- **p2p/tcpx/src/channel_manager.cc**: Revert lines 149-193 to original logic

### Documentation
- **This file** (TOPOLOGY_FIX.md): Keep as analysis/lessons learned
- **CRITICAL_FIXES.md**: Mark as "REVERTED - analysis was wrong"
- **README.md**: Update to reflect correct understanding

---

**Status**: ⚠️ **AWAITING REVERT** - Current code is broken, needs immediate fix!

