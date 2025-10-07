# TCPX Multi-Channel Project - Current Status

**Last Updated**: 2025-10-06  
**Status**: ⚠️ **CODE NEEDS REVERT - BROKEN BY INCORRECT FIX**

---

## Executive Summary

The multi-channel TCPX implementation is **functionally complete** but currently **broken** due to an incorrect "fix" applied today. The code needs to be reverted to restore working state.

### What Works
- ✅ Single-channel mode (1 GPU → 1 NIC)
- ✅ Multi-GPU mode (4 GPUs → 4 NICs)
- ✅ Pipelined recv/unpack processing
- ✅ Sliding window flow control
- ✅ GPU-NIC topology detection

### What's Broken
- ❌ Two-phase NIC selection (allows incompatible GPU-NIC pairs)
- ❌ Multi-channel mode on single GPU (tries to use NICs on different PCIe roots)

---

## Critical Understanding: Hardware Topology

### GCE H100 Node Configuration

**Per Node**:
- 8 GPUs (H100)
- 4 NICs (200 Gbps each)
- 4 PCIe root complexes

**PCIe Topology**:
```
pci0000:01: GPU 0, GPU 1, eth1  (1 NIC for 2 GPUs)
pci0000:07: GPU 2, GPU 3, eth2  (1 NIC for 2 GPUs)
pci0000:81: GPU 4, GPU 5, eth3  (1 NIC for 2 GPUs)
pci0000:87: GPU 6, GPU 7, eth4  (1 NIC for 2 GPUs)
```

### Key Constraint: PCIe Isolation

**Hardware Limitation**:
- A GPU can **only** use NICs on its PCIe root complex
- Attempting to use a NIC on a different PCIe root causes `fatal, rx no cmsg`
- This is a **hardware constraint**, not a software preference

**Example**:
- GPU 0 (pci0000:01) can use eth1 (pci0000:01) ✅
- GPU 0 (pci0000:01) **cannot** use eth2 (pci0000:07) ❌

---

## What Went Wrong Today

### The Mistake

**Incorrect Assumption**:
> "GPU 0 should be able to use 2 NICs (eth1 + eth2) for 2× bandwidth"

**Reality**:
- GPU 0 can only use **1 NIC** (eth1)
- To use multiple NICs, you need **multiple GPUs**

### The Broken "Fix"

**File**: `p2p/tcpx/src/channel_manager.cc` (lines 149-193)

**What was changed**:
- Implemented "two-phase selection" that accepts NICs with `score < 0`
- Allowed GPU 0 to select eth2 (score=-1000, different PCIe root)

**Why it's wrong**:
- `score < 0` means **different PCIe root complex**
- Different PCIe root = **hardware incompatibility**
- Result: `fatal, rx no cmsg` → connection failure

### The Original Code Was Correct

```cpp
// ORIGINAL (CORRECT):
for (const auto& cand : sorted) {
  if (!cand.cuda_supported) continue;
  if (!gpu_pci_segments.empty() && cand.score < 0) continue;  // ✅ Enforces hardware constraint
  selected.push_back(cand);
}
```

This logic **correctly rejects** NICs on different PCIe roots.

---

## How to Actually Use Multiple NICs

### Option 1: Multi-GPU Single-Process

**Use multiple GPUs, each with its own NIC**:

```bash
# Use 4 GPUs to utilize all 4 NICs
UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi server 0,2,4,6
```

**Mapping**:
- Channel 0: GPU 0 → eth1
- Channel 1: GPU 2 → eth2
- Channel 2: GPU 4 → eth3
- Channel 3: GPU 6 → eth4

**Result**: 4× aggregate bandwidth

### Option 2: Multi-Process (NCCL-Style)

**One process per GPU**:

```bash
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES=$gpu \
  UCCL_TCPX_NUM_CHANNELS=1 \
  ./tests/test_tcpx_perf_multi server 0 &
done
```

**Mapping** (automatic via topology detection):
- Process 0: GPU 0 → eth1
- Process 1: GPU 1 → eth1 (shared)
- Process 2: GPU 2 → eth2
- Process 3: GPU 3 → eth2 (shared)
- ... etc

**Result**: 8 processes, 4 NICs, full utilization

---

## Immediate Action Required

### 1. Revert the Broken Fix

**File**: `p2p/tcpx/src/channel_manager.cc`

**Revert lines 149-193** to original single-phase selection:

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

### 2. Test Single-Channel Mode

```bash
# Server
UCCL_TCPX_NUM_CHANNELS=1 ./tests/test_tcpx_perf_multi server 0

# Client
UCCL_TCPX_NUM_CHANNELS=1 ./tests/test_tcpx_perf_multi client <ip> 0
```

**Expected**:
```
[ChannelManager] Channel 0 → netDev 0 (eth1, score=296)
[ChannelManager] Created 1 channel(s) for GPU 0
```

### 3. Test Multi-GPU Mode

```bash
# Server (4 GPUs → 4 NICs)
UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi server 0,2,4,6

# Client
UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi client <ip> 0,2,4,6
```

**Expected**: 4× bandwidth compared to single-channel

---

## Project Architecture

### Core Components

1. **ChannelManager** (`src/channel_manager.cc`)
   - GPU-NIC topology detection
   - Multi-channel lifecycle management
   - Memory registration across channels

2. **SlidingWindow** (`src/sliding_window.cc`)
   - Flow control (max 16 inflight per channel)
   - Request tracking and completion

3. **Bootstrap** (`src/bootstrap.cc`)
   - Handle exchange between server/client
   - Multi-channel coordination

4. **Unpack Kernels** (`device/unpack_kernels.cu`)
   - GPU kernel for reassembling scattered packets
   - Optimized for small/large transfers

### Test Programs

1. **test_tcpx_perf_multi.cc** (884 lines)
   - Multi-channel performance testing
   - Pipelined recv/unpack processing
   - Comprehensive debug logging

2. **test_tcpx_transfer_multi.cc** (752 lines)
   - Multi-channel correctness testing
   - Preserves all debugging experience

---

## Key Design Decisions

### 1. Chunk Size: 512KB
- Validated through extensive testing
- Balances latency vs. throughput
- Works well with 16-slot sliding window

### 2. Sliding Window: 16 slots (server), 12 slots (client)
- Server: Max 16 inflight recvs per channel
- Client: Max 12 inflight sends per channel (leave margin)
- Prevents deadlock and memory pressure

### 3. Pipelined Processing
- Server: irecv → poll → unpack kernel (async) → continue
- Unpack kernels execute in parallel across channels
- Kernel completion checked via sliding window

### 4. Round-Robin Distribution
- Chunks distributed across channels: 0, 1, 0, 1, ...
- Each channel gets equal share of data
- Maximizes parallelism across NICs

---

## Documentation Structure

### Essential Reading
1. **CURRENT_STATUS.md** (this file) - Current state and action items
2. **TOPOLOGY_FIX.md** - Analysis of today's mistake and lessons learned
3. **CRITICAL_FIXES.md** - Previous bug fixes (mark as "REVERTED")

### Reference
4. **TEST_TCPX_PERF_EXPLAINED.md** - Detailed code walkthrough
5. **SLIDING_WINDOW_VISUAL.md** - Sliding window visualization
6. **TCPX_LOGIC_MAPPING.md** - TCPX API reference

### Historical
7. **PERF_DIARY.md** - Performance optimization history
8. **PHASE1_COMPLETE.md**, **PHASE3_COMPLETE.md** - Development milestones

---

## Next Steps (After Revert)

### Short Term
1. ✅ Revert broken fix
2. ✅ Verify single-channel mode works
3. ✅ Test multi-GPU mode (4 GPUs → 4 NICs)

### Medium Term
4. Implement multi-process launcher script
5. Test 8-process mode (full NIC utilization)
6. Performance benchmarking and optimization

### Long Term
7. Integration with UCCL collective operations
8. Multi-node testing (2 nodes × 8 GPUs)
9. Production deployment

---

## Contact and Handoff

### Key Files to Understand
- `src/channel_manager.cc` - Topology detection and channel management
- `tests/test_tcpx_perf_multi.cc` - Main test program
- `device/unpack_kernels.cu` - GPU unpack kernel

### Common Issues
- `rx no cmsg`: GPU-NIC PCIe mismatch (check topology)
- Hanging: Sliding window deadlock (check window sizes)
- Low performance: Chunk size or window size tuning

### Testing Commands
```bash
# Single-channel (baseline)
UCCL_TCPX_NUM_CHANNELS=1 ./tests/test_tcpx_perf_multi server 0

# Multi-GPU (4× bandwidth)
UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi server 0,2,4,6
```

---

**Status**: ⚠️ **REVERT REQUIRED** - See "Immediate Action Required" section above

