# Step 3 Implementation Summary

**Date**: 2025-10-07
**Status**: ✅ Complete - Bugs Fixed - Ready for Testing
**Update**: 2025-10-07 - Fixed critical bandwidth calculation bug

---

## 📋 What Was Done

### 1. Code Implementation ✅

**File**: `tests/test_tcpx_perf_orchestrator.cc`

**Initial Implementation**:
- Added `<chrono>` and `<thread>` headers (lines 108-109)
- Added `getEnvSize()` helper function (lines 136-145)
- Implemented server receive loop (lines 530-636)
- Implemented client send loop (lines 757-859)

**Bug Fixes** (2025-10-07):
- ✅ Fixed bandwidth calculation (each GPU now transfers full test_size)
- ✅ Fixed buffer overflow (increased to 256 MB max, added validation)
- ✅ Fixed channel count assumption (query actual count per GPU)

**Total**: ~250 lines of new code

**See**: `STEP3_BUGFIXES.md` for detailed bug analysis

### 2. Test Infrastructure ✅

**New Scripts**:
- `test_step3_bandwidth.sh` - Test script with bandwidth measurement
- `analyze_bandwidth.sh` - Automated bandwidth analysis
- `STEP3_DATA_PLANE.md` - Detailed technical documentation
- `STEP3_QUICKSTART.md` - Quick reference guide
- `STEP3_SUMMARY.md` - This file

### 3. Build Verification ✅

```bash
$ make test_tcpx_perf_orchestrator
✅ Compiled successfully with no warnings or errors
```

---

## 🎯 Implementation Approach

### Simplified Design (Step 3)

We chose a **simple synchronous approach** for Step 3:

**What We Implemented**:
- ✅ Basic send/recv with `tcpx_isend()` / `tcpx_irecv()`
- ✅ Round-robin channel selection across all GPUs
- ✅ Synchronous completion (wait for each chunk)
- ✅ Performance measurement and reporting

**What We Deferred**:
- ⏳ Sliding window flow control (not needed for sync)
- ⏳ Unpack kernel (direct receive works)
- ⏳ Asynchronous pipelining (sync is simpler)

**Rationale**:
1. **Validate architecture first**: Prove single-process works
2. **Easier to debug**: Synchronous is simpler
3. **Good enough**: May still hit >10 GB/s
4. **Can optimize later**: Add complexity if needed

---

## 📊 Key Features

### Round-Robin Channel Selection

**Algorithm**:
```cpp
// Total channels = 8 GPUs × N channels/GPU
int total_channels = kNumGPUs * num_channels_per_gpu;

// For each chunk:
int channel_global_id = global_chunk_idx % total_channels;
int gpu_id = channel_global_id / num_channels_per_gpu;
int channel_local_id = channel_global_id % num_channels_per_gpu;
```

**Example** (4 channels/GPU = 32 total):
```
Chunk 0  → GPU 0, Channel 0
Chunk 1  → GPU 0, Channel 1
Chunk 2  → GPU 0, Channel 2
Chunk 3  → GPU 0, Channel 3
Chunk 4  → GPU 1, Channel 0
...
Chunk 31 → GPU 7, Channel 3
Chunk 32 → GPU 0, Channel 0 (wraps around)
```

**Benefits**:
- Even load distribution across all channels
- All GPUs and NICs utilized
- Simple and predictable

### Synchronous Completion

**Client Side**:
```cpp
// Send chunk
tcpx_isend(ch.send_comm, src_ptr, size, tag, mhandle, &request);

// Wait for completion
while (!done) {
  tcpx_test(request, &done, &sent_size);
  if (!done) sleep(10us);
}
```

**Server Side**:
```cpp
// Post receive
tcpx_irecv(ch.recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &request);

// Wait for completion
while (!done) {
  tcpx_test(request, &done, &received_size);
  if (!done) sleep(10us);
}

// Mark as consumed
tcpx_irecv_consumed(ch.recv_comm, 1, request);
```

**Benefits**:
- Simple and easy to debug
- No risk of exhausting request pool
- No complex flow control needed

**Trade-off**:
- May not fully saturate NICs
- Lower throughput than async pipelining
- Acceptable for initial validation

---

## 🧪 Testing Plan

### Test Matrix

| Test | Channels/GPU | Total Channels | Target BW | Status |
|------|--------------|----------------|-----------|--------|
| Step 3a | 4 | 32 | >5 GB/s | 🔄 To Test |
| Step 3b | 8 | 64 | >10 GB/s | ⏳ After 3a |

### Test Configuration

**Default**:
```bash
UCCL_TCPX_NUM_CHANNELS=4      # 4 channels per GPU
UCCL_TCPX_PERF_SIZE=67108864  # 64MB test size
UCCL_TCPX_PERF_ITERS=20       # 20 iterations
UCCL_TCPX_CHUNK_BYTES=524288  # 512KB chunks
```

**Calculation**:
- Test size: 64MB
- Chunk size: 512KB
- Chunks per iteration: 64MB ÷ 512KB = 128 chunks
- With 32 channels: 128 ÷ 32 = 4 chunks per channel

---

## 📈 Performance Expectations

### Baseline (Multi-Process)
- **Bandwidth**: 2.75 GB/s
- **Architecture**: 8 processes, 1 GPU each, 1 channel each
- **NICs**: 4 (2 GPUs per NIC)

### Step 3 Targets

**With 4 Channels/GPU**:
- **Target**: >5 GB/s (2x improvement)
- **Architecture**: 1 process, 8 GPUs, 4 channels each
- **NICs**: 4 (8 channels per NIC)

**With 8 Channels/GPU**:
- **Target**: >10 GB/s (4x improvement)
- **Architecture**: 1 process, 8 GPUs, 8 channels each
- **NICs**: 4 (16 channels per NIC)

### NCCL Reference
- **Bandwidth**: 19.176 GB/s
- **Long-term goal**: Get within 20% (>15 GB/s)

---

## 🚀 How to Test

### Step 3a: 4 Channels/GPU

**Server (Node 0)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh server
```

**Client (Node 1)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh client <SERVER_IP>
```

**Analyze**:
```bash
./analyze_bandwidth.sh
```

**Expected Output**:
```
Average bandwidth: >5 GB/s
✅ SUCCESS: Bandwidth >5 GB/s achieved!
```

### Step 3b: 8 Channels/GPU (If 3a Passes)

```bash
# Server
UCCL_TCPX_NUM_CHANNELS=8 ./test_step3_bandwidth.sh server

# Client
UCCL_TCPX_NUM_CHANNELS=8 ./test_step3_bandwidth.sh client <SERVER_IP>

# Analyze
./analyze_bandwidth.sh
```

**Expected Output**:
```
Average bandwidth: >10 GB/s
✅ SUCCESS: Bandwidth >10 GB/s achieved!
```

---

## ✅ Success Criteria

### Step 3a (4 channels/GPU):
- [ ] Data transfer completes without errors
- [ ] Both client and server report bandwidth
- [ ] Bandwidth >2.75 GB/s (better than baseline)
- [ ] Bandwidth >5 GB/s (target)

### Step 3b (8 channels/GPU):
- [ ] Data transfer completes without errors
- [ ] Bandwidth >5 GB/s (better than 4 channels)
- [ ] Bandwidth >10 GB/s (target)

---

## 📁 Files Changed

### Modified Files
| File | Lines | Change |
|------|-------|--------|
| `tests/test_tcpx_perf_orchestrator.cc` | 108-109 | Added headers |
| `tests/test_tcpx_perf_orchestrator.cc` | 117-120 | Fixed buffer constants |
| `tests/test_tcpx_perf_orchestrator.cc` | 136-145 | Added getEnvSize() |
| `tests/test_tcpx_perf_orchestrator.cc` | 530-636 | Server receive loop (fixed) |
| `tests/test_tcpx_perf_orchestrator.cc` | 757-859 | Client send loop (fixed) |

### New Files
| File | Purpose |
|------|---------|
| `test_step3_bandwidth.sh` | Test script |
| `analyze_bandwidth.sh` | Bandwidth analysis |
| `STEP3_DATA_PLANE.md` | Technical documentation |
| `STEP3_QUICKSTART.md` | Quick reference |
| `STEP3_SUMMARY.md` | This file |
| `STEP3_BUGFIXES.md` | Bug fix documentation |

---

## 🎯 Next Steps

### After Step 3 Passes:
1. **Measure bandwidth**: Compare with baseline and NCCL
2. **Analyze bottlenecks**: CPU, NIC, or GPU?
3. **Decide on next step**:
   - Option A: Step 4 (thread affinity)
   - Option B: Step 5 (instrumentation)
   - Option C: Optimize Step 3 (sliding window, unpack kernel)

### If Step 3 Fails:
1. Check for TCPX errors in logs
2. Verify Phase 1 still works
3. Test with smaller test size
4. Add more debug logging

---

## ⚠️ Known Limitations

### 1. Synchronous Completion
- **Issue**: Wait for each chunk before sending next
- **Impact**: May not fully saturate NICs
- **Fix**: Add sliding window if bandwidth is low

### 2. No Unpack Kernel
- **Issue**: Direct receive (no scatter-gather)
- **Impact**: May not work if TCPX requires unpack
- **Fix**: Add unpack kernel if needed

### 3. No Flow Control
- **Issue**: No sliding window
- **Impact**: N/A (synchronous prevents overflow)
- **Fix**: Add if we go async

---

## 📊 Progress Status

| Step | Status | Time Spent | Remaining |
|------|--------|------------|-----------|
| Step 2.5: Devmem Validation | ✅ Complete | 1 day | - |
| Step 2: Control Plane | ✅ Complete | 0.5 day | - |
| Phase 1: Round-Robin NIC | ✅ Complete | 0.5 day | - |
| **Step 3: Data Plane** | ✅ **Complete** | **0.5 day** | **-** |
| Step 4: Thread Affinity | ⏳ Not Started | - | 0.5-1 day |
| Step 5: Instrumentation | ⏳ Not Started | - | 0.5-1 day |
| Step 6: Validation | ⏳ Not Started | - | 1-2 days |

**Total Progress**: 4/7 steps complete (57%)  
**Estimated Remaining**: 2-4 days work, 3-6 days calendar

---

## ⚠️ Important Note on Bandwidth Expectations

**After Bug Fixes**:
- Bandwidth measurements will be **accurate** (not inflated)
- Each GPU now transfers **full test_size** (not shared)
- Total data: **test_size × 8 GPUs** (e.g., 64 MB × 8 = 512 MB)
- Comparable to baseline multi-process (apples-to-apples)

**Expected Results**:
- With 4 ch/GPU: **2-5 GB/s** (realistic, not 10+ GB/s)
- With 8 ch/GPU: **5-10 GB/s** (realistic, not 20+ GB/s)
- Baseline: **2.75 GB/s** (multi-process)
- Target: **2-4× improvement** over baseline

**See**: `STEP3_BUGFIXES.md` for detailed explanation

---

**Status**: ✅ Step 3 complete with bug fixes, ready to test on GCP
**Next Action**: Run `test_step3_bandwidth.sh` on both nodes and measure **accurate** bandwidth

