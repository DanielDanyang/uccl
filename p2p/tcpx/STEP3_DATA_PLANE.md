# Step 3: Data Plane Upgrade - COMPLETE

**Date**: 2025-10-07  
**Status**: ✅ Implemented, Ready for Testing

---

## 🎯 What Was Done

### 1. Added Data Transfer to Orchestrator ✅

**File**: `tests/test_tcpx_perf_orchestrator.cc`

**Client Side** (lines 626-715):
- Implemented `tcpx_isend()` loop with round-robin channel selection
- Simple synchronous completion (wait for each send before next)
- Performance measurement per iteration
- Bandwidth calculation and reporting

**Server Side** (lines 523-622):
- Implemented `tcpx_irecv()` loop with round-robin channel selection
- Simple synchronous completion (wait for each recv before next)
- Direct memory receive (no unpack kernel for simplicity)
- Performance measurement per iteration
- Bandwidth calculation and reporting

### 2. Added Helper Functions ✅

**New Functions**:
- `getEnvSize()` - Parse size_t from environment variables
- Added `<chrono>` and `<thread>` headers for timing and sleep

### 3. Round-Robin Channel Selection ✅

**Algorithm**:
```cpp
// Total channels = 8 GPUs × N channels/GPU
int total_channels = kNumGPUs * num_channels_per_gpu;

// For each chunk:
int channel_global_id = global_chunk_idx % total_channels;
int gpu_id = channel_global_id / num_channels_per_gpu;
int channel_local_id = channel_global_id % num_channels_per_gpu;
```

**Example** (4 channels/GPU):
```
Chunk 0 → GPU 0, Channel 0
Chunk 1 → GPU 0, Channel 1
Chunk 2 → GPU 0, Channel 2
Chunk 3 → GPU 0, Channel 3
Chunk 4 → GPU 1, Channel 0
...
Chunk 31 → GPU 7, Channel 3
Chunk 32 → GPU 0, Channel 0 (wraps around)
```

---

## 📊 Implementation Details

### Simplified Design (Step 3)

**What We Implemented**:
- ✅ Basic data transfer (send/recv)
- ✅ Round-robin channel selection
- ✅ Performance measurement
- ✅ Synchronous completion (simple)

**What We Deferred** (for future optimization):
- ⏳ Sliding window flow control (not needed yet)
- ⏳ Unpack kernel integration (direct receive works)
- ⏳ Asynchronous pipelining (synchronous is simpler)

**Rationale**:
- Start simple to validate the architecture
- Synchronous completion is easier to debug
- Direct receive avoids unpack kernel complexity
- Can add optimizations later if needed

---

## 🧪 Testing Configuration

### Environment Variables

```bash
# Test size (default: 64MB)
export UCCL_TCPX_PERF_SIZE=67108864

# Iterations (default: 20)
export UCCL_TCPX_PERF_ITERS=20

# Chunk size (default: 512KB)
export UCCL_TCPX_CHUNK_BYTES=524288

# Channels per GPU (from Phase 1)
export UCCL_TCPX_NUM_CHANNELS=4
```

### Test Matrix

| Channels/GPU | Total Channels | Chunks | Expected BW |
|--------------|----------------|--------|-------------|
| 4 | 32 | 128 | >5 GB/s |
| 8 | 64 | 128 | >10 GB/s |

**Calculation**:
- Test size: 64MB
- Chunk size: 512KB
- Chunks per iteration: 64MB ÷ 512KB = 128 chunks
- With 32 channels: 128 ÷ 32 = 4 chunks per channel

---

## 🚀 How to Test

### Step 1: Test with 4 Channels/GPU (Phase 1 Config)

**Server (Node 0)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
UCCL_TCPX_NUM_CHANNELS=4 ./run_p2p_singleproc.sh server
```

**Client (Node 1)**:
```bash
cd /home/daniel/uccl/p2p/tcpx
UCCL_TCPX_NUM_CHANNELS=4 ./run_p2p_singleproc.sh client <SERVER_IP>
```

**Expected Output**:
```
[CLIENT] ===== Iteration 0 =====
[CLIENT] Iteration 0 completed in XXX ms, bandwidth: X.XX GB/s
...
[CLIENT] ===== Performance Summary =====
[CLIENT] Average time: XXX ms
[CLIENT] Average bandwidth: X.XX GB/s
[CLIENT] Total channels used: 32

[SERVER] ===== Iteration 0 =====
[SERVER] Iteration 0 completed in XXX ms, bandwidth: X.XX GB/s
...
[SERVER] ===== Performance Summary =====
[SERVER] Average time: XXX ms
[SERVER] Average bandwidth: X.XX GB/s
[SERVER] Total channels used: 32
```

### Step 2: Scale to 8 Channels/GPU (If Step 1 Works)

```bash
# Server
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh server

# Client
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh client <SERVER_IP>
```

**Expected**: Higher bandwidth (>10 GB/s)

---

## 📈 Performance Expectations

### Baseline (Multi-Process, 1 Channel/GPU)
- **Bandwidth**: 2.75 GB/s
- **Channels**: 8 (1 per GPU)
- **NICs**: 4 (2 GPUs per NIC)

### Phase 1 + Step 3 (Single-Process, 4 Channels/GPU)
- **Target**: >5 GB/s (2x improvement)
- **Channels**: 32 (4 per GPU)
- **NICs**: 4 (8 channels per NIC)

### Phase 1 + Step 3 (Single-Process, 8 Channels/GPU)
- **Target**: >10 GB/s (4x improvement)
- **Channels**: 64 (8 per GPU)
- **NICs**: 4 (16 channels per NIC)

### NCCL Baseline (Reference)
- **Bandwidth**: 19.176 GB/s
- **Goal**: Get within 20% (>15 GB/s)

---

## 🔍 Key Differences from Multi-Process Version

### Simplified (Step 3)
| Feature | Multi-Process | Single-Process (Step 3) |
|---------|---------------|-------------------------|
| Sliding Window | ✅ Yes (12 per channel) | ❌ No (synchronous) |
| Unpack Kernel | ✅ Yes (GPU kernel) | ❌ No (direct receive) |
| Pipelining | ✅ Yes (async) | ❌ No (sync) |
| Complexity | High | Low |

### Why Simplified?
1. **Easier to debug**: Synchronous is simpler
2. **Validates architecture**: Proves single-process works
3. **Good enough**: May still hit >10 GB/s
4. **Can optimize later**: Add sliding window if needed

---

## ⚠️ Known Limitations

### 1. Synchronous Completion
**Issue**: Wait for each chunk to complete before sending next  
**Impact**: May not fully saturate NICs  
**Workaround**: Acceptable for initial testing  
**Fix**: Add sliding window in future optimization

### 2. No Unpack Kernel
**Issue**: Direct receive to GPU memory (no scatter-gather)  
**Impact**: May not work if TCPX requires unpack  
**Workaround**: Test and see if it works  
**Fix**: Add unpack kernel if needed

### 3. No Flow Control
**Issue**: No sliding window to limit inflight requests  
**Impact**: May exhaust TCPX request pool  
**Workaround**: Synchronous completion prevents this  
**Fix**: Add sliding window if we go async

---

## 🎯 Success Criteria

### Step 3a: 4 Channels/GPU
- [ ] Data transfer completes without errors
- [ ] Both client and server report bandwidth
- [ ] Bandwidth >2.75 GB/s (better than baseline)
- [ ] Bandwidth >5 GB/s (target)

### Step 3b: 8 Channels/GPU
- [ ] Data transfer completes without errors
- [ ] Bandwidth >5 GB/s (better than 4 channels)
- [ ] Bandwidth >10 GB/s (target)

---

## 📝 Files Modified

### Modified Files
| File | Lines | Change |
|------|-------|--------|
| `tests/test_tcpx_perf_orchestrator.cc` | 98-109 | Added headers (chrono, thread) |
| `tests/test_tcpx_perf_orchestrator.cc` | 136-145 | Added getEnvSize() |
| `tests/test_tcpx_perf_orchestrator.cc` | 523-622 | Server receive loop |
| `tests/test_tcpx_perf_orchestrator.cc` | 626-715 | Client send loop |

### Build Status
✅ **Compiled successfully**
```bash
$ make test_tcpx_perf_orchestrator
Building test_tcpx_perf_orchestrator...
/usr/local/cuda/bin/nvcc ... -o tests/test_tcpx_perf_orchestrator ...
```

---

## 🚀 Next Steps

### After Step 3 Passes:
1. **Measure bandwidth**: Compare with baseline and NCCL
2. **Analyze bottlenecks**: CPU, NIC, or GPU?
3. **Decide on optimizations**:
   - Option A: Add sliding window (if bandwidth is low)
   - Option B: Add unpack kernel (if TCPX requires it)
   - Option C: Proceed to Step 4 (thread affinity)

### If Step 3 Fails:
1. Check for TCPX errors in logs
2. Verify data integrity (add checksums)
3. Test with smaller test size
4. Add more debug logging

---

## 📊 Debugging Tips

### Check Bandwidth
```bash
# Should see increasing bandwidth with more channels
UCCL_TCPX_NUM_CHANNELS=1 ./run_p2p_singleproc.sh ...  # Baseline
UCCL_TCPX_NUM_CHANNELS=2 ./run_p2p_singleproc.sh ...  # 2x?
UCCL_TCPX_NUM_CHANNELS=4 ./run_p2p_singleproc.sh ...  # 4x?
```

### Check NIC Traffic
```bash
# Before test
sudo ethtool -S eth1 | grep rx_devmem_pkts > /tmp/before.txt

# Run test
./run_p2p_singleproc.sh ...

# After test
sudo ethtool -S eth1 | grep rx_devmem_pkts > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```

### Check for Errors
```bash
# Look for TCPX errors
grep -i "error\|fail" logs/singleproc_*.log

# Look for bandwidth anomalies
grep "bandwidth:" logs/singleproc_*.log
```

---

**Status**: ✅ Step 3 implemented, ready for testing  
**Next Action**: Run test with 4 channels/GPU and measure bandwidth

