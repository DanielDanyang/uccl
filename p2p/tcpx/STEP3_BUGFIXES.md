# Step 3 Bug Fixes

**Date**: 2025-10-07  
**Status**: ✅ Fixed and Compiled

---

## 🐛 Bugs Fixed

### Bug 1: Incorrect Bandwidth Calculation (CRITICAL) ✅

**Problem**:
- Original code had all GPUs **share** `test_size` bytes per iteration
- Each GPU only transferred `test_size / 8` bytes
- Bandwidth was calculated as `test_size / time`, which was **8× too optimistic**

**Example**:
```
test_size = 64 MB
8 GPUs × (64 MB / 8) = 64 MB total (not 512 MB!)
Reported: 10 GB/s
Actual: 1.25 GB/s (8× lower)
```

**Root Cause**:
```cpp
// OLD CODE (WRONG):
size_t offset = 0;
while (offset < test_size) {  // All GPUs share this offset
  int gpu_id = (chunk_idx % total_channels) / num_channels_per_gpu;
  // Send chunk from gpu_id at offset
  offset += chunk_size;  // Shared offset advances
}
// Result: Only test_size bytes total across all GPUs
```

**Fix**:
```cpp
// NEW CODE (CORRECT):
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  size_t offset = 0;  // Per-GPU offset
  while (offset < test_size_per_gpu) {  // Each GPU transfers full test_size_per_gpu
    // Send chunk from gpu_id at offset
    offset += chunk_size;  // Per-GPU offset advances
  }
}
// Result: test_size_per_gpu × 8 GPUs = total bytes transferred
```

**Impact**:
- ✅ Bandwidth now correctly reflects actual throughput
- ✅ Matches baseline multi-process semantics (each GPU transfers full test_size)
- ✅ Fair comparison with NCCL and other benchmarks

---

### Bug 2: Fixed Buffer Size Overflow (HIGH) ✅

**Problem**:
- Buffer size was hardcoded to 64 MB + 4 KB
- `UCCL_TCPX_PERF_SIZE` could be set higher (e.g., 128 MB)
- Would access unregistered memory → TCPX errors or crashes

**Root Cause**:
```cpp
// OLD CODE (WRONG):
constexpr size_t kRegisteredBytes = 64 * 1024 * 1024 + 4096;  // Fixed 64 MB

// Later:
size_t test_size = getEnvSize("UCCL_TCPX_PERF_SIZE", 67108864);  // Could be 128 MB!
// If test_size > kRegisteredBytes → buffer overflow
```

**Fix**:
```cpp
// NEW CODE (CORRECT):
constexpr size_t kDefaultTransferSize = 64 * 1024 * 1024;  // Default
constexpr size_t kMaxTransferSize = 256 * 1024 * 1024;     // Max allowed
constexpr size_t kRegisteredBytes = kMaxTransferSize + 4096;  // Allocate max

// Validate at runtime:
size_t test_size_per_gpu = getEnvSize("UCCL_TCPX_PERF_SIZE", kDefaultTransferSize);
if (test_size_per_gpu > kMaxTransferSize) {
  std::cerr << "[ERROR] UCCL_TCPX_PERF_SIZE exceeds max buffer size" << std::endl;
  return 1;
}
```

**Impact**:
- ✅ Supports test sizes up to 256 MB per GPU
- ✅ Validates at runtime to prevent overflow
- ✅ Clear error message if user exceeds limit

---

### Bug 3: Assumed Uniform Channel Count (MEDIUM) ✅

**Problem**:
- Code assumed every GPU has exactly `num_channels_per_gpu` channels
- If `ChannelManager` trims channels (e.g., NIC probe failure), would crash
- `get_channel(channel_local_id)` would access out-of-bounds

**Root Cause**:
```cpp
// OLD CODE (WRONG):
int total_channels = kNumGPUs * num_channels_per_gpu;  // Assumes all GPUs have same count

int channel_global_id = chunk_idx % total_channels;
int gpu_id = channel_global_id / num_channels_per_gpu;  // Assumes uniform distribution
int channel_local_id = channel_global_id % num_channels_per_gpu;

ChannelResources& ch = ctx.mgr->get_channel(channel_local_id);  // May be out of bounds!
```

**Fix**:
```cpp
// NEW CODE (CORRECT):
// Calculate actual total channels
int total_channels = 0;
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  total_channels += gpus[gpu_id].mgr->get_num_channels();  // Query actual count
}

// Per-GPU loop with actual channel count
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  int num_channels = ctx.mgr->get_num_channels();  // Actual count for this GPU
  int channel_local_id = chunk_idx % num_channels;  // Safe modulo
  ChannelResources& ch = ctx.mgr->get_channel(channel_local_id);  // Always in bounds
}
```

**Impact**:
- ✅ Robust to per-GPU channel count variations
- ✅ No crashes if ChannelManager trims channels
- ✅ Correct total channel count reporting

---

## 📊 Code Changes Summary

### Modified Lines

**Constants** (lines 117-120):
```cpp
// Before:
constexpr size_t kTransferSize = 64 * 1024 * 1024;
constexpr size_t kRegisteredBytes = kTransferSize + 4096;

// After:
constexpr size_t kDefaultTransferSize = 64 * 1024 * 1024;
constexpr size_t kMaxTransferSize = 256 * 1024 * 1024;
constexpr size_t kRegisteredBytes = kMaxTransferSize + 4096;
```

**Server Initialization** (lines 530-557):
```cpp
// Added:
- Validation: test_size_per_gpu <= kMaxTransferSize
- Actual total_channels calculation (query each GPU)
- Better logging (per-GPU and total test size)
```

**Server Transfer Loop** (lines 561-636):
```cpp
// Changed:
- Outer loop: for each GPU
- Inner loop: for each chunk within GPU (per-GPU offset)
- Tag calculation: includes gpu_id to avoid collisions
- Bandwidth: total_bytes = test_size_per_gpu × kNumGPUs
```

**Client Initialization** (lines 757-788):
```cpp
// Added:
- Validation: test_size_per_gpu <= kMaxTransferSize
- Actual total_channels calculation (query each GPU)
- Better logging (per-GPU and total test size)
```

**Client Transfer Loop** (lines 792-859):
```cpp
// Changed:
- Outer loop: for each GPU
- Inner loop: for each chunk within GPU (per-GPU offset)
- Tag calculation: includes gpu_id to avoid collisions
- Bandwidth: total_bytes = test_size_per_gpu × kNumGPUs
```

---

## 🧪 Testing Impact

### Before Fix (WRONG):
```
Test size: 64 MB (shared across 8 GPUs)
Each GPU transfers: 8 MB
Total transferred: 64 MB
Reported bandwidth: 10 GB/s (for 64 MB)
Actual bandwidth: 1.25 GB/s (for 8 MB per GPU)
```

### After Fix (CORRECT):
```
Test size per GPU: 64 MB
Each GPU transfers: 64 MB
Total transferred: 512 MB (64 MB × 8 GPUs)
Reported bandwidth: 10 GB/s (for 512 MB)
Actual bandwidth: 10 GB/s (correct!)
```

### Comparison with Baseline:
```
Baseline (multi-process):
  - 8 processes × 64 MB each = 512 MB total
  - Bandwidth = 512 MB / time

New (single-process, after fix):
  - 8 GPUs × 64 MB each = 512 MB total
  - Bandwidth = 512 MB / time

✅ Now apples-to-apples comparison!
```

---

## ✅ Verification

### Build Status:
```bash
$ make test_tcpx_perf_orchestrator
✅ Compiled successfully with no warnings or errors
```

### Expected Behavior:
1. **Bandwidth is now accurate** (not 8× inflated)
2. **Buffer overflow prevented** (validates test_size <= 256 MB)
3. **Robust to channel variations** (queries actual channel count)

### Expected Output:
```
[SERVER] Test size per GPU: 67108864 bytes (64 MB)
[SERVER] Total test size: 536870912 bytes (512 MB)  ← 8× larger than before
[SERVER] Iterations: 20
[SERVER] Chunk size: 524288 bytes
[SERVER] Total channels: 32

[SERVER] ===== Iteration 0 =====
[SERVER] Iteration 0 completed in XXX ms, bandwidth: X.XX GB/s  ← Accurate now

[SERVER] ===== Performance Summary =====
[SERVER] Average bandwidth: X.XX GB/s  ← Comparable to baseline
```

---

## 📈 Performance Expectations (Updated)

### Before Fix (Inflated):
```
Reported: 10 GB/s
Actual: 1.25 GB/s (8× lower)
```

### After Fix (Accurate):
```
With 4 channels/GPU (32 total):
  Target: >5 GB/s (realistic)
  
With 8 channels/GPU (64 total):
  Target: >10 GB/s (realistic)
```

### Comparison:
```
Baseline (multi-process): 2.75 GB/s
Target (single-process):  >5 GB/s (2× improvement)
NCCL (reference):         19.176 GB/s
```

---

## 🎯 Next Steps

### Testing:
1. Run test with 4 channels/GPU
2. Verify bandwidth is **lower** than before (but accurate)
3. Compare with baseline (should be 2-4× improvement)
4. Scale to 8 channels/GPU if successful

### Expected Results:
- Bandwidth will be **~8× lower** than previous (wrong) measurements
- But now **accurate** and **comparable** to baseline
- Should still see **2-4× improvement** over 2.75 GB/s baseline

---

**Status**: ✅ All bugs fixed, compiled, ready to test  
**Impact**: Bandwidth measurements now accurate and comparable to baseline  
**Next Action**: Re-run tests and expect realistic (lower) bandwidth numbers

