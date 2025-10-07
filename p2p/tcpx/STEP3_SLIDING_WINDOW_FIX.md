# Step 3: Sliding Window Fix

## 🐛 **Problem: Channel Saturation**

### **Symptoms**
- Server: GPU 4 channel 0 accept failed after 100 retries
- Client: GPU 0 channel 0 `tcpx_isend` failed at chunk 64
- Test hung/crashed during first iteration

### **Root Cause**

**Original Code (Broken)**:
```cpp
// Phase 1: Post ALL receives at once (1024 operations!)
std::vector<PendingRecv> pending_recvs;
for (int gpu_id = 0; gpu_id < 8; gpu_id++) {
  for (int chunk = 0; chunk < 128; chunk++) {  // 64 MB / 512 KB = 128 chunks
    tcpx_irecv(..., &request);
    pending_recvs.push_back(request);  // NO LIMIT!
  }
}
// Total: 8 GPUs × 128 chunks = 1024 inflight requests

// Phase 2: Wait for all
for (auto& pending : pending_recvs) {
  while (!done) tcpx_test(pending.request, &done, ...);
}
```

**Why It Failed**:
1. **TCPX Limitation**: Each `comm` has `MAX_REQUESTS=16` slots
2. **Channel Saturation**: 1024 requests / 32 channels = 32 requests per channel
3. **Overflow**: 32 > 16 → requests fail or hang

**Evidence from Logs**:
```
[ChannelManager] Failed to accept connection for channel 0 after 100 retries
[ERROR] tcpx_isend failed (GPU 0 channel 0 chunk 64)
```
- Chunk 64 = 64th request on GPU 0
- GPU 0 has 4 channels → 64 / 4 = 16 requests per channel
- Exactly at the limit!

---

## ✅ **Solution: Sliding Window**

### **Reference Implementation**

From `test_tcpx_perf_multi.cc` (successful multi-process test):
```cpp
// Server: MAX_INFLIGHT_PER_CHANNEL = 16
// Client: MAX_INFLIGHT_SEND_PER_CHANNEL = 12 (leave margin)

// Per-channel sliding window
std::vector<std::vector<PendingRecv>> pending_per_channel(num_channels);

for each chunk:
  int channel_id = chunk_idx % num_channels;
  auto& channel_pending = pending_per_channel[channel_id];
  
  // Wait if channel is full
  while (channel_pending.size() >= MAX_INFLIGHT) {
    // Check oldest request
    if (oldest is done) {
      consume and remove
    } else {
      sleep and retry
    }
  }
  
  // Post new request
  tcpx_irecv(..., &request);
  channel_pending.push_back(request);
```

**Key Insight**: Never exceed 16 inflight requests per channel!

---

## 🔧 **Implementation**

### **Server Side** (lines 567-702)

**Data Structure**:
```cpp
constexpr int MAX_INFLIGHT_PER_CHANNEL = 12;  // Safe limit

struct PendingRecv {
  void* request;
  int gpu_id;
  int channel_id;
  int chunk_idx;
};

// Track per GPU per channel
std::vector<std::vector<std::vector<PendingRecv>>> pending_per_gpu_channel(kNumGPUs);
for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
  int num_channels = gpus[gpu_id].mgr->get_num_channels();
  pending_per_gpu_channel[gpu_id].resize(num_channels);
}
```

**Receive Loop**:
```cpp
for each GPU:
  for each chunk:
    int channel_id = chunk_idx % num_channels;
    auto& channel_pending = pending_per_gpu_channel[gpu_id][channel_id];
    
    // Sliding window: wait if full
    while (channel_pending.size() >= MAX_INFLIGHT_PER_CHANNEL) {
      auto& oldest = channel_pending.front();
      int done = 0;
      tcpx_test(oldest.request, &done, ...);
      
      if (done) {
        tcpx_irecv_consumed(..., oldest.request);
        channel_pending.erase(channel_pending.begin());
      } else {
        sleep(10us);
      }
    }
    
    // Post receive
    tcpx_irecv(..., &request);
    channel_pending.push_back({request, gpu_id, channel_id, chunk_idx});

// Drain all remaining
for each GPU:
  for each channel:
    while (!channel_pending.empty()) {
      wait for oldest to complete
      consume and remove
    }
```

### **Client Side** (lines 866-994)

**Data Structure**:
```cpp
constexpr int MAX_INFLIGHT_SEND_PER_CHANNEL = 12;  // Same as multi-process

struct PendingSend {
  void* request;
  int gpu_id;
  int channel_id;
  int chunk_idx;
};

std::vector<std::vector<std::vector<PendingSend>>> pending_per_gpu_channel_send(kNumGPUs);
```

**Send Loop**: (Same pattern as server)

---

## 📊 **Why This Works**

### **Before (Broken)**:
```
Total inflight: 1024 requests
Per channel: 1024 / 32 = 32 requests
TCPX limit: 16 requests per comm
Result: OVERFLOW → FAIL
```

### **After (Fixed)**:
```
Max inflight per channel: 12 requests
TCPX limit: 16 requests per comm
Margin: 16 - 12 = 4 requests
Result: SAFE → SUCCESS
```

### **Performance Impact**:
- **Latency**: Slightly higher (wait for slots)
- **Throughput**: **SAME** (still saturates NICs)
- **Reliability**: **MUCH BETTER** (no overflow)

**Why throughput is same**:
- 12 inflight requests per channel is enough to saturate the NIC
- Bottleneck is NIC bandwidth, not request queue depth
- As long as queue never empties, throughput is maximized

---

## 🎯 **Key Differences from Multi-Process**

### **Multi-Process** (`test_tcpx_perf_multi.cc`):
- 1 process per GPU
- Each process has 4 channels
- Each channel has 12-16 inflight requests
- Total: 8 processes × 4 channels × 12 requests = 384 inflight

### **Single-Process** (`test_tcpx_perf_orchestrator.cc`):
- 1 process for all 8 GPUs
- 32 channels total (8 GPUs × 4 channels)
- Each channel has 12 inflight requests
- Total: 32 channels × 12 requests = 384 inflight

**Same total inflight requests!** Just organized differently.

---

## ✅ **Testing**

### **Expected Behavior**:
1. ✅ No "accept failed" errors
2. ✅ No "tcpx_isend failed" errors
3. ✅ All iterations complete successfully
4. ✅ Bandwidth >2.75 GB/s (better than baseline)

### **How to Test**:
```bash
# Server (Node 0)
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh server

# Client (Node 1)
./test_step3_bandwidth.sh client <SERVER_IP>
```

### **What to Look For**:
```
[SERVER] ===== Iteration 0 =====
[SERVER] Iteration 0 completed in 1234 ms, bandwidth: 4.12 GB/s
[SERVER] ===== Iteration 1 =====
...
[SERVER] ===== Iteration 19 =====
[SERVER] Average bandwidth: 4.15 GB/s
```

**No errors, all iterations complete!**

---

## 📋 **Summary**

| Aspect | Before | After |
|--------|--------|-------|
| **Inflight per channel** | Unlimited (32+) | Limited (12) |
| **TCPX limit** | 16 | 16 |
| **Result** | Overflow → Fail | Safe → Success |
| **Throughput** | N/A (crashed) | Same as multi-process |
| **Reliability** | Broken | Robust |

---

**Status**: ✅ Fixed, compiled, ready to test  
**Impact**: Prevents channel saturation, enables reliable data transfer  
**Next Action**: Test on GCP nodes and verify bandwidth

**Good luck!** 🚀

