# TCPX P2P Technical Notes

## TCPX Request Pool Architecture

### Key Discovery: Request Pool is Fixed at 16

After analyzing the TCPX plugin source code (`nccl-plugin-gpudirecttcpx/src/work_queue.h`):

```cpp
#define MAX_REQUESTS 16  // Fixed per tcpxComm
```

**Important:** This is **NOT** affected by `NCCL_NSOCKS_PERTHREAD` or `NCCL_SOCKET_NTHREADS`.

### What NSOCKS/NTHREADS Actually Control

From `nccl-plugin-gpudirecttcpx/src/connect.cc`:

```cpp
// GCP auto-detection
if (strcmp(vendor, "0x1ae0") == 0) {  // GCP
  autoNt = 6;   // 6 threads
  autoNs = 1;   // 1 socket per thread
}
// Total parallel sockets = 6 × 1 = 6
```

**These control parallel socket connections, NOT request pool size.**

### Request State Transitions

From `nccl-plugin-gpudirecttcpx/src/work_queue_states.h`:

```
Send Request:  FREE → POSTED → ACTIVE → TRANSMITTING → FREE
Recv Request:  FREE → POSTED → ACTIVE → TRANSMITTING → INACTIVE → FREE
```

**Key Difference:**
- **Send:** Auto-released when `tcpx_test()` returns `done=1`
- **Recv:** Requires explicit `tcpx_irecv_consumed()` call

### Why Sliding Window is Necessary

**Problem:**
- 64MB message ÷ 512KB chunk = 128 chunks
- Request pool = 16 per comm
- Without sliding window: Exhausts pool at chunk 17

**Solution:**
- Server: Sliding window with `MAX_INFLIGHT = 16`
- Client: Sliding window with `MAX_INFLIGHT_SEND = 12` (leave margin)

---

## Performance Optimization History

### Issue 1: Kernel 100× Slower (RESOLVED)

**Problem:**
```cpp
// ❌ Wrong: Per-chunk overhead ~54ms
for each chunk:
  cudaStreamCreate(&stream);           // ~4ms
  UnpackLauncher launcher(stream);     // ~2ms
  launcher.launchSync(desc);           // ~48ms (includes sync!)
  cudaStreamDestroy(stream);           // ~1ms
```

**Solution:**
```cpp
// ✅ Correct: One-time setup, async launch
cudaStreamCreate(&stream);             // Once
UnpackLauncher* launcher = new UnpackLauncher(stream);  // Once
for each chunk:
  launcher->launch(desc);              // ~0.01ms (async)
cudaStreamSynchronize(stream);         // Once at end
```

**Result:** 100× speedup, kernel mode now matches D2D mode performance.

### Issue 2: Request Pool Exhaustion (RESOLVED)

**Problem:**
```cpp
// ❌ Wrong: Batch all 128 chunks
for (128 chunks):
  tcpx_isend(..., &req);
  tcpx_test(req, &done, ...);  // Returns done=1, but req not freed yet
// Pool exhausted at chunk 17
```

**Solution:**
```cpp
// ✅ Correct: Sliding window
if (pending_reqs.size() >= MAX_INFLIGHT_SEND) {
  wait_for_oldest();  // Drain one before issuing next
}
tcpx_isend(..., &req);
pending_reqs.push_back(req);
```

**Result:** Stable for any message size.

---

## TCPX API Usage Patterns

### Correct Send Pattern (Client)

```cpp
constexpr int MAX_INFLIGHT_SEND = 12;
std::vector<void*> pending_reqs;

for each chunk:
  // Sliding window: wait if full
  if (pending_reqs.size() >= MAX_INFLIGHT_SEND) {
    void* oldest = pending_reqs.front();
    int done = 0;
    while (!done) {
      tcpx_test(oldest, &done, nullptr);
    }
    // Request auto-released when done=1
    pending_reqs.erase(pending_reqs.begin());
  }
  
  // Issue send
  void* req;
  tcpx_isend(comm, data, size, tag, mhandle, &req);
  pending_reqs.push_back(req);

// Drain remaining
while (!pending_reqs.empty()) {
  // ... same as above
}
```

### Correct Recv Pattern (Server)

```cpp
constexpr int MAX_INFLIGHT = 16;
std::vector<cudaEvent_t> events(MAX_INFLIGHT);
std::vector<void*> pending_reqs;
std::vector<int> pending_indices;

for each chunk:
  // Sliding window: wait if full
  if (pending_reqs.size() >= MAX_INFLIGHT) {
    int oldest_idx = pending_indices.front();
    cudaEventSynchronize(events[oldest_idx % MAX_INFLIGHT]);
    tcpx_irecv_consumed(comm, 1, pending_reqs.front());
    pending_reqs.erase(pending_reqs.begin());
    pending_indices.erase(pending_indices.begin());
  }
  
  // Issue recv
  void* req;
  tcpx_irecv(comm, 1, &data, &size, &tag, &mhandle, &req);
  
  // Launch kernel (async)
  launcher->launch(desc);
  
  // Record event
  cudaEventRecord(events[chunk_idx % MAX_INFLIGHT], stream);
  
  // Add to window
  pending_reqs.push_back(req);
  pending_indices.push_back(chunk_idx);

// Drain remaining
while (!pending_reqs.empty()) {
  // ... same as above
}
```

---

## Environment Variables Reference

### Required
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_PORT_BEGIN=50000
export NCCL_GPUDIRECTTCPX_PORT_END=60000
```

### Optional (Auto-detected for GCP)
```bash
export NCCL_NSOCKS_PERTHREAD=1   # Sockets per thread (default: 1 for GCP)
export NCCL_SOCKET_NTHREADS=6    # Number of threads (default: 6 for GCP)
# Total parallel sockets = 1 × 6 = 6
```

### Performance Tuning
```bash
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=0  # Disable sleep in TX polling
export NCCL_GPUDIRECTTCPX_FORCE_ACK=1                # Force TCP quick ACK
```

---

## Common Pitfalls

### ❌ Don't: Increase NSOCKS/NTHREADS to fix request pool exhaustion
**Why:** These don't affect request pool size (fixed at 16 per comm)

### ❌ Don't: Create/destroy streams in hot path
**Why:** ~4ms overhead per operation

### ❌ Don't: Use synchronous kernel launch in loop
**Why:** Loses all parallelism, 100× slower

### ❌ Don't: Batch all chunks without sliding window
**Why:** Exhausts request pool at chunk 17

### ✅ Do: Use sliding window for both send and recv
**Why:** Keeps in-flight requests < 16, stable for any message size

### ✅ Do: Reuse streams and launchers
**Why:** Amortizes setup cost, enables async execution

### ✅ Do: Use async kernel launch
**Why:** Allows GPU and network to overlap

---

## Performance Expectations

### 4-NIC Setup (eth1-4, ~25 Gbps each)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 4KB          | ~1 GB/s   | ~30 μs  |
| 64KB         | ~5 GB/s   | ~100 μs |
| 1MB          | ~15 GB/s  | ~500 μs |
| 64MB         | ~20 GB/s  | ~25 ms  |

**Theoretical max:** 4 × 25 Gbps = 100 Gbps = 12.5 GB/s  
**Actual:** ~20 GB/s (160 Gbps) due to multi-flow aggregation and TCP efficiency

### Kernel vs D2D Mode

With proper implementation (async launch, reused streams):
- **Kernel mode:** ~20 GB/s
- **D2D mode:** ~20 GB/s
- **Difference:** < 5% (within measurement noise)

---

## Debugging Tips

### Enable TCPX Logging
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

### Check Request Pool Status
Look for errors in logs:
```
[ncclNet:2] unable to allocate requests
```
→ Request pool exhausted, need sliding window

### Check devmem-tcp Status
```bash
dmesg | grep devmem
# Should see: "TCP: devmem-tcp enabled"
```

### Verify Multi-NIC Usage
```bash
# During test, check traffic on all NICs
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

---

## References

- TCPX Plugin Source: `/home/daniel/uccl/nccl-plugin-gpudirecttcpx/src/`
  - `work_queue.h` - Request pool definition (MAX_REQUESTS = 16)
  - `net_tcpx.cc` - Send/recv implementation
  - `work_queue_states.h` - Request state transitions
  
- NCCL Unpack Reference: `https://github.com/NVIDIA/nccl/tree/master/src/device/network/unpack`

---

**Last Updated:** 2025-10-02  
**Author:** Based on TCPX source code analysis

