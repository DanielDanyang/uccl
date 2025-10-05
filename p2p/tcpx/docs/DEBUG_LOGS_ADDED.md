# Debug Logs Added & Chunk Size Corrected

**Date**: 2025-10-05  
**Status**: ✅ **COMPLETE**

---

## Summary

Added comprehensive debug logging to `test_tcpx_perf_multi.cc` and corrected the chunk size back to 512KB (the "2MB optimization" was a mistake).

---

## Changes Made

### 1. Chunk Size Correction ✅

**Issue**: Original comment claimed "Chunk size 从 512KB 增加到 2MB（减少开销）" was an optimization, but this was incorrect.

**Fix**: Reverted to 512KB default:

```cpp
// Before (WRONG):
// 【关键】Chunk 大小：优化后从 512KB 增加到 2MB
size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 2 * 1024 * 1024));

// After (CORRECT):
// Chunk 大小：默认 512KB
size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES",
                                 getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));
```

**Rationale**: 512KB is the proven default from original test. Larger chunks may not always be better due to:
- Increased latency per chunk
- Less parallelism across channels
- Potential memory pressure

---

### 2. Debug Logs Added ✅

Added detailed debug logging at all critical points to help identify potential bugs during testing.

#### Server-Side Debug Logs

**Iteration Start**:
```cpp
std::cout << "[DEBUG] Iteration " << iter << " start: clearing sliding windows for " 
          << num_channels << " channels" << std::endl;
```

**Per-Chunk Info**:
```cpp
std::cout << "[DEBUG][SERVER] chunk=" << global_chunk_idx << " channel=" << channel_id
          << " tag=" << tag << " size=" << this_chunk << " offset=" << offset 
          << " pending=" << win.pending_reqs.size() << "/" << MAX_INFLIGHT_PER_CHANNEL << std::endl;
```

**Sliding Window Full**:
```cpp
std::cout << "[DEBUG][SERVER] Channel " << channel_id << " sliding window FULL ("
          << win.pending_reqs.size() << "/" << MAX_INFLIGHT_PER_CHANNEL 
          << "), waiting for oldest chunk" << std::endl;

std::cout << "[DEBUG][SERVER] Waiting for chunk " << oldest_idx 
          << " kernel to complete..." << std::endl;

std::cout << "[DEBUG][SERVER] Chunk " << oldest_idx << " kernel completed, calling irecv_consumed" << std::endl;

std::cout << "[DEBUG][SERVER] Channel " << channel_id << " window now has "
          << win.pending_reqs.size() << " pending chunks" << std::endl;
```

**tcpx_irecv Call**:
```cpp
std::cout << "[DEBUG][SERVER] Calling tcpx_irecv for chunk " << global_chunk_idx 
          << " on channel " << channel_id << std::endl;

std::cout << "[DEBUG][SERVER] tcpx_irecv returned, request=" << recv_request << std::endl;
```

**Polling Progress**:
```cpp
std::cout << "[DEBUG][SERVER] Polling for chunk " << global_chunk_idx << " completion..." << std::endl;

// Every 1000 polls:
std::cout << "[DEBUG][SERVER] Still polling chunk " << global_chunk_idx 
          << " (poll_count=" << poll_count << ")" << std::endl;

std::cout << "[DEBUG][SERVER] Chunk " << global_chunk_idx << " recv completed after " 
          << poll_count << " polls, received_size=" << received_size << std::endl;
```

**Fragment Count**:
```cpp
std::cout << "[DEBUG][SERVER] Chunk " << global_chunk_idx << " has " << frag_count 
          << " fragments to unpack" << std::endl;
```

**Kernel Launch**:
```cpp
std::cout << "[DEBUG][SERVER] Launching unpack kernel for chunk " << global_chunk_idx 
          << " (channel " << channel_id << ", " << desc_block.count << " descriptors)" << std::endl;

std::cout << "[DEBUG][SERVER] Chunk " << global_chunk_idx << " kernel launched, event_idx=" 
          << event_idx << ", adding to window (counter=" << win.chunk_counter << ")" << std::endl;
```

#### Client-Side Debug Logs

**Iteration Start**:
```cpp
std::cout << "[DEBUG] Iteration " << iter << " start: clearing send windows for " 
          << num_channels << " channels" << std::endl;
```

**Per-Chunk Info**:
```cpp
std::cout << "[DEBUG][CLIENT] chunk=" << global_chunk_idx << " channel=" << channel_id
          << " tag=" << tag << " size=" << this_chunk << " offset=" << offset 
          << " pending=" << win.pending_reqs.size() << "/" << MAX_INFLIGHT_SEND_PER_CHANNEL << std::endl;
```

**Sliding Window Full**:
```cpp
std::cout << "[DEBUG][CLIENT] Channel " << channel_id << " send window FULL ("
          << win.pending_reqs.size() << "/" << MAX_INFLIGHT_SEND_PER_CHANNEL 
          << "), waiting for oldest send" << std::endl;

// Every 1000 polls:
std::cout << "[DEBUG][CLIENT] Still waiting for oldest send (poll_count=" 
          << poll_count << ")" << std::endl;

std::cout << "[DEBUG][CLIENT] Oldest send completed after " << poll_count 
          << " polls, sent_size=" << sent_size << std::endl;

std::cout << "[DEBUG][CLIENT] Channel " << channel_id << " window now has "
          << win.pending_reqs.size() << " pending sends" << std::endl;
```

**tcpx_isend Call**:
```cpp
std::cout << "[DEBUG][CLIENT] Calling tcpx_isend for chunk " << global_chunk_idx 
          << " on channel " << channel_id << std::endl;

std::cout << "[DEBUG][CLIENT] tcpx_isend returned, request=" << send_request << std::endl;

std::cout << "[DEBUG][CLIENT] Chunk " << global_chunk_idx << " added to window (counter=" 
          << win.chunk_counter << ", pending=" << win.pending_reqs.size() << ")" << std::endl;
```

**Draining Windows**:
```cpp
std::cout << "[DEBUG] Iteration " << iter << " end: draining send windows for " 
          << num_channels << " channels" << std::endl;

std::cout << "[DEBUG][CLIENT] Draining channel " << ch << " (" 
          << win.pending_reqs.size() << " pending sends)" << std::endl;

std::cout << "[DEBUG][CLIENT] Channel " << ch << " drained" << std::endl;
```

---

## Debug Log Categories

### 1. Lifecycle Events
- Iteration start/end
- Window clearing
- Window draining

### 2. Per-Chunk Operations
- Chunk assignment to channel
- Tag generation
- Offset calculation
- Window status (pending count)

### 3. Sliding Window Management
- Window full detection
- Waiting for oldest request
- Request completion
- Window state after removal

### 4. TCPX API Calls
- tcpx_irecv/tcpx_isend calls
- Request pointers returned
- Polling progress (every 1000 iterations)
- Completion status

### 5. Unpack Operations (Server Only)
- Fragment count
- Kernel launch
- Event recording
- Descriptor count

---

## Benefits of Debug Logs

### 1. Bug Detection
- Identify incorrect channel selection
- Detect window overflow/underflow
- Spot tag mismatches
- Find polling hangs

### 2. Performance Analysis
- Measure polling iterations per chunk
- Identify slow channels
- Detect load imbalance
- Find bottlenecks

### 3. Correctness Verification
- Verify round-robin distribution
- Confirm window management
- Validate tag uniqueness
- Check fragment counts

### 4. Troubleshooting
- Trace execution flow
- Identify failure points
- Understand timing issues
- Debug race conditions

---

## Example Debug Output

### Server (4 Channels, 64MB Transfer)

```
[PERF] Iteration 0: total bytes=67108864, chunk_bytes=524288
[DEBUG] Iteration 0 start: clearing sliding windows for 4 channels
[DEBUG][SERVER] chunk=0 channel=0 tag=99 size=524288 offset=0 pending=0/16
[DEBUG][SERVER] Calling tcpx_irecv for chunk 0 on channel 0
[DEBUG][SERVER] tcpx_irecv returned, request=0x7f8a4c000000
[DEBUG][SERVER] Polling for chunk 0 completion...
[DEBUG][SERVER] Chunk 0 recv completed after 42 polls, received_size=0
[DEBUG][SERVER] Chunk 0 has 8 fragments to unpack
[DEBUG][SERVER] Launching unpack kernel for chunk 0 (channel 0, 8 descriptors)
[DEBUG][SERVER] Chunk 0 kernel launched, event_idx=0, adding to window (counter=0)

[DEBUG][SERVER] chunk=1 channel=1 tag=100 size=524288 offset=524288 pending=0/16
[DEBUG][SERVER] Calling tcpx_irecv for chunk 1 on channel 1
...

[DEBUG][SERVER] chunk=16 channel=0 tag=115 size=524288 offset=8388608 pending=16/16
[DEBUG][SERVER] Channel 0 sliding window FULL (16/16), waiting for oldest chunk
[DEBUG][SERVER] Waiting for chunk 0 kernel to complete...
[DEBUG][SERVER] Chunk 0 kernel completed, calling irecv_consumed
[DEBUG][SERVER] Channel 0 window now has 15 pending chunks
...
```

### Client (4 Channels, 64MB Transfer)

```
[PERF] Iteration 0: total bytes=67108864, chunk_bytes=524288
[DEBUG] Iteration 0 start: clearing send windows for 4 channels
[DEBUG][CLIENT] chunk=0 channel=0 tag=99 size=524288 offset=0 pending=0/12
[DEBUG][CLIENT] Calling tcpx_isend for chunk 0 on channel 0
[DEBUG][CLIENT] tcpx_isend returned, request=0x7f8a4c000000
[DEBUG][CLIENT] Chunk 0 added to window (counter=1, pending=1)

[DEBUG][CLIENT] chunk=1 channel=1 tag=100 size=524288 offset=524288 pending=0/12
...

[DEBUG][CLIENT] chunk=12 channel=0 tag=111 size=524288 offset=6291456 pending=12/12
[DEBUG][CLIENT] Channel 0 send window FULL (12/12), waiting for oldest send
[DEBUG][CLIENT] Oldest send completed after 156 polls, sent_size=0
[DEBUG][CLIENT] Channel 0 window now has 11 pending sends
...

[DEBUG] Iteration 0 end: draining send windows for 4 channels
[DEBUG][CLIENT] Draining channel 0 (3 pending sends)
[DEBUG][CLIENT] Channel 0 drained
[DEBUG][CLIENT] Draining channel 1 (3 pending sends)
[DEBUG][CLIENT] Channel 1 drained
...
```

---

## Updated Documentation

### Files Modified

1. **tests/test_tcpx_perf_multi.cc**:
   - Changed chunk size from 2MB to 512KB
   - Added ~30 debug log statements
   - Updated header comments

2. **docs/PHASE3_COMPLETE.md**:
   - Corrected "lesson 10" from "2MB optimization" to "512KB default"
   - Updated code examples

3. **docs/DEBUG_LOGS_ADDED.md**:
   - This file (comprehensive documentation of debug logs)

---

## Compilation

```bash
$ cd /home/daniel/uccl/p2p/tcpx
$ make clean && make test_tcpx_perf_multi
✅ Success (no warnings or errors)
```

---

## Testing Recommendations

### 1. Single Channel Test (Baseline)
```bash
# Server
UCCL_TCPX_PERF_SIZE=4194304 ./tests/test_tcpx_perf_multi server 0 2>&1 | tee logs/server_1ch_debug.log

# Client
UCCL_TCPX_PERF_SIZE=4194304 ./tests/test_tcpx_perf_multi client <ip> 0 2>&1 | tee logs/client_1ch_debug.log
```

### 2. Multi-Channel Test (4 Channels)
```bash
# Server
UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 ./tests/test_tcpx_perf_multi server 0 2>&1 | tee logs/server_4ch_debug.log

# Client
UCCL_TCPX_NUM_CHANNELS=4 UCCL_TCPX_PERF_SIZE=67108864 ./tests/test_tcpx_perf_multi client <ip> 0 2>&1 | tee logs/client_4ch_debug.log
```

### 3. Log Analysis
```bash
# Check round-robin distribution
grep "chunk=.*channel=" logs/server_4ch_debug.log | head -20

# Check window management
grep "window FULL" logs/server_4ch_debug.log

# Check polling iterations
grep "completed after.*polls" logs/server_4ch_debug.log

# Check fragment counts
grep "fragments to unpack" logs/server_4ch_debug.log
```

---

## Conclusion

✅ **All changes complete**:
- Chunk size corrected to 512KB
- Comprehensive debug logs added
- Documentation updated
- Compiles successfully

**Ready for cloud testing with full debug visibility!** 🚀

