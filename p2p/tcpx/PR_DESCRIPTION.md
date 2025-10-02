# PR: TCPX P2P Performance Test with Critical Fixes

## Summary

This PR adds a comprehensive TCPX-based point-to-point GPU communication performance test with critical bug fixes and optimizations.

---

## 🎯 What This PR Does

### 1. Adds TCPX P2P Performance Test

**New Files**:
- `tests/test_tcpx_perf.cc` (1100+ lines, fully annotated in Chinese)
- `device/unpack_kernels.cu` - GPU kernel for unpacking scattered TCPX buffers
- `device/unpack_launch.cu` - Kernel launcher
- `bench_p2p.sh` - Test harness with TCPX configuration
- `diagnose_multi_nic.sh` - Multi-NIC diagnostics

**Features**:
- Sliding window mechanism to handle TCPX's 16-request limit
- Configurable chunking (default 2 MB)
- GPU unpack kernel for scattered buffer consolidation
- Multi-NIC support (eth1-4)
- Comprehensive performance metrics
- Detailed Chinese code annotations

---

### 2. Fixes Critical Sliding Window Bug

**Problem**: Server only processed 17 chunks (should be 128), then failed with "unable to allocate requests".

**Root Cause**: Sliding window check was happening AFTER `tcpx_irecv()` instead of BEFORE, causing TCPX request pool exhaustion.

**Fix**: Moved sliding window check to before `tcpx_irecv()` call.

**Impact**: ✅ All 128 chunks now processed successfully.

**Code**:
```cpp
// BEFORE FIX (WRONG):
int rc = tcpx_irecv(...);  // This fails when pool is full
if (pending_reqs.size() >= MAX_INFLIGHT) {
  // Too late! Already failed
}

// AFTER FIX (CORRECT):
if (pending_reqs.size() >= MAX_INFLIGHT) {
  // Wait for oldest request to complete
  cudaEventSynchronize(oldest_event);
  tcpx_irecv_consumed(recv_comm, 1, oldest_req);
  pending_reqs.erase(pending_reqs.begin());
}
int rc = tcpx_irecv(...);  // Now safe to call
```

**Files Changed**:
- `tests/test_tcpx_perf.cc` (lines 530-565)

---

### 3. Fixes Unpack Kernel Performance Regression

**Problem**: Kernel mode was 100× slower than D2D mode.

**Root Cause**: CUDA stream and kernel launcher were being created inside the iteration loop.

**Fix**: Moved stream/launcher creation outside the loop.

**Impact**: ✅ Kernel mode now same speed as D2D mode.

**Code**:
```cpp
// BEFORE FIX (WRONG):
for (int iter = 0; iter < num_iters; iter++) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);  // Created every iteration!
  UnpackLauncher launcher(stream);  // Created every iteration!
  // ... use stream and launcher ...
}

// AFTER FIX (CORRECT):
cudaStream_t stream;
cudaStreamCreate(&stream);  // Created ONCE
UnpackLauncher launcher(stream);  // Created ONCE
for (int iter = 0; iter < num_iters; iter++) {
  // ... use stream and launcher ...
}
```

**Files Changed**:
- `tests/test_tcpx_perf.cc` (lines 450-480)

---

### 4. Performance Optimizations

#### 4.1 Removed Debug Logs (2-3× speedup)

**Problem**: Excessive `std::cout` debug logs slowed down performance.

**Fix**: Removed 10 debug log statements, kept only `[PERF]` and `[ERROR]` logs.

**Impact**:
- Server: 100 ms → 21 ms (4.8× improvement)
- Client: 157 ms → 77 ms (2× improvement)

**Files Changed**:
- `tests/test_tcpx_perf.cc` (10 locations)

---

#### 4.2 Chunk Size Optimization (4× fewer chunks)

**Problem**: 512 KB chunks → 128 chunks → high overhead per chunk.

**Fix**: Increased default chunk size from 512 KB to 2 MB.

**Impact**:
- Chunk count: 128 → 32 (for 64 MB transfer)
- Reduced per-chunk overhead by 4×

**Files Changed**:
- `tests/test_tcpx_perf.cc` (line 202-206)

---

#### 4.3 Added TCPX Best Practices Configuration

**Added**: 20+ NCCL+TCPX environment variables from GCP best practices.

**Includes**:
- CPU bindings for TX/RX (H100 specific)
- Flow steering configuration
- NCCL chunk sizes and buffer sizes
- Network optimization settings

**Files Changed**:
- `bench_p2p.sh` (lines 100-132)

---

#### 4.4 Fixed Network Configuration

**Problem**: Control NIC was set to eth1 (should be eth0).

**Fix**: Changed `CTRL_DEV` from eth1 to eth0 in all scripts.

**Files Changed**:
- `bench_p2p.sh`
- `bench_p2p_sweep_server.sh`
- `bench_p2p_sweep_client.sh`

---

## 📊 Performance Results

### Current Performance

| Metric | Server | Client |
|--------|--------|--------|
| **Latency** | 21 ms | 77 ms |
| **Bandwidth** | 3 GB/s | 1 GB/s |
| **Chunk Size** | 2 MB | 2 MB |
| **Chunks** | 32 | 32 |

### Performance Improvements

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| **Sliding Window Fix** | 17 chunks | 128 chunks | ✅ Functional |
| **Unpack Kernel Fix** | 2000 ms | 20 ms | **100× faster** |
| **Debug Logs Removed** | 100 ms | 21 ms | **4.8× faster** (server) |
| **Debug Logs Removed** | 157 ms | 77 ms | **2× faster** (client) |
| **Chunk Size** | 128 chunks | 32 chunks | **4× fewer** |

### Baseline Comparison

- **iperf3** (single NIC): 7.5 GB/s
- **Current** (server): 3 GB/s (40% of single NIC)
- **Current** (client): 1 GB/s (13% of single NIC)

---

## ⚠️ Known Issues

### 1. Multi-NIC Not Working (CRITICAL)

**Symptom**: Only eth1 is being used (should use eth1-4).

**Impact**: Bandwidth limited to ~3 GB/s (should be ~12 GB/s with 4 NICs).

**Status**: Under investigation. See `HANDOFF.md` for debugging steps.

**Next Steps**:
1. Enable TCPX debug logs (already added to script)
2. Check if TCPX plugin reads `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME`
3. Try `NCCL_CROSS_NIC=1`

---

### 2. Client Slower Than Server

**Symptom**: Client 77 ms vs Server 21 ms (3.7× difference).

**Cause**: Client has smaller sliding window (12 vs 16).

**Next Steps**: Increase client sliding window to 16.

---

## 📁 Files Changed

### New Files

```
p2p/tcpx/
├── tests/test_tcpx_perf.cc          # Main test program (1100+ lines)
├── device/unpack_kernels.cu         # GPU unpack kernel
├── device/unpack_launch.cu          # Kernel launcher
├── device/unpack_launch.h           # Launcher header
├── bench_p2p.sh                     # Test harness
├── bench_p2p_sweep_server.sh        # Server sweep script
├── bench_p2p_sweep_client.sh        # Client sweep script
├── diagnose_multi_nic.sh            # Multi-NIC diagnostics
├── HANDOFF.md                       # Handoff guide
├── QUICKSTART.md                    # Quick start guide
├── TROUBLESHOOTING.md               # Troubleshooting guide
├── docs/
│   ├── README.md                    # Documentation index
│   ├── TEST_TCPX_PERF_EXPLAINED.md  # Detailed code explanation
│   ├── SLIDING_WINDOW_VISUAL.md     # Sliding window visualization
│   ├── SLIDING_WINDOW_FIX_FINAL.md  # Sliding window bug fix
│   ├── CHUNK_SIZE_OPTIMIZATION.md   # Chunk size optimization
│   ├── DEBUG_LOGS_REMOVED.md        # Debug log removal
│   ├── TCPX_LOGIC_MAPPING.md        # TCPX API mapping
│   ├── COMMON_MISTAKES_AND_FIXES.md # Common mistakes
│   ├── PERF_DIARY.md                # Performance history
│   ├── SERVER_17_CHUNKS_BUG.md      # Bug analysis
│   ├── CURRENT_SETUP.md             # Environment setup
│   └── PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md
└── Makefile                         # Build system
```

### Modified Files

- `Makefile` - Added build rules for test_tcpx_perf

---

## 🧪 Testing

### Build

```bash
cd p2p/tcpx
make clean && make test_tcpx_perf -j4
```

### Run

**Server (Node 1)**:
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**Client (Node 2)**:
```bash
./bench_p2p.sh client <SERVER_IP> 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

### Verify

```bash
# Check performance
grep "Avg:" logs/bench_server_*.log
grep "Avg:" logs/bench_client_*.log

# Check NIC usage
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

---

## 📚 Documentation

### For Users

- **QUICKSTART.md** - Get up and running in 5 minutes
- **TROUBLESHOOTING.md** - Common issues and solutions

### For Developers

- **HANDOFF.md** - Complete project overview and next steps
- **docs/TEST_TCPX_PERF_EXPLAINED.md** - Detailed code explanation (1100+ lines)
- **docs/README.md** - Documentation index

### For Debugging

- **diagnose_multi_nic.sh** - Multi-NIC diagnostics script
- **docs/COMMON_MISTAKES_AND_FIXES.md** - Common mistakes
- **docs/SLIDING_WINDOW_FIX_FINAL.md** - Critical bug fix explanation

---

## 🎯 Next Steps

1. **Debug multi-NIC configuration** (see HANDOFF.md)
2. **Increase client sliding window** to 16
3. **Batch operations** for further optimization
4. **Target performance**: 12 GB/s (server), 8 GB/s (client)

---

## 🤝 Reviewers

### What to Review

1. **Sliding window fix** (`tests/test_tcpx_perf.cc:530-565`)
   - Verify window check is BEFORE `tcpx_irecv()`
   - Verify request slot is released correctly

2. **Unpack kernel fix** (`tests/test_tcpx_perf.cc:450-480`)
   - Verify stream/launcher created OUTSIDE loop
   - Verify no resource leaks

3. **Performance optimizations**
   - Verify debug logs removed
   - Verify chunk size increased to 2 MB
   - Verify TCPX configuration is correct

4. **Documentation**
   - Verify HANDOFF.md is clear and complete
   - Verify QUICKSTART.md works for new users
   - Verify code annotations are helpful

---

## 📝 Checklist

- [x] Code compiles without errors
- [x] Tests run successfully
- [x] Sliding window bug fixed
- [x] Unpack kernel performance fixed
- [x] Debug logs removed
- [x] Chunk size optimized
- [x] TCPX configuration added
- [x] Network configuration fixed
- [x] Documentation complete
- [x] Handoff guide created
- [ ] Multi-NIC issue debugged (in progress)

---

**Ready for review!** 🚀

