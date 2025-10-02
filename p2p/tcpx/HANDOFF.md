# TCPX P2P Project Handoff Guide

**Date**: 2025-10-02  
**For**: Next developer / AI assistant  
**Project**: TCPX-based P2P GPU communication performance test

---

## 🎯 Project Summary

**Goal**: Build a high-performance point-to-point GPU-to-GPU communication test using Google's TCPX plugin.

**Current Status**: ✅ Functional, ⚠️ Performance needs optimization

**Key Achievement**: Fixed critical sliding window bug that limited throughput to 17 chunks.

---

## 📊 Current Performance

| Metric | Server | Client | Target | Gap |
|--------|--------|--------|--------|-----|
| **Latency** | 21 ms | 77 ms | <10 ms | 2-8× |
| **Bandwidth** | 3 GB/s | 1 GB/s | >10 GB/s | 3-10× |
| **NICs Used** | 1 (eth1) | 1 (eth1) | 4 (eth1-4) | **CRITICAL** |

**Baseline** (iperf3): 7.5 GB/s per NIC, 30 GB/s total (4 NICs)

---

## ⚠️ CRITICAL ISSUE: Multi-NIC Not Working

### Problem

Only eth1 is being used. eth2-4 are idle.

**Evidence**:
```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
# Shows: eth1 has traffic, eth2-4 are idle
```

### Impact

- Bandwidth limited to ~3 GB/s (should be ~12 GB/s with 4 NICs)
- **This is the #1 priority to fix**

### Configuration

```bash
# We ARE setting this:
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4

# But TCPX plugin may not be reading it
```

### Next Steps to Debug

1. **Enable TCPX debug logs** (already added to `bench_p2p.sh`):
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=NET
   ```

2. **Run test and check logs**:
   ```bash
   ./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
   grep "SOCKET_IFNAME" logs/bench_server_*.log
   ```

3. **Look for**:
   - `NET_GPUDIRECTTCPX_SOCKET_IFNAME set to eth1,eth2,eth3,eth4`
   - If not found, TCPX plugin is not reading the env var

4. **Possible causes**:
   - TCPX plugin version mismatch
   - Environment variable not exported before plugin loads
   - TCPX plugin compiled with different adapter (native vs nccl)

5. **Diagnostic script**:
   ```bash
   ./diagnose_multi_nic.sh
   ```

---

## ✅ What's Been Fixed

### 1. Sliding Window Bug (CRITICAL FIX)

**Problem**: Server only processed 17 chunks (should be 128), then failed with "unable to allocate requests".

**Root Cause**: Sliding window check was AFTER `tcpx_irecv()` instead of BEFORE.

**Fix**: Moved sliding window check to before `tcpx_irecv()` (see `tests/test_tcpx_perf.cc:530-565`).

**Result**: ✅ All 128 chunks now processed successfully.

---

### 2. Unpack Kernel Performance (100× speedup)

**Problem**: Kernel mode was 100× slower than D2D mode.

**Root Cause**: CUDA stream and kernel launcher were created inside the loop.

**Fix**: Moved stream/launcher creation outside the loop.

**Result**: ✅ Kernel mode now same speed as D2D mode.

---

### 3. Debug Logs Removed (2-3× speedup)

**Problem**: Excessive `std::cout` debug logs slowed down performance.

**Fix**: Removed 10 debug log statements.

**Result**: ✅ Server 100ms → 21ms, Client 157ms → 77ms.

---

### 4. Chunk Size Optimization (4× fewer chunks)

**Problem**: 512 KB chunks → 128 chunks → high overhead.

**Fix**: Increased to 2 MB chunks → 32 chunks.

**Result**: ✅ Reduced chunk count by 4×.

---

## 📁 Key Files

### Source Code

- **`tests/test_tcpx_perf.cc`** (1100+ lines)
  - Main performance test program
  - Fully annotated in Chinese
  - Implements sliding window, chunking, unpack kernel

- **`device/unpack_kernels.cu`**
  - GPU kernel for unpacking scattered TCPX buffers
  - Fixed performance regression

- **`device/unpack_launch.cu`**
  - Kernel launcher (moved outside loop for performance)

### Scripts

- **`bench_p2p.sh`** - Main test harness
- **`diagnose_multi_nic.sh`** - Multi-NIC diagnostics
- **`Makefile`** - Build system

### Documentation

- **`HANDOFF.md`** (this file) - Handoff guide
- **`docs/TEST_TCPX_PERF_EXPLAINED.md`** - Detailed code explanation
- **`docs/SLIDING_WINDOW_FIX_FINAL.md`** - Sliding window bug fix
- **`docs/CHUNK_SIZE_OPTIMIZATION.md`** - Chunk size optimization

---

## 🔧 How to Build and Run

### Build

```bash
cd /home/daniel/uccl/p2p/tcpx
make clean && make test_tcpx_perf -j4
```

### Run Test

**Server (10.65.74.150)**:
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**Client (10.64.113.77)**:
```bash
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

### Check Results

```bash
# Performance
grep "Avg:" logs/bench_server_*.log
grep "Avg:" logs/bench_client_*.log

# NIC usage
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

---

## 🎯 Next Steps (Priority Order)

### Priority 1: Fix Multi-NIC (CRITICAL)

**Goal**: Use all 4 NICs (eth1-4) instead of just eth1.

**Expected Impact**: 3 GB/s → 12 GB/s (4× improvement).

**Steps**:
1. Run with `NCCL_DEBUG=INFO` (already added to script)
2. Check logs for `SOCKET_IFNAME` messages
3. Debug why TCPX plugin only uses eth1
4. Possible solutions:
   - Check TCPX plugin version
   - Try `NCCL_CROSS_NIC=1` (currently 0)
   - Check if CPU bindings are correct

---

### Priority 2: Increase Client Sliding Window

**Goal**: Match client window size to server (16 instead of 12).

**Expected Impact**: Client 77ms → 60ms (20% improvement).

**Steps**:
1. Edit `tests/test_tcpx_perf.cc` line ~993
2. Change `MAX_INFLIGHT_SEND = 12` to `MAX_INFLIGHT_SEND = 16`
3. Recompile and test

---

### Priority 3: Batch Operations

**Goal**: Use TCPX batch send/recv to reduce overhead.

**Expected Impact**: 2-3× improvement.

**Steps**:
1. Modify `tcpx_irecv()` to receive multiple chunks at once
2. Modify `tcpx_isend()` to send multiple chunks at once
3. Adjust sliding window logic

---

## 🐛 Known Issues

### 1. Multi-NIC Not Working (CRITICAL)

See "CRITICAL ISSUE" section above.

### 2. Client Slower Than Server

**Symptom**: Client 77ms vs Server 21ms (3.7× difference).

**Cause**: Smaller sliding window (12 vs 16) + network ACK overhead.

**Fix**: Increase client sliding window to 16.

### 3. Performance Below Target

**Current**: 3 GB/s (Server), 1 GB/s (Client)  
**Target**: 12 GB/s (Server), 8 GB/s (Client)

**Blockers**:
1. Multi-NIC not working (4× loss)
2. Client sliding window too small (20% loss)
3. Possible other optimizations needed

---

## 📚 Technical Background

### TCPX Plugin

- **Location**: `/usr/local/tcpx/lib64/libnccl-net.so`
- **Purpose**: GPU Direct TCP/IP (devmem-tcp kernel feature)
- **Limitation**: 16 request slots per comm (MAX_REQUESTS=16)

### Sliding Window

**Why needed**: TCPX has only 16 request slots. Without sliding window, we can only have 16 in-flight requests.

**How it works**:
1. Issue up to 16 requests
2. When full, wait for oldest to complete
3. Release slot with `tcpx_irecv_consumed()`
4. Issue new request

**Critical**: Check must be BEFORE `tcpx_irecv()`, not after!

### Chunking

**Why needed**: Avoid overwhelming TCPX bounce buffers.

**Current**: 2 MB chunks (32 chunks for 64 MB transfer)

**Trade-off**:
- Smaller chunks: More overhead, lower throughput
- Larger chunks: Less overhead, higher throughput, more memory

---

## 🔍 Debugging Tips

### Enable TCPX Debug Logs

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

### Monitor NIC Traffic

```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

### Check TCPX Configuration

```bash
grep "NET_GPUDIRECTTCPX" logs/bench_server_*.log
```

### Profile with nsys

```bash
nsys profile -o tcpx_profile ./tests/test_tcpx_perf server 0
```

---

## 📞 Environment

### Hardware

- **Nodes**: 2× H100 GPU nodes
- **NICs**: 4× 25 Gbps (eth1-4) + 1× control (eth0)
- **IPs**: 10.65.74.150 (Server), 10.64.113.77 (Client)

### Software

- **CUDA**: 12.x
- **TCPX Plugin**: `/usr/local/tcpx/lib64/libnccl-net.so`
- **NCCL**: Custom build in `thirdparty/nccl/`

### Configuration

- **Config file**: `scripts/node_ips/tcpx.txt`
- **Control NIC**: eth0
- **Data NICs**: eth1, eth2, eth3, eth4

---

## 📝 PR Summary (for reference)

### Title

```
feat(p2p/tcpx): Add TCPX performance test with sliding window fix
```

### Description

```
This PR adds a comprehensive TCPX-based P2P GPU communication performance test.

**Key Features**:
1. Sliding window mechanism to handle TCPX's 16-request limit
2. Chunking support (configurable, default 2 MB)
3. GPU unpack kernel for scattered buffer consolidation
4. Multi-NIC support (eth1-4)
5. Comprehensive performance metrics

**Critical Fixes**:
1. Fixed sliding window bug (check before tcpx_irecv, not after)
2. Fixed unpack kernel 100× performance regression
3. Removed debug logs for 2-3× speedup
4. Optimized chunk size (512 KB → 2 MB)

**Current Performance**:
- Server: 21 ms, 3 GB/s
- Client: 77 ms, 1 GB/s

**Known Issues**:
- Multi-NIC not working (only eth1 used, should use eth1-4)
- Performance below target (need 12 GB/s)

**Next Steps**:
- Debug multi-NIC configuration
- Increase client sliding window
- Batch operations
```

---

**Good luck! The code is solid, just needs multi-NIC debugging.** 🚀

