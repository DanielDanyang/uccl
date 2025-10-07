# Step 2.5: Devmem Validation - READY TO TEST

**Date**: 2025-10-07  
**Status**: ✅ Code complete, compiled successfully  
**Next**: Run on GCP nodes

---

## What Was Built

### Test Program: `test_devmem_validation`

**Purpose**: Verify single-process can use multiple channels on same NIC

**Test Configuration**:
- 1 process
- 1 GPU
- **4 channels** (all on same NIC)
- Concurrent channel creation (std::thread)
- Simple send/recv test per channel

**Why This Test**:
- Original multi-process conflicts occurred with multi-channel on same NIC
- This is the critical scenario to validate before full refactor
- If this passes → single-process architecture will work
- If this fails → need to contact Google or reconsider approach

---

## How to Run

### On GCP Nodes

```bash
cd /home/daniel/uccl/p2p/tcpx

# Node 0 (Server) - GPU 0, eth1 (dev_id=0)
./run_devmem_validation.sh server

# Node 1 (Client) - GPU 0, eth1 (dev_id=0)
./run_devmem_validation.sh client <NODE0_IP>
```

### Test Different NICs

```bash
# eth1 (dev_id=0)
./run_devmem_validation.sh server 0.0.0.0 0 0

# eth2 (dev_id=1)
./run_devmem_validation.sh server 0.0.0.0 0 1

# eth3 (dev_id=2)
./run_devmem_validation.sh server 0.0.0.0 0 2

# eth4 (dev_id=3)
./run_devmem_validation.sh server 0.0.0.0 0 3
```

---

## Expected Results

### ✅ SUCCESS (Devmem conflicts resolved)

```
=== Devmem Validation Test ===
Role: server
GPU: 0
Dev: 0 (eth1)
Channels: 4 (all on same NIC)
==============================

TCPX devices: 4
Device 0: eth1 (speed=200000 Mbps)

[GPU 0 CH 0] Starting server on dev 0
[GPU 0 CH 1] Starting server on dev 0
[GPU 0 CH 2] Starting server on dev 0
[GPU 0 CH 3] Starting server on dev 0
[GPU 0 CH 0] Channel created successfully (server)
[GPU 0 CH 1] Channel created successfully (server)
[GPU 0 CH 2] Channel created successfully (server)
[GPU 0 CH 3] Channel created successfully (server)
[GPU 0 CH 0] Recv OK (16777216 bytes)
[GPU 0 CH 1] Recv OK (16777216 bytes)
[GPU 0 CH 2] Recv OK (16777216 bytes)
[GPU 0 CH 3] Recv OK (16777216 bytes)
[GPU 0 CH 0] Test PASSED
[GPU 0 CH 1] Test PASSED
[GPU 0 CH 2] Test PASSED
[GPU 0 CH 3] Test PASSED

=== TEST RESULTS ===
Success: 4 / 4
Failed:  0 / 4

=== ALL CHANNELS PASSED ===
Result: Single-process CAN use multiple channels on same NIC
Devmem conflicts: RESOLVED
Proceed to Step 3: Full refactor
```

**Next Action**: Proceed to Step 2 (Control plane refactor)

---

### ❌ FAILURE (Devmem conflicts persist)

```
[GPU 0 CH 0] Channel created successfully (server)
[GPU 0 CH 1] tcpx_reg_mr failed
[GPU 0 CH 1] Test FAILED

=== TEST RESULTS ===
Success: 1 / 4
Failed:  3 / 4

=== TEST FAILED ===
Result: Single-process CANNOT use multiple channels on same NIC
Devmem conflicts: STILL PRESENT
Action: Contact Google, reconsider approach
```

**Next Action**: Contact Google, review TCPX plugin source, consider alternatives

---

## Technical Details

### Port Allocation

Each channel uses a unique port:
```
port = 20000 + gpu_id * 8 + ch_id

GPU 0, CH 0: 20000
GPU 0, CH 1: 20001
GPU 0, CH 2: 20002
GPU 0, CH 3: 20003
```

### TCPX API Flow

**Server**:
1. `tcpx_listen()` → create listen comm
2. Bootstrap: send handle to client
3. `tcpx_accept_v5()` → accept connection
4. `tcpx_reg_mr()` → register GPU buffer
5. `tcpx_irecv()` → async receive
6. `tcpx_test()` → poll for completion

**Client**:
1. Bootstrap: receive handle from server
2. `tcpx_connect_v5()` → connect to server
3. `tcpx_reg_mr()` → register GPU buffer
4. `tcpx_isend()` → async send
5. `tcpx_test()` → poll for completion

### Concurrent Execution

All 4 channels run concurrently in separate threads:
```cpp
std::vector<std::thread> threads;
for (int ch = 0; ch < 4; ch++) {
    threads.emplace_back([=]() {
        run_channel_test(gpu_id, ch, dev_id, role, peer_ip);
    });
}
for (auto& t : threads) t.join();
```

---

## Files Created/Modified

### New Files
- `tests/test_devmem_validation.cc` (280 lines)
- `run_devmem_validation.sh` (51 lines)
- `READY_TO_TEST.md` (this file)

### Modified Files
- `Makefile` (+5 lines: test_devmem_validation target)

---

## Build Status

✅ **Compiled successfully** (v2 - fixed tcpx_reg_mr issues)

```bash
$ make test_devmem_validation
Building test_devmem_validation...
/usr/local/cuda/bin/nvcc ... -o tests/test_devmem_validation ...
```

No errors, no warnings.

### Key Fixes Applied

Based on analysis of `server.log` and `client.log`:

1. **4KB Memory Alignment** (devmem-tcp requirement)
   - Changed from `cudaMalloc` to `cuMemAlloc`
   - Added 4KB alignment: `addr = (addr + 4095) & ~4095`
   - This is CRITICAL for devmem-tcp to work

2. **CUDA Driver API**
   - Using `CUdeviceptr` instead of `void*`
   - Proper CUDA context initialization
   - Matches working test_tcpx_perf_multi.cc pattern

3. **Accept Retry Logic**
   - Added retry loop for `tcpx_accept_v5` (may return nullptr)
   - Up to 100 retries with 10ms sleep
   - Matches NCCL pattern

4. **Better Error Messages**
   - Print pointer and size on tcpx_reg_mr failure
   - Print mhandle on success for debugging

---

## Next Steps

### If Test Passes ✅

1. **Update plan status** in `docs/IMPLEMENTATION_STARTED.md`
2. **Proceed to Step 2**: Control plane refactor
   - Create `run_p2p_singleproc.sh`
   - Implement bootstrap with per-GPU port ranges
   - Update NIC configuration (all 4 NICs)
3. **Then Step 3**: Data plane upgrade
   - Create `GlobalChannelManager` singleton
   - Add mutex protection
   - Implement `register_devmem_once()`

### If Test Fails ❌

1. **Capture error logs** for analysis
2. **Contact Google** for guidance on TCPX plugin resource limits
3. **Review TCPX plugin source** (if available)
4. **Consider alternatives**:
   - Hybrid: Single process but serialize channel creation
   - Plugin modification request
   - Different architecture (stay with multi-process, optimize differently)

---

## Timeline Impact

**If test passes**: On track for 11-14 day timeline  
**If test fails**: +3-5 days for investigation and pivot

---

**Status**: ✅ Ready to test on GCP  
**Command**: `./run_devmem_validation.sh server` (Node 0)  
**Command**: `./run_devmem_validation.sh client <NODE0_IP>` (Node 1)

