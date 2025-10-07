# Single-Process Refactor - Implementation Started

**Date**: 2025-10-07  
**Status**: Step 2.5 (Devmem Validation) ready to test

---

## Plan Revisions Complete

Based on technical review, the following critical issues were addressed in SINGLE_PROCESS_PLAN.md:

### 1. Bootstrap Concurrency (Specified)
- **Port allocation**: `BASE + gpu_id * 8 + ch_id`
- **Approach**: Per-GPU port ranges (GPU 0: 20000-20007, GPU 1: 20008-20015, ...)
- **Sequencing**: All workers bind/listen concurrently, connect in deterministic order
- **Mapping**: Port number directly encodes (gpu_id, ch_id)

### 2. Devmem Validation (Corrected)
- **Test**: 1 process, 1 GPU, **4 channels**, all on eth1
- **Why**: Multi-channel on same NIC is where original conflicts occurred
- **Critical**: Must test this before full refactor

### 3. NIC Configuration (Added)
- **Required**: `export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4`
- **Why**: Single process must advertise all 4 NICs to plugin

### 4. Thread Safety (Decided)
- **Approach**: GlobalChannelManager singleton
- **Protection**: Mutex for devmem cache, mutex for channel map
- **Assumption**: TCPX plugin is thread-safe (NCCL uses it multi-threaded)

### 5. Thread Affinity (Checkpoint Added)
- **Before Step 4**: Run prototype with only env vars
- **Check**: `ps -eLo pid,tid,psr,comm` to see actual CPU binding
- **Decision**: If auto-bind → env vars only; else → manual pthread

---

## Step 2.5: Devmem Validation (READY TO TEST)

### Files Created

1. **tests/test_devmem_validation.cc**
   - Minimal test: 1 process, 1 GPU, 4 channels, 1 NIC
   - All 4 channels use same NIC (eth1)
   - Tests if devmem conflict still occurs in single-process
   - Concurrent channel creation (std::thread)

2. **run_devmem_validation.sh**
   - Simple runner script
   - Sets environment variables
   - Builds if needed

3. **Makefile** (updated)
   - Added `test_devmem_validation` target
   - Added to `all` target
   - Added to `clean` target

### How to Run

```bash
cd /home/daniel/uccl/p2p/tcpx

# Build
make test_devmem_validation

# Node 0 (Server)
./run_devmem_validation.sh server

# Node 1 (Client)
./run_devmem_validation.sh client <NODE0_IP>
```

### Expected Outcomes

**Success** (no devmem conflicts):
```
[GPU 0 CH 0] Channel created successfully on eth1
[GPU 0 CH 1] Channel created successfully on eth1
[GPU 0 CH 2] Channel created successfully on eth1
[GPU 0 CH 3] Channel created successfully on eth1
...
=== ALL CHANNELS PASSED ===
Result: Single-process CAN use multiple channels on same NIC
Devmem conflicts: RESOLVED
Proceed to Step 3: Full refactor
```

**Failure** (devmem conflicts persist):
```
[GPU 0 CH 0] Channel created successfully on eth1
[GPU 0 CH 1] Failed to create channel
...
```
→ Contact Google, reconsider approach

---

## Next Steps

### If Validation Passes ✅
1. Proceed to Step 2: Control plane refactor
   - Create `run_p2p_singleproc.sh`
   - Implement bootstrap with per-GPU port ranges
   - Update NIC configuration (all 4 NICs)

2. Then Step 3: Data plane upgrade
   - Create `GlobalChannelManager` singleton
   - Add mutex protection
   - Implement `register_devmem_once()`

### If Validation Fails ❌
1. Contact Google for guidance
2. Review TCPX plugin source for resource limits
3. Consider alternative approaches:
   - Hybrid: Single process but serialize channel creation
   - Plugin modification request
   - Different architecture

---

## Timeline Update

**Original estimate**: 5 days work, 7-8 days calendar  
**Revised estimate**: 7-10 days work, 11-14 days calendar

**Current progress**:
- [x] Plan revision (based on technical review)
- [x] Step 2.5 implementation (devmem validation test)
- [ ] Step 2.5 execution (run test on GCP)
- [ ] Step 2: Control plane refactor
- [ ] Step 3: Data plane upgrade
- [ ] Step 4: Thread affinity
- [ ] Step 5: Instrumentation
- [ ] Step 6: Validation

---

## Key Decisions Made

1. **Bootstrap strategy**: Per-GPU port ranges (Option B)
2. **Devmem test**: 4 channels on same NIC (not 2 GPUs)
3. **NIC config**: Comma-separated list for single process
4. **Thread safety**: Global singleton + mutexes
5. **Affinity checkpoint**: Test before implementing

---

## Code Changes Summary

### New Files
- `tests/test_devmem_validation.cc` (145 lines)
- `run_devmem_validation.sh` (45 lines)
- `docs/IMPLEMENTATION_STARTED.md` (this file)

### Modified Files
- `Makefile` (+5 lines: test_devmem_validation target)
- `docs/SINGLE_PROCESS_PLAN.md` (revised based on review)

### No Changes Yet
- `run_p2p_fullmesh.sh` (will create new `run_p2p_singleproc.sh`)
- `src/channel_manager.cc` (will create new `src/global_channel_manager.cc`)
- `tests/test_tcpx_perf_multi.cc` (will create new `test_tcpx_perf_orchestrator.cc`)

---

## Risk Status

| Risk | Status | Mitigation |
|------|--------|------------|
| Devmem conflicts persist | **Testing now** | Step 2.5 validation |
| Bootstrap concurrency bugs | Mitigated | Specified port allocation strategy |
| ChannelManager thread safety | Planned | Global singleton + mutexes in Step 3 |
| TCPX plugin not thread-safe | Assumed safe | NCCL uses it multi-threaded |
| Double thread binding | Checkpoint planned | Test before Step 4 |
| Implementation time overruns | Acknowledged | Realistic 11-14 day timeline |

---

**Status**: Ready to run Step 2.5 validation test  
**Next Action**: Execute test on GCP nodes  
**Decision Point**: Proceed to Step 2 if test passes

