# Single-Process P2P Refactor Plan

**Goal**: Refactor P2P benchmark from 8-process to 1-process architecture to enable multi-NIC per GPU.

**Timeline**: ~5-7 days  
**Target Performance**: >15 GB/s bus bandwidth (within 20% of NCCL)

---

## Why Single-Process?

**Current (Multi-Process)**:
- 8 processes/node, 1 GPU each, 1 NIC each
- **Problem**: Cannot share NICs across GPUs (devmem conflicts)
- **Performance**: 2.75 GB/s/GPU (single-NIC only)

**Target (Single-Process)**:
- 1 process/node, 8 GPUs, all 4 NICs visible
- **Benefit**: NIC sharing enables multi-NIC per GPU (no devmem conflicts)
- **Expected**: >10 GB/s/GPU (4 NICs × multi-channel)

**Evidence**: NCCL uses single-process and achieves 19.176 GB/s bus bandwidth.

---

## Target Architecture

### Process Layout
- **1 process per node** managing all 8 GPUs
- **N worker threads** (one per GPU) inside single process
- **All 4 NICs** visible to the process
- **8 channels per GPU** striped across NICs

### GPU↔NIC Binding Matrix
| GPU | NUMA | Primary NICs | Channels |
|-----|------|--------------|----------|
| 0-3 | 0 | eth1, eth2 | 8 (4 per NIC) |
| 4-7 | 1 | eth3, eth4 | 8 (4 per NIC) |

### Thread CPU Affinity
- **TX threads**: NUMA-local cores (reuse NCCL bindings)
- **RX threads**: NUMA-local cores (separate from TX)
- **Binding**: `pthread_setaffinity_np()` per worker thread

---

## Critical Open Questions (MUST RESOLVE)

### 1. Bootstrap Concurrency Strategy
**Problem**: 8 GPU workers × 8 channels = 64 connections per node pair
**Critical Details**:
- Port allocation: `UCCL_TCPX_BOOTSTRAP_PORT_BASE + gpu_id * 8 + ch_id`
  - GPU 0: ports 20000-20007 (8 channels)
  - GPU 1: ports 20008-20015 (8 channels)
  - GPU 7: ports 20056-20063 (8 channels)
- Connection metadata: How does listener map incoming socket → (gpu_id, ch_id)?
- Sequencing: All workers bind/listen concurrently, then connect in deterministic order

**Chosen Approach** (Option B - Per-GPU Port Ranges):
- Each worker thread binds its own port range (8 ports for 8 channels)
- Server: All workers listen concurrently
- Client: Connect in order (GPU 0 ch 0, GPU 0 ch 1, ..., GPU 7 ch 7)
- Mapping: Port number directly encodes (gpu_id, ch_id)

**Risks**:
- Deadlock if connection order not deterministic
- Mispairing if port calculation wrong

**Mitigation**: Explicit connection sequencing, extensive logging

### 2. Devmem Resource Limits
**Assumption**: Single-process → no devmem conflicts
**Risk**: TCPX plugin may limit by channel/queue, not process
**Validation**: Step 2.5 early test - **MUST test multi-channel on same NIC**
- 1 process, 1 GPU, **4 channels**, all on eth1
- This is where original conflicts occurred!
**Fallback**: If conflicts persist, contact Google or reconsider approach

### 3. NIC Configuration for Single Process
**Problem**: Current launcher sets `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` to single NIC per worker
**Required**: Single process must advertise all 4 NICs to plugin
**Solution**:
```bash
# OLD (multi-process): Each worker sees 1 NIC
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1  # GPU 0-1
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2  # GPU 2-3
...

# NEW (single-process): Process sees all NICs
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
```
**Action**: Update launcher in Step 2

### 4. Thread Safety Guarantees
**Problem**: Current code not designed for multi-threading
**Decisions**:
- GlobalChannelManager singleton (not per-thread)
- Mutex for devmem registration cache
- Mutex for channel map
- TCPX plugin thread safety: Assume safe (NCCL uses it multi-threaded)

**Action**: Implement in Step 3

### 5. Thread Affinity Ownership
**Problem**: Both plugin and application may try to bind threads
**Checkpoint** (before Step 4):
1. Run prototype with only env vars set
2. Check actual CPU binding: `ps -eLo pid,tid,psr,comm`
3. Decision:
   - If threads auto-bind → use env vars only
   - If threads NOT bound → implement manual pthread_setaffinity_np()

**Action**: Explicit checkpoint before Step 4 implementation

---

## Implementation Steps

### Step 1: Define Architecture (1-2 hours)

**Tasks**:
1. Document GPU↔NIC binding matrix
2. Design process-level config struct
3. Define channel distribution strategy

**Deliverable**: Architecture design doc (this file)

---

### Step 2: Refactor Control Plane (1-2 days)

**Critical Issues to Resolve**:
1. **Bootstrap state machine redesign**
   - 8 GPU workers in same address space → concurrent bind/listen/accept
   - Port reuse conflicts: How to avoid multiple threads binding same port?
   - Client/server pairing order: How to ensure correct GPU pair matching?
   - **Solution options**:
     - Option A: Single bootstrap thread serializes all handshakes
     - Option B: Per-GPU port ranges (e.g., GPU 0: 20000-20007, GPU 1: 20008-20015)
     - Option C: Use SO_REUSEPORT + connection metadata to route

2. **Thread safety**
   - Current bootstrap code not thread-safe
   - Need mutex/lock strategy for concurrent handshakes

**Tasks**:
1. Create `run_p2p_singleproc.sh`
   - Spawn single binary per node
   - Remove per-GPU forking from `run_p2p_fullmesh.sh`

2. **Redesign bootstrap logic** (critical!)
   - Design: Single bootstrap thread or concurrent with port ranges?
   - Implement: Thread-safe handshake state machine
   - Test: Verify all 8 GPU pairs connect correctly
   - **Deliverable**: Bootstrap design doc explaining concurrency strategy

3. Create process-level config struct
   ```cpp
   struct ProcessConfig {
       std::vector<int> gpu_ids;        // [0,1,2,3,4,5,6,7]
       std::vector<std::string> nics;   // ["eth1","eth2","eth3","eth4"]
       int channels_per_gpu;            // 8
       std::map<int, std::vector<std::string>> gpu_to_nics;  // GPU→NIC mapping
   };
   ```

**Deliverables**:
- `run_p2p_singleproc.sh`
- Bootstrap design doc (concurrency strategy)
- Thread-safe bootstrap implementation
- `ProcessConfig` struct in header

**Estimated Time**: 1-2 days (was 4-6 hours - too optimistic!)

---

### Step 2.5: Early Devmem Validation (0.5 day) **[NEW - CRITICAL]**

**Why**: Verify single-process assumption before full refactor

**Tasks**:
1. **Create minimal test**: 1 process, 1 GPU, **4 channels**, 1 NIC
   - All 4 channels use eth1 (same NIC!)
   - **This is where original conflicts occurred** (multi-channel on same NIC)
   - Test if devmem conflict still occurs in single-process

2. **Test config**:
   ```bash
   export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1
   export UCCL_TCPX_NUM_CHANNELS=4
   # Run single-process test with 1 GPU, 4 channels
   ```

3. **Decision point**:
   - ✅ If no conflict → proceed to Step 3
   - ❌ If conflict persists → contact Google, reconsider approach

**Deliverable**: Validation test result + decision to proceed or pivot

---

### Step 3: Upgrade Data Plane (2-3 days)

**Option A**: Fork `tests/test_tcpx_perf_multi.cc`
**Option B**: Create `tests/test_tcpx_perf_orchestrator.cc` (recommended)

**Critical Issues to Resolve**:
1. **ChannelManager thread safety**
   - Current: No thread-safe guarantees
   - Problem: Multiple workers calling `create_channel()` concurrently
   - Solution: Add mutex for channel map access

2. **Global NIC devmem registration**
   - Current: Each ChannelManager instance may re-register same NIC
   - Problem: Redundant registration or conflicts
   - Solution: Global registration cache with mutex

3. **TCPX plugin thread safety**
   - Unknown: Does TCPX plugin support concurrent calls from multiple threads?
   - Action: Test or audit plugin source

**Core Changes**:

#### 3.1 Multi-Threaded Worker Model
```cpp
// Pseudo-code (REVISED)
class GlobalChannelManager {
private:
    std::mutex devmem_mutex_;
    std::set<std::string> registered_nics_;  // Global cache
    std::mutex channel_mutex_;
    std::map<std::pair<int, int>, TcpxChannel*> channels_;  // (gpu_id, ch_id)

public:
    void register_devmem_once(const std::string& nic) {
        std::lock_guard<std::mutex> lock(devmem_mutex_);
        if (registered_nics_.count(nic)) return;  // Already registered
        // ... actual registration ...
        registered_nics_.insert(nic);
    }

    TcpxChannel* create_channel(int gpu_id, const std::string& nic, int ch_id) {
        std::lock_guard<std::mutex> lock(channel_mutex_);
        // ... create channel ...
    }
};

GlobalChannelManager g_channel_mgr;  // Process-wide singleton

void run_orchestrator(ProcessConfig& config) {
    std::vector<std::thread> workers;

    for (int gpu_id : config.gpu_ids) {
        workers.emplace_back([gpu_id, &config]() {
            // Set CUDA context
            cudaSetDevice(gpu_id);

            // Get NICs for this GPU
            auto nics = config.gpu_to_nics[gpu_id];

            // Register devmem for all NICs (once per process)
            for (const auto& nic : nics) {
                g_channel_mgr.register_devmem_once(nic);
            }

            // Create channels (distributed across NICs)
            for (int ch = 0; ch < config.channels_per_gpu; ch++) {
                std::string nic = nics[ch % nics.size()];  // Round-robin
                g_channel_mgr.create_channel(gpu_id, nic, ch);
            }

            // Run P2P transfers
            run_p2p_worker(gpu_id, g_channel_mgr);
        });
    }

    for (auto& t : workers) t.join();
}
```

#### 3.2 ChannelManager Upgrade

**Current**: Single `map_gpu_to_ifaces` ownership, no thread safety
**Target**: Global singleton with thread-safe multi-NIC pool

**Changes**:
```cpp
class GlobalChannelManager {
private:
    // Thread-safe global state
    std::mutex devmem_mutex_;
    std::set<std::string> registered_nics_;  // Devmem registration cache

    std::mutex channel_mutex_;
    std::map<int, std::vector<std::string>> gpu_to_nics_;
    std::map<std::pair<int, int>, TcpxChannel*> channels_;  // (gpu_id, ch_id)

public:
    // Thread-safe API
    void add_nic_pool(int gpu_id, const std::vector<std::string>& nics);
    void register_devmem_once(const std::string& nic);  // Idempotent
    TcpxChannel* create_channel(int gpu_id, const std::string& nic, int ch_id);
    TcpxChannel* get_channel(int gpu_id, int ch_id);
};
```

**Key Changes**:
- Global singleton (not per-worker instance)
- Mutex-protected devmem registration cache
- Mutex-protected channel map
- Idempotent `register_devmem_once()`

**Deliverables**:
- `tests/test_tcpx_perf_orchestrator.cc`
- `src/global_channel_manager.cc` (new file)
- `include/global_channel_manager.h` (new file)
- Thread safety tests

---

### Step 4: Thread Affinity Layer (0.5-1 day)

**Critical Issue to Resolve**:
- **Double binding risk**: TCPX plugin may read `NCCL_GPUDIRECTTCPX_*_BINDINGS` and bind threads itself
- **Question**: Does plugin auto-bind, or do we need manual `pthread_setaffinity_np()`?
- **Action**: Audit plugin source or test with env vars only

**Tasks**:
1. **Investigate plugin behavior**
   - Test: Set env vars, check if threads auto-bind
   - Audit: Review TCPX plugin source for binding logic
   - Decision: Use env vars only OR manual pthread calls

2. Parse NCCL binding strings into CPU sets (if manual binding needed)

3. Implement `pthread_setaffinity_np()` in worker threads (if manual binding needed)

4. Add runtime verifier (log actual CPU per thread)

**Environment Variables** (reuse NCCL's):
```bash
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

**Implementation** (if manual binding needed):
```cpp
void set_thread_affinity(int gpu_id, const std::string& role) {
    // Check if plugin already bound this thread
    int current_cpu = sched_getcpu();
    cpu_set_t current_set;
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &current_set);

    // If already bound (by plugin), log and return
    if (CPU_COUNT(&current_set) < 208) {  // Not all CPUs
        LOG("GPU %d %s thread already bound by plugin to CPU %d",
            gpu_id, role.c_str(), current_cpu);
        return;
    }

    // Manual binding
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    auto cores = parse_binding_for_gpu(gpu_id, role);  // "TX" or "RX"
    for (int cpu : cores) {
        CPU_SET(cpu, &cpuset);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Verify
    int actual_cpu = sched_getcpu();
    LOG("GPU %d %s thread manually bound to CPU %d", gpu_id, role.c_str(), actual_cpu);
}
```

**Deliverables**:
- Plugin binding behavior investigation report
- Thread affinity utility functions (if needed)
- Runtime verification logs

---

### Step 5: Instrumentation (0.5-1 day)

**Critical Addition**: Per-NIC traffic verification

**Tasks**:
1. **Add per-NIC counters** (confirm simultaneous traffic)
   - Capture `ethtool -S eth{1,2,3,4} | grep rx_devmem_pkts` before/after
   - Log delta per NIC to verify all 4 NICs active
   - **Critical**: This is how we confirm multi-NIC actually works!

2. **Add per-channel traffic counters**
   - Log bytes sent/received per channel
   - Verify channels distributed across NICs as expected
   - Example output:
     ```
     GPU 0: ch0→eth1 (10GB), ch1→eth2 (10GB), ch2→eth1 (10GB), ch3→eth2 (10GB)
     GPU 4: ch0→eth3 (10GB), ch1→eth4 (10GB), ch2→eth3 (10GB), ch3→eth4 (10GB)
     ```

3. Capture `/proc/interrupts` delta (pre/post run)

4. Log aggregated bandwidth (per GPU, per NIC)

5. Add feature flag for architecture toggle

6. Implement fallback to multi-process

**Feature Flag**:
```bash
export UCCL_TCPX_SINGLE_PROCESS=1  # Enable single-process mode
```

**Deliverables**:
- Per-NIC traffic verification (ethtool)
- Per-channel traffic counters
- Instrumentation code
- Feature flag logic
- Fallback mechanism

---

### Step 6: Validation (2-3 days)

#### 6.1 Functional Smoke Test (2 hours)
**Config**: 1 GPU/node, new binary  
**Goal**: Verify basic P2P works  
**Success**: No errors, bandwidth ≥ baseline (2.75 GB/s)

#### 6.2 Scale to 8 GPUs (4 hours)
**Config**: 8 GPUs, 1 channel/GPU  
**Goal**: Confirm no devmem conflicts  
**Success**: All GPU pairs communicate, no "rx no cmsg" errors

#### 6.3 Enable Multi-NIC × Multi-Channel (1-2 days)
**Config**: 4 NICs × 8 channels/GPU
**Goal**: Approach NCCL performance
**Success**: Bandwidth >10 GB/s/GPU

**Test Matrix**:
| NICs | Channels/GPU | Expected BW/GPU | Verification |
|------|--------------|-----------------|--------------|
| 1 | 1 | ~2.75 GB/s (baseline) | ethtool eth1 only |
| 2 | 2 | ~5 GB/s (2x) | ethtool eth1+eth2 |
| 4 | 4 | ~10 GB/s (4x) | ethtool all 4 NICs |
| 4 | 8 | ~15 GB/s (target) | ethtool all 4 NICs |

**Per-Test Verification** (CRITICAL):
```bash
# Before test
ethtool -S eth1 | grep rx_devmem_pkts > /tmp/before_eth1.txt
ethtool -S eth2 | grep rx_devmem_pkts > /tmp/before_eth2.txt
ethtool -S eth3 | grep rx_devmem_pkts > /tmp/before_eth3.txt
ethtool -S eth4 | grep rx_devmem_pkts > /tmp/before_eth4.txt

# Run test
./run_p2p_singleproc.sh ...

# After test
ethtool -S eth1 | grep rx_devmem_pkts > /tmp/after_eth1.txt
ethtool -S eth2 | grep rx_devmem_pkts > /tmp/after_eth2.txt
ethtool -S eth3 | grep rx_devmem_pkts > /tmp/after_eth3.txt
ethtool -S eth4 | grep rx_devmem_pkts > /tmp/after_eth4.txt

# Verify deltas
# Expected for 4-NIC test: All 4 NICs show significant increase
# Expected for 2-NIC test: Only 2 NICs show increase
```

**Channel Distribution Verification**:
- Add logging in orchestrator to show which channel uses which NIC
- Example expected output:
  ```
  GPU 0: ch0→eth1, ch1→eth2, ch2→eth1, ch3→eth2, ch4→eth1, ch5→eth2, ch6→eth1, ch7→eth2
  GPU 4: ch0→eth3, ch1→eth4, ch2→eth3, ch3→eth4, ch4→eth3, ch5→eth4, ch6→eth3, ch7→eth4
  ```

#### 6.4 CPU Utilization Tracking (2 hours)
**Tools**: `mpstat -P ALL 1`, `ps -eLo pid,tid,psr,comm`
**Goal**: Verify thread bindings stick
**Success**: Threads on expected NUMA-local cores

---

## Success Criteria

### Functional
- [ ] Single-process P2P works (1 GPU smoke test)
- [ ] No devmem conflicts (8 GPUs, 1 channel)
- [ ] Multi-NIC works (4 NICs, no errors)
- [ ] Multi-channel works (8 channels/GPU)

### Performance
- [ ] Bandwidth >10 GB/s/GPU (4 NICs × multi-channel)
- [ ] Bandwidth >15 GB/s bus BW (target)
- [ ] Within 20% of NCCL (19.176 GB/s)

### Quality
- [ ] Stable and reproducible
- [ ] Thread affinity verified
- [ ] Well-documented
- [ ] Fallback to multi-process works

---

## Risk Mitigation (UPDATED)

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **Devmem conflicts persist** | Medium | **Step 2.5**: Early validation test before full refactor |
| **Bootstrap concurrency bugs** | High | Design doc + careful testing, consider single-thread bootstrap |
| **ChannelManager thread safety** | High | Global singleton + mutex, thorough testing |
| **TCPX plugin not thread-safe** | Medium | Audit source, add global lock if needed |
| **Double thread binding** | Medium | Investigate plugin behavior first (Step 4) |
| **Performance below target** | Medium | Profile with `perf`, per-NIC verification with ethtool |
| **Implementation time overruns** | High | Keep multi-process fallback, realistic timeline (11-14 days) |
| **Multi-channel degrades perf** | Low | Ensure channels distributed across NICs, verify with logging |

---

## Timeline (REVISED)

| Step | Effort | Calendar | Notes |
|------|--------|----------|-------|
| 1. Define architecture | 1-2 hours | 0.5 day | |
| 2. Refactor control plane | 1-2 days | 2 days | **Bootstrap concurrency critical** |
| 2.5. Early devmem validation | 0.5 day | 0.5 day | **NEW - verify assumption** |
| 3. Upgrade data plane | 2-3 days | 3 days | **Thread safety critical** |
| 4. Thread affinity | 0.5-1 day | 1 day | **Investigate plugin behavior** |
| 5. Instrumentation | 0.5-1 day | 1 day | **Per-NIC verification** |
| 6. Validation | 2-3 days | 3 days | **ethtool verification** |
| **Total** | **~7-10 days** | **11-14 days** | **More realistic estimate** |

**Previous estimate**: 5 days (too optimistic)
**Revised estimate**: 7-10 days work, 11-14 days calendar (with debugging buffer)

---

## Next Action

**BEFORE CODING**: Resolve critical open questions

### Immediate Actions (Day 1)

1. **Decide bootstrap strategy** (Critical Question #1)
   - Review current bootstrap code
   - Design concurrency approach (Option A/B/C)
   - Document decision in design doc

2. **Plan devmem validation** (Critical Question #2)
   - Design minimal test (1 proc, 2 GPUs, 1 NIC)
   - Prepare to pivot if assumption fails

3. **Audit TCPX plugin** (Critical Questions #3, #4)
   - Check thread safety guarantees
   - Check auto-binding behavior
   - Document findings

### Then Start Implementation (Day 2+)

```bash
# 1. Create bootstrap design doc
vim p2p/tcpx/docs/BOOTSTRAP_DESIGN.md
# Document: Concurrency strategy, port allocation, pairing logic

# 2. Prototype launcher
cp run_p2p_fullmesh.sh run_p2p_singleproc.sh
vim run_p2p_singleproc.sh  # Remove per-GPU forking

# 3. Create devmem validation test (Step 2.5)
vim tests/test_devmem_validation.cc
# Test: 1 process, 2 GPUs, same NIC

# 4. Create orchestrator skeleton (after validation passes)
cp tests/test_tcpx_perf_multi.cc tests/test_tcpx_perf_orchestrator.cc
vim tests/test_tcpx_perf_orchestrator.cc  # Add multi-threaded worker model
```

---

**Status**: Plan revised with critical questions identified
**Last Updated**: 2025-10-07 (revised based on technical review)
**Timeline**: 11-14 days (realistic estimate)
**Next**: Resolve open questions before coding

