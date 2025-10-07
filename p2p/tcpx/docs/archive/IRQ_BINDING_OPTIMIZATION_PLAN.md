# IRQ Binding Optimization Plan for TCPX P2P Benchmark

**Goal**: Match NCCL's IRQ affinity and thread binding strategy to improve P2P benchmark performance from current ~2.75 GB/s to closer to NCCL's ~18.7 GB/s bus bandwidth.

**Status**: Planning phase  
**Date**: 2025-10-07  
**Context**: Single-NIC + 1-channel configuration is stable but underperforming. Multi-NIC attempts failed due to devmem resource conflicts between processes.

---

## Background

### Current Performance Gap
- **P2P Benchmark**: ~2.75 GB/s per GPU (server), ~1.17 GB/s (client)
- **NCCL AllReduce**: ~18.7 GB/s bus bandwidth
- **Gap**: ~6.8x difference

### Known Differences
1. **Channel count**: P2P uses 1 channel/GPU, NCCL uses 8 channels/GPU
2. **Communication pattern**: P2P is unidirectional point-to-point, NCCL is bidirectional collective
3. **IRQ/Thread bindings**: Unknown if P2P matches NCCL's optimized CPU affinity settings

### Hypothesis
NCCL achieves higher performance partly through careful IRQ steering and thread pinning that:
- Reduces CPU contention and cache thrashing
- Keeps network processing on NUMA-local cores
- Balances interrupt load across cores
- Minimizes context switches

---

## Step 1: Gather Current IRQ Binding Information

### 1.1 Analyze NCCL's Binding Strategy

**Objective**: Understand exactly how NCCL configures IRQ affinities and thread bindings.

**Tasks**:

1. **Extract binding configuration from NCCL test script**
   - File: `collective/rdma/run_nccl_test_tcpx.sh`
   - Look for environment variables:
     - `NCCL_GPUDIRECTTCPX_TX_BINDINGS`
     - `NCCL_GPUDIRECTTCPX_RX_BINDINGS`
     - `NCCL_GPUDIRECTTCPX_TX_IRQ_BINDINGS`
     - `NCCL_GPUDIRECTTCPX_RX_IRQ_BINDINGS`
   - Document the exact format and values used

2. **Run NCCL test with enhanced logging**
   - Add `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,ENV`
   - Capture full initialization logs showing:
     - Which cores are assigned to TX/RX threads per NIC
     - Whether IRQ affinities are programmed by NCCL/dp-manager
     - Any per-channel binding overrides
   - Save logs to `collective/rdma/logs/nccl_binding_analysis_<timestamp>.log`

3. **Check NCCL plugin source/documentation**
   - If available, review TCPX plugin docs for recommended IRQ binding practices
   - Look for GCP A3-high specific tuning guides
   - Document any best practices or reference configurations

**Deliverables**:
- Document: `docs/NCCL_BINDING_CONFIG.md` with:
  - Complete list of binding-related env vars and their values
  - Explanation of binding format (e.g., "eth1:8-21,112-125" means cores 8-21 and 112-125)
  - Any NUMA-aware logic observed

---

### 1.2 Collect System IRQ Assignment Information

**Objective**: Map current IRQ assignments to CPUs and NICs on both nodes.

**Tasks**:

1. **Capture IRQ distribution snapshot**
   - On both Node 0 and Node 1, collect:
     ```bash
     # IRQ assignments and counts
     cat /proc/interrupts > irq_snapshot_node<N>_<timestamp>.txt
     
     # Current IRQ affinity masks
     for irq in /proc/irq/*/smp_affinity; do
       echo "$irq: $(cat $irq)"
     done > irq_affinity_node<N>_<timestamp>.txt
     
     # Identify gVNIC IRQs
     grep -i gve /proc/interrupts
     ```
   - Save to `p2p/tcpx/logs/irq_info/`

2. **Map IRQs to NICs**
   - For each NIC (eth1-4), identify:
     - IRQ numbers for TX queues
     - IRQ numbers for RX queues
     - Default CPU affinity for each IRQ
     - NUMA node of the NIC (from PCI topology)
   - Create mapping table:
     ```
     NIC   | IRQ Range | Type | Default CPUs | NUMA
     ------|-----------|------|--------------|-----
     eth1  | 100-115   | RX   | 0-15         | 0
     eth1  | 116-131   | TX   | 0-15         | 0
     ...
     ```

3. **Check dp-manager IRQ programming**
   - Verify if dp-manager service is actively managing IRQ affinities:
     ```bash
     # Check dp-manager status
     sudo systemctl status dp-manager
     
     # Check dp-manager logs for IRQ programming
     sudo journalctl -u dp-manager -n 500 | grep -i irq
     
     # Check if IRQ affinities change during NCCL test
     # (capture before/after snapshots)
     ```

4. **Document NIC queue configuration**
   - For each NIC:
     ```bash
     ethtool -l eth1  # Show queue counts
     ethtool -x eth1  # Show RSS indirection table
     ethtool -S eth1 | grep -i queue  # Per-queue stats
     ```

**Deliverables**:
- Directory: `p2p/tcpx/logs/irq_info/` containing:
  - `irq_snapshot_node0.txt`, `irq_snapshot_node1.txt`
  - `irq_affinity_node0.txt`, `irq_affinity_node1.txt`
  - `nic_irq_mapping.md` (human-readable table)
  - `dp_manager_status.txt`

---

### 1.3 Baseline P2P Benchmark Performance

**Objective**: Establish detailed performance baseline before any IRQ tuning.

**Tasks**:

1. **Run comprehensive baseline tests**
   - Single-NIC tests (eth1, eth2, eth3, eth4 separately)
   - All 8 GPUs in fullmesh mode
   - Capture:
     - Bandwidth (server and client)
     - Request depth / window utilization
     - CPU utilization per core (via `mpstat -P ALL 1`)
     - IRQ counts before/after (from `/proc/interrupts`)

2. **Profile CPU usage during test**
   - Run test with `perf` monitoring:
     ```bash
     # On server node
     perf record -a -g -o perf_baseline_server.data -- sleep 30 &
     # Start P2P test
     # After test:
     perf report -i perf_baseline_server.data > perf_baseline_server.txt
     ```
   - Identify hotspots: which cores are busy, what functions dominate

3. **Capture network-level metrics**
   - During test, monitor:
     ```bash
     # Per-NIC packet rates and errors
     watch -n 1 'ethtool -S eth1 | grep -E "rx_packets|tx_packets|rx_errors|tx_errors"'
     
     # CPU softirq load
     mpstat -I SUM 1
     ```

4. **Document request/window behavior**
   - From P2P logs, extract:
     - Average outstanding requests per channel
     - Window full events (if logged)
     - Timeout occurrences
     - Chunk processing latency

**Deliverables**:
- Document: `docs/BASELINE_PERFORMANCE.md` with:
  - Bandwidth table (per GPU, per NIC)
  - CPU utilization summary (which cores are hot)
  - IRQ distribution (which IRQs fired most)
  - Request depth statistics
  - Identified bottlenecks (CPU-bound? Network-bound? Synchronization?)

---

## Step 2: Design Matching Binding Policy

### 2.1 Build IRQ-to-CPU Mapping Table

**Objective**: Create a comprehensive mapping that matches NCCL's strategy.

**Tasks**:

1. **Analyze NCCL's binding logic**
   - From Step 1.1, extract the exact core ranges NCCL uses
   - Example from current config:
     ```
     TX_BINDINGS: eth1:8-21,112-125; eth2:8-21,112-125; eth3:60-73,164-177; eth4:60-73,164-177
     RX_BINDINGS: eth1:22-35,126-139; eth2:22-35,126-139; eth3:74-87,178-191; eth4:74-87,178-191
     ```
   - Understand the pattern:
     - Why these specific ranges?
     - How do they relate to NUMA topology?
     - Are hyperthreads included (e.g., 112-125 = hyperthreads of 8-21)?

2. **Map to hardware topology**
   - Cross-reference with `lscpu` output:
     - Which cores are on NUMA 0 vs NUMA 1?
     - Which are physical cores vs hyperthreads?
     - Which cores are closest to each NIC's PCIe root?
   - Create table:
     ```
     NIC   | NUMA | Physical Cores | Hyperthreads | TX Cores (NCCL) | RX Cores (NCCL)
     ------|------|----------------|--------------|-----------------|----------------
     eth1  | 0    | 0-51           | 104-155      | 8-21,112-125    | 22-35,126-139
     eth2  | 0    | 0-51           | 104-155      | 8-21,112-125    | 22-35,126-139
     eth3  | 1    | 52-103         | 156-207      | 60-73,164-177   | 74-87,178-191
     eth4  | 1    | 52-103         | 156-207      | 60-73,164-177   | 74-87,178-191
     ```

3. **Design IRQ affinity policy**
   - Decide on IRQ steering strategy:
     - **Option A**: Static affinity (set once at boot/test start)
       - Pro: Simple, predictable
       - Con: May not adapt to load
     - **Option B**: Dynamic (let dp-manager handle)
       - Pro: Potentially better load balancing
       - Con: Less control, harder to debug
     - **Recommendation**: Start with static, matching NCCL's pattern

4. **Define per-NIC IRQ masks**
   - For each NIC, calculate CPU affinity bitmask:
     - eth1 RX IRQs → cores 22-35,126-139
     - eth1 TX IRQs → cores 8-21,112-125
     - (repeat for eth2-4)
   - Convert to hex masks for `/proc/irq/*/smp_affinity`
   - Document in table format

**Deliverables**:
- Document: `docs/IRQ_BINDING_DESIGN.md` with:
  - Complete NIC-to-core mapping table
  - Rationale for each binding decision
  - CPU affinity masks (both human-readable ranges and hex values)
  - Comparison with NCCL's configuration

---

### 2.2 Decide on Implementation Approach

**Objective**: Choose how to apply IRQ bindings in the P2P benchmark.

**Options**:

1. **Approach A: Pre-test IRQ setup script**
   - Create `scripts/setup_irq_affinity.sh`
   - Run once before benchmark (requires sudo)
   - Sets all IRQ affinities system-wide
   - Pros: Simple, one-time setup
   - Cons: Affects all processes, requires root

2. **Approach B: Per-process binding in launcher**
   - Modify `run_p2p_fullmesh.sh` to set IRQ affinities before each GPU process
   - Use `taskset` or `numactl` to pin process to cores
   - Pros: Process-isolated, more flexible
   - Cons: More complex, may conflict with dp-manager

3. **Approach C: Hybrid (recommended)**
   - Use pre-test script for IRQ affinities (system-level)
   - Use launcher for thread bindings (process-level)
   - Export NCCL binding env vars to let TCPX plugin handle thread affinity
   - Pros: Matches NCCL's model, clear separation of concerns
   - Cons: Requires both system and process-level changes

**Decision Criteria**:
- Does dp-manager override our IRQ settings?
- Can we run with sudo on test nodes?
- Do we need per-GPU customization or uniform policy?

**Deliverables**:
- Document: `docs/BINDING_IMPLEMENTATION_APPROACH.md` with:
  - Chosen approach and justification
  - Required permissions/prerequisites
  - Interaction with dp-manager
  - Rollback/cleanup procedure

---

## Step 3: Implement Bindings in the Benchmark

### 3.1 Create IRQ Affinity Setup Script

**Objective**: Automate IRQ affinity configuration to match NCCL.

**Tasks**:

1. **Script structure** (`scripts/setup_tcpx_irq_affinity.sh`):
   - Parse NIC-to-core mapping from config file or hardcoded table
   - For each NIC (eth1-4):
     - Identify all associated IRQ numbers (from `/proc/interrupts`)
     - Determine if IRQ is TX or RX (from IRQ name/description)
     - Calculate appropriate CPU mask
     - Write mask to `/proc/irq/<irq>/smp_affinity`
   - Log all changes for verification
   - Provide `--dry-run` mode to preview changes
   - Provide `--reset` mode to restore defaults

2. **Safety features**:
   - Backup current affinity settings before changes
   - Validate IRQ numbers exist before writing
   - Check for write permissions (require sudo)
   - Verify changes took effect (read back and compare)

3. **Integration with test workflow**:
   - Add to `run_p2p_fullmesh.sh` as optional pre-flight check
   - Document in README how to run manually
   - Consider adding to node setup/provisioning scripts

**Deliverables**:
- Script: `scripts/setup_tcpx_irq_affinity.sh`
- Config: `scripts/tcpx_irq_config.yaml` (NIC-to-core mappings)
- Documentation: `scripts/README_IRQ_SETUP.md`

---

### 3.2 Update P2P Benchmark Launcher

**Objective**: Export NCCL-compatible binding environment variables.

**Tasks**:

1. **Add binding configuration to `run_p2p_fullmesh.sh`**:
   - Define TX/RX binding strings matching NCCL format
   - Export before launching each GPU process:
     ```bash
     export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;..."
     export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;..."
     ```
   - Optionally add IRQ bindings if supported:
     ```bash
     export NCCL_GPUDIRECTTCPX_TX_IRQ_BINDINGS="..."
     export NCCL_GPUDIRECTTCPX_RX_IRQ_BINDINGS="..."
     ```

2. **Add per-GPU process pinning** (optional):
   - Use `taskset` to pin each GPU process to NUMA-local cores
   - Example: GPU 0 → cores 0-51 (NUMA 0), GPU 4 → cores 52-103 (NUMA 1)
   - Ensure pinning doesn't conflict with TX/RX thread bindings

3. **Add validation checks**:
   - Before test, verify:
     - IRQ affinities are set correctly (if using setup script)
     - Environment variables are exported
     - No conflicting bindings
   - Log configuration summary at test start

**Deliverables**:
- Updated: `p2p/tcpx/run_p2p_fullmesh.sh`
- Documentation: Update `p2p/tcpx/README.md` with binding options

---

### 3.3 Extend ChannelManager (if needed)

**Objective**: Allow per-channel thread affinity overrides.

**Tasks**:

1. **Assess current ChannelManager capabilities**:
   - Review `p2p/tcpx/src/channel_manager.cc`
   - Check if TCPX plugin already handles thread affinity via env vars
   - Determine if additional control is needed

2. **If extension needed**:
   - Add API to set thread affinity per channel
   - Implement using `pthread_setaffinity_np()` or similar
   - Expose via command-line flags or config file
   - Log actual thread-to-core assignments

3. **If not needed**:
   - Document that TCPX plugin handles this internally
   - Verify via logs that threads are pinned correctly

**Deliverables**:
- Analysis: `docs/CHANNEL_MANAGER_BINDING_ANALYSIS.md`
- Code changes: Only if needed, otherwise document why not

---

## Step 4: Instrument and Test

### 4.1 Add Binding Verification Logging

**Objective**: Confirm bindings are applied correctly at runtime.

**Tasks**:

1. **Add pre-test diagnostics**:
   - In `run_p2p_fullmesh.sh`, before launching processes:
     - Dump current IRQ affinities for all gVNIC IRQs
     - Verify exported environment variables
     - Check CPU topology (NUMA nodes, core counts)
   - Save to `logs/binding_verification_<timestamp>.log`

2. **Add runtime thread affinity logging**:
   - In test program or ChannelManager:
     - Log actual CPU affinity of TX/RX threads after creation
     - Use `sched_getaffinity()` to read current mask
     - Compare with expected values from env vars
   - Example log line:
     ```
     [ChannelManager] Channel 0 TX thread: expected cores 8-21,112-125, actual cores 8-21,112-125 ✓
     ```

3. **Add optional verbose mode**:
   - Flag: `--verbose-bindings` or `UCCL_TCPX_DEBUG_BINDINGS=1`
   - When enabled, log:
     - Every IRQ affinity read/write
     - Every thread creation and affinity setting
     - CPU usage per core during test (via periodic sampling)

**Deliverables**:
- Enhanced logging in launcher and test program
- Example verification log showing correct bindings

---

### 4.2 Run Controlled Experiments

**Objective**: Measure performance impact of IRQ binding changes.

**Test Matrix**:

| Test ID | IRQ Affinity | Thread Bindings | Channels | Expected Outcome |
|---------|--------------|-----------------|----------|------------------|
| T1      | Default      | None            | 1        | Baseline (2.75 GB/s) |
| T2      | NCCL-matched | None            | 1        | Slight improvement? |
| T3      | NCCL-matched | NCCL-matched    | 1        | Moderate improvement |
| T4      | NCCL-matched | NCCL-matched    | 2        | Test multi-channel (may fail) |
| T5      | NCCL-matched | NCCL-matched    | 4        | Test higher concurrency |

**For each test**:

1. **Setup**:
   - Apply IRQ affinity configuration (if applicable)
   - Set environment variables
   - Clear any cached state

2. **Run**:
   - Execute fullmesh test (all 8 GPUs)
   - Capture logs, bandwidth, CPU usage
   - Monitor IRQ counts during test

3. **Collect metrics**:
   - Bandwidth (server/client, per GPU, aggregate)
   - CPU utilization (per core, per NUMA node)
   - IRQ distribution (which cores handled most interrupts)
   - Request depth / window utilization
   - Any errors or warnings

4. **Compare**:
   - Against baseline (T1)
   - Against NCCL performance target
   - Identify which changes had most impact

**Deliverables**:
- Test results: `docs/BINDING_TEST_RESULTS.md`
- Performance graphs: Bandwidth vs test ID
- Analysis: Which bindings matter most?

---

### 4.3 Profile CPU and Network Behavior

**Objective**: Understand where time is spent and identify remaining bottlenecks.

**Tasks**:

1. **CPU profiling with perf**:
   - Run best-performing configuration with `perf record`
   - Analyze:
     - Which functions consume most CPU?
     - Are TX/RX threads on correct cores?
     - Is there unexpected contention (locks, atomics)?
   - Compare with NCCL profile (if available)

2. **Network queue analysis**:
   - During test, capture:
     ```bash
     ethtool -S eth1 | grep -E "queue|drop|error"
     ```
   - Check for:
     - Queue drops (indicates overload)
     - Uneven queue utilization (indicates poor RSS/IRQ balance)
     - Error counters

3. **Interrupt distribution analysis**:
   - Before/after snapshots of `/proc/interrupts`
   - Calculate IRQs per core
   - Verify IRQs are hitting intended cores
   - Check for IRQ imbalance

4. **Thread scheduling analysis**:
   - Use `pidstat` or `top` to monitor per-thread CPU usage
   - Verify TX/RX threads are not migrating between cores
   - Check for idle cores (wasted capacity)

**Deliverables**:
- Profile reports: `logs/perf_analysis_<config>.txt`
- IRQ distribution analysis: `logs/irq_distribution_<config>.md`
- Bottleneck summary: `docs/BOTTLENECK_ANALYSIS.md`

---

## Step 5: Iterate and Refine

### 5.1 Analyze Results and Identify Gaps

**Objective**: Determine why performance still lags NCCL (if it does).

**Questions to answer**:

1. **Is IRQ binding effective?**
   - Did IRQ affinity changes improve bandwidth?
   - Are interrupts balanced across cores?
   - Is CPU utilization more efficient?

2. **Are thread bindings correct?**
   - Are TX/RX threads on intended cores?
   - Is there core contention or migration?
   - Do bindings match NCCL's pattern exactly?

3. **What's the remaining gap?**
   - If still far from NCCL performance, what else differs?
   - Channel count? (1 vs 8)
   - Communication pattern? (P2P vs collective)
   - Request pipelining depth?
   - Chunk size or other tuning parameters?

4. **Are there new bottlenecks?**
   - After fixing IRQ/thread issues, what's the new limiting factor?
   - CPU-bound? Memory bandwidth? Network?

**Deliverables**:
- Analysis document: `docs/PERFORMANCE_GAP_ANALYSIS.md`
- Prioritized list of next optimization targets

---

### 5.2 Sweep Parameter Space

**Objective**: Find optimal configuration within binding framework.

**Parameters to sweep**:

1. **Core range adjustments**:
   - Try wider/narrower core ranges
   - Test with/without hyperthreads
   - Experiment with core isolation (isolcpus)

2. **Channel count** (if multi-channel becomes viable):
   - Test 1, 2, 4, 8 channels
   - Measure scaling efficiency
   - Identify point of diminishing returns

3. **Chunk size**:
   - Current: 524KB
   - Try: 256KB, 1MB, 2MB
   - Measure impact on bandwidth and latency

4. **Request window depth**:
   - Current: 16 outstanding requests
   - Try: 8, 32, 64
   - Find optimal pipelining depth

5. **Socket/thread counts**:
   - Current: 4 sockets, 1 thread
   - Try: 2 sockets/2 threads, 8 sockets/1 thread
   - Match NCCL's configuration

**Methodology**:
- Change one parameter at a time
- Run multiple iterations for statistical significance
- Document all configurations tested
- Build performance heatmap (e.g., bandwidth vs channels vs chunk size)

**Deliverables**:
- Parameter sweep results: `docs/PARAMETER_SWEEP_RESULTS.md`
- Optimal configuration: `configs/optimal_tcpx_config.yaml`

---

### 5.3 Advanced Tuning (if needed)

**Objective**: Apply additional optimizations if basic binding doesn't close the gap.

**Potential areas**:

1. **Queue depth tuning**:
   - Increase NIC RX/TX queue sizes (ethtool -G)
   - Tune kernel network buffers (sysctl net.core.*)
   - Match NCCL's queue configuration

2. **CPU frequency scaling**:
   - Disable CPU frequency scaling (set to performance governor)
   - Lock cores to max frequency
   - Reduce jitter from frequency transitions

3. **NUMA balancing**:
   - Disable automatic NUMA balancing (may cause migrations)
   - Explicitly bind memory allocations to local NUMA node
   - Use `numactl --membind` for GPU memory

4. **Interrupt coalescing**:
   - Tune interrupt coalescing parameters (ethtool -C)
   - Balance between latency and throughput
   - Match NCCL/dp-manager settings

5. **Kernel tuning**:
   - Increase network buffer sizes
   - Tune TCP parameters (if applicable)
   - Adjust scheduler settings (e.g., CFS bandwidth)

**Deliverables**:
- Advanced tuning guide: `docs/ADVANCED_TUNING.md`
- System configuration script: `scripts/tune_system_for_tcpx.sh`

---

### 5.4 Document Final Configuration

**Objective**: Create reproducible setup for optimal performance.

**Deliverables**:

1. **Configuration file**: `configs/production_tcpx_config.yaml`
   - All IRQ affinity settings
   - All environment variables
   - All tuning parameters
   - Expected performance metrics

2. **Setup guide**: `docs/PRODUCTION_SETUP_GUIDE.md`
   - Step-by-step instructions
   - Prerequisites and dependencies
   - Verification steps
   - Troubleshooting common issues

3. **Performance report**: `docs/FINAL_PERFORMANCE_REPORT.md`
   - Before/after comparison
   - Breakdown of improvements by optimization
   - Remaining gap to NCCL (if any) and explanation
   - Recommendations for future work

4. **Handoff package**:
   - All scripts, configs, and docs
   - Test results and logs
   - Known issues and limitations
   - Contact info for questions

---

## Success Criteria

### Minimum Success
- [ ] IRQ affinities match NCCL's configuration
- [ ] Thread bindings verified correct via logs
- [ ] Bandwidth improves by at least 20% over baseline
- [ ] Configuration is documented and reproducible

### Target Success
- [ ] Bandwidth reaches 50% of NCCL's performance (~9 GB/s per GPU)
- [ ] CPU utilization is efficient (no hot cores, balanced load)
- [ ] IRQ distribution is balanced across intended cores
- [ ] Multi-channel configuration works without devmem conflicts

### Stretch Success
- [ ] Bandwidth reaches 80%+ of NCCL's performance (~15 GB/s per GPU)
- [ ] Understand and document all remaining performance gaps
- [ ] Provide clear path to close remaining gaps (if any)

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Step 1: Gather info | 2-3 hours | Access to both nodes, sudo |
| Step 2: Design | 1-2 hours | Step 1 complete |
| Step 3: Implement | 3-4 hours | Step 2 complete |
| Step 4: Test | 4-6 hours | Step 3 complete |
| Step 5: Iterate | 2-8 hours | Step 4 results |
| **Total** | **12-23 hours** | Assumes no major blockers |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| dp-manager overrides our IRQ settings | High | Test with dp-manager disabled, or coordinate with it |
| Multi-channel still fails due to devmem conflicts | Medium | Focus on single-channel optimization first |
| IRQ binding has minimal impact | Medium | Proceed to other optimizations (chunk size, etc.) |
| Requires root access we don't have | High | Work with sysadmin or use alternative approaches |
| NCCL's advantage is purely algorithmic (ring) | High | Document gap, focus on P2P-specific optimizations |

---

## Next Immediate Actions

1. **Run NCCL test with full logging** (Step 1.1, task 2)
   - Modify `run_nccl_test_tcpx.sh` to add debug flags
   - Capture complete initialization and binding logs
   - Extract exact binding configuration used

2. **Collect IRQ snapshots** (Step 1.2, task 1)
   - On both nodes, capture `/proc/interrupts` and affinity masks
   - Identify gVNIC IRQs and their current CPU assignments
   - Document in structured format

3. **Run baseline P2P test with profiling** (Step 1.3)
   - Single-NIC, 1-channel configuration (known working)
   - Capture bandwidth, CPU usage, IRQ distribution
   - Establish clear baseline for comparison

**Estimated time for immediate actions**: 2-3 hours

