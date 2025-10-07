# IRQ Binding Investigation Plan

## Executive Summary

This document outlines a systematic plan to investigate and potentially improve P2P TCPX performance by matching NCCL's IRQ binding and CPU affinity strategy. The current P2P benchmark achieves ~2.75 GB/s per GPU (server) and ~1.17 GB/s (client), while NCCL AllReduce achieves ~18.7 GB/s bus bandwidth. One hypothesis is that NCCL's careful IRQ and thread CPU affinity tuning contributes to its superior performance.

## Background

### Current Status
- **P2P TCPX benchmark**: Single-NIC, single-channel per GPU, verified working
- **NCCL+TCPX**: Uses all 4 NICs, 8 channels per GPU, significantly higher throughput
- **Known difference**: NCCL explicitly sets `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `NCCL_GPUDIRECTTCPX_RX_BINDINGS`

### Hardware Topology (A3-high GCE nodes)
- 8x H100 GPUs per node
- 4x gVNIC interfaces (eth1-4), 100 Gbps each
- 2 NUMA domains:
  - NUMA 0: GPU 0-3, eth1-2
  - NUMA 1: GPU 4-7, eth3-4
- 208 CPU cores total (104 physical cores × 2 hyperthreads)

### NCCL's Binding Configuration (from run_nccl_test_tcpx.sh)
```
NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

**Pattern analysis**:
- eth1/eth2 (NUMA 0): TX cores 8-21,112-125 (28 cores), RX cores 22-35,126-139 (28 cores)
- eth3/eth4 (NUMA 1): TX cores 60-73,164-177 (28 cores), RX cores 74-87,178-191 (28 cores)
- Each NIC gets 28 TX cores + 28 RX cores = 56 cores total
- Cores are NUMA-local to the NICs

---

## Step 1: Gather Current IRQ and Binding Information

### Objective
Understand the complete IRQ and CPU affinity landscape on both nodes to establish a baseline and identify what NCCL is doing differently.

### 1.1 Collect Hardware IRQ Assignments

**On both Node 0 and Node 1**, run the following commands and save outputs:

#### a) IRQ table snapshot
```bash
cat /proc/interrupts > irq_snapshot_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- IRQ numbers for `gve` (gVNIC driver) entries
- Which CPUs are handling each IRQ (columns show per-CPU interrupt counts)
- Identify IRQs for eth1, eth2, eth3, eth4

#### b) IRQ affinity masks
```bash
for irq in $(grep gve /proc/interrupts | awk '{print $1}' | tr -d ':'); do
  echo "IRQ $irq: $(cat /proc/irq/$irq/smp_affinity 2>/dev/null || echo 'N/A')"
done > irq_affinity_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- Default CPU affinity masks (hexadecimal bitmasks)
- Whether IRQs are pinned to specific cores or spread across all cores

#### c) NIC-to-IRQ mapping
```bash
for nic in eth1 eth2 eth3 eth4; do
  echo "=== $nic ==="
  ethtool -S $nic | grep -i irq || echo "No IRQ stats"
  echo ""
done > nic_irq_mapping_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- Which IRQ numbers correspond to which NIC
- TX queue IRQs vs RX queue IRQs

#### d) PCI and NUMA topology
```bash
for nic in eth1 eth2 eth3 eth4; do
  pci_addr=$(ethtool -i $nic | grep bus-info | awk '{print $2}')
  numa_node=$(cat /sys/class/net/$nic/device/numa_node 2>/dev/null || echo "N/A")
  echo "$nic: PCI=$pci_addr NUMA=$numa_node"
done > nic_numa_topology_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- Confirm eth1/eth2 on NUMA 0, eth3/eth4 on NUMA 1

### 1.2 Capture NCCL Runtime Behavior

#### a) Run NCCL test with verbose logging
Modify `run_nccl_test_tcpx.sh` to add:
```bash
-x NCCL_DEBUG=TRACE
-x NCCL_DEBUG_SUBSYS=ALL
```
Save output to a file:
```bash
./run_nccl_test_tcpx.sh nccl 2 8 0 1 2>&1 | tee nccl_trace_$(date +%Y%m%d_%H%M%S).log
```

**What to look for in logs**:
- Messages about thread affinity: `"thread X running on cpu Y"`
- Flow steering setup messages
- Any mentions of IRQ programming or CPU binding
- Channel creation and NIC assignment per channel

#### b) Monitor CPU utilization during NCCL test
In a separate terminal while NCCL test is running:
```bash
# Sample CPU usage every 1 second for 60 seconds
mpstat -P ALL 1 60 > nccl_cpu_usage_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- Which CPU cores show high utilization during the test
- Whether utilization matches the binding ranges (8-21, 22-35, etc.)

#### c) Monitor IRQ distribution during NCCL test
```bash
# Before test
cat /proc/interrupts > irq_before_nccl.txt

# Run NCCL test (let it run for ~30 seconds)

# After test
cat /proc/interrupts > irq_after_nccl.txt

# Diff to see which IRQs fired
diff irq_before_nccl.txt irq_after_nccl.txt > irq_delta_nccl.txt
```
**What to look for**:
- Which gVNIC IRQs incremented significantly
- Which CPUs handled those IRQs (compare columns)

### 1.3 Capture P2P Benchmark Baseline

#### a) Run current P2P test with profiling
```bash
# On Node 0 (server)
./run_p2p_fullmesh.sh server 2>&1 | tee p2p_baseline_server_$(date +%Y%m%d_%H%M%S).log

# On Node 1 (client)
./run_p2p_fullmesh.sh client <NODE0_IP> 2>&1 | tee p2p_baseline_client_$(date +%Y%m%d_%H%M%S).log
```

#### b) Monitor CPU usage during P2P test
```bash
mpstat -P ALL 1 60 > p2p_cpu_usage_$(date +%Y%m%d_%H%M%S).txt
```

#### c) Monitor IRQ distribution during P2P test
```bash
cat /proc/interrupts > irq_before_p2p.txt
# Run P2P test
cat /proc/interrupts > irq_after_p2p.txt
diff irq_before_p2p.txt irq_after_p2p.txt > irq_delta_p2p.txt
```

#### d) Extract baseline metrics from logs
From the P2P logs, record:
- Bandwidth per GPU (server and client)
- Total bandwidth
- Request depth / outstanding chunks
- Any warnings or errors

### 1.4 Check for dp-manager (Flow Steering Daemon)

```bash
# Check if dp-manager is running
ps aux | grep dp-manager

# Check dp-manager logs if available
journalctl -u dp-manager --since "1 hour ago" > dp_manager_logs_$(date +%Y%m%d_%H%M%S).txt
```
**What to look for**:
- Whether dp-manager is actively managing flow steering
- Any errors or warnings related to IRQ or flow steering

---

## Step 2: Design Matching Binding Policy

### Objective
Based on data from Step 1, design a CPU binding strategy for the P2P benchmark that matches or approximates NCCL's approach.

### 2.1 Build IRQ-to-NIC-to-CPU Mapping Table

Create a table (spreadsheet or markdown) with columns:
- NIC name (eth1-4)
- NUMA node
- IRQ numbers (list all)
- Default CPU affinity (from /proc/irq/*/smp_affinity)
- Observed CPU usage (from mpstat during NCCL test)
- NCCL TX binding range
- NCCL RX binding range

**Example**:
| NIC  | NUMA | IRQ Numbers | Default Affinity | NCCL TX Cores | NCCL RX Cores |
|------|------|-------------|------------------|---------------|---------------|
| eth1 | 0    | 45,46,47... | 0xFFFF...        | 8-21,112-125  | 22-35,126-139 |
| eth2 | 0    | 50,51,52... | 0xFFFF...        | 8-21,112-125  | 22-35,126-139 |
| eth3 | 1    | 55,56,57... | 0xFFFF...        | 60-73,164-177 | 74-87,178-191 |
| eth4 | 1    | 58,59,60... | 0xFFFF...        | 60-73,164-177 | 74-87,178-191 |

### 2.2 Decide on Binding Strategy

**Option A: Static IRQ affinity + environment variables**
- Pre-configure IRQ affinity masks to match NCCL's CPU ranges
- Export `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `RX_BINDINGS` in `run_p2p_fullmesh.sh`
- Let TCPX plugin handle thread pinning based on environment variables

**Option B: Dynamic per-process binding**
- Each GPU process calculates which NIC it will use
- Script sets IRQ affinity for that NIC before launching the process
- Export per-process binding environment variables

**Option C: Hybrid approach**
- Set global IRQ affinity once at system startup (or in a setup script)
- Export NCCL-style binding environment variables in the benchmark launcher
- Optionally extend ChannelManager to verify/enforce thread affinity

**Recommendation**: Start with **Option A** (simplest, closest to NCCL's approach).

### 2.3 Determine CPU Core Allocation

Based on NCCL's pattern:
- **For GPU 0-3 (NUMA 0)**: Use cores 8-35 and 112-139
  - TX: 8-21, 112-125 (28 cores)
  - RX: 22-35, 126-139 (28 cores)
- **For GPU 4-7 (NUMA 1)**: Use cores 60-87 and 164-191
  - TX: 60-73, 164-177 (28 cores)
  - RX: 74-87, 178-191 (28 cores)

**Question to answer from Step 1 data**:
- Are these physical cores or logical cores (hyperthreads)?
- Do the IRQ handlers run on the same cores, or different ones?
- Is there overlap between TX/RX cores and IRQ cores?

### 2.4 Plan IRQ Affinity Masks

For each NIC, calculate the CPU affinity bitmask that corresponds to the desired cores.

**Example for eth1 (NUMA 0)**:
- Desired cores: 8-35, 112-139 (56 cores total)
- Convert to hexadecimal bitmask (use a script or online calculator)
- Write to `/proc/irq/<irq>/smp_affinity`

**Note**: This requires root/sudo access. Plan to either:
- Run a setup script with sudo before the benchmark
- Integrate into a systemd service or startup script
- Use `sudo` within `run_p2p_fullmesh.sh` (with appropriate permissions)

---

## Step 3: Implement Bindings in the Benchmark

### Objective
Modify `run_p2p_fullmesh.sh` and related scripts to apply the designed binding policy.

### 3.1 Create IRQ Affinity Setup Script

**File**: `p2p/tcpx/scripts/setup_irq_affinity.sh`

**Functionality**:
- Parse NIC-to-IRQ mapping (from `/proc/interrupts` or ethtool)
- For each NIC, set IRQ affinity to the calculated CPU mask
- Log all changes for verification
- Require root/sudo

**Inputs**:
- NIC name (eth1-4)
- Desired CPU core range (e.g., "8-21,112-125")

**Outputs**:
- Modified `/proc/irq/*/smp_affinity` files
- Log file with before/after affinity values

### 3.2 Modify run_p2p_fullmesh.sh

**Changes needed**:

#### a) Add binding environment variables
Export the same variables NCCL uses:
```bash
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

#### b) Call IRQ setup script (optional, if using static IRQ affinity)
```bash
# Before launching GPU processes
if [[ "${SETUP_IRQ_AFFINITY:-0}" == "1" ]]; then
  sudo ./scripts/setup_irq_affinity.sh
fi
```

#### c) Add verbose logging flag
```bash
VERBOSE=${VERBOSE:-0}
if [[ "${VERBOSE}" == "1" ]]; then
  # Dump IRQ affinity, CPU bindings, etc.
fi
```

### 3.3 Extend ChannelManager (Optional)

**File**: `p2p/tcpx/src/channel_manager.cc`

**Potential enhancements**:
- Read `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `RX_BINDINGS` environment variables
- Parse core ranges for the selected NIC
- Use `pthread_setaffinity_np()` to pin TCPX threads to specified cores
- Log actual thread affinity for verification

**Note**: This may not be necessary if the TCPX plugin already handles these environment variables internally. Verify from Step 1 NCCL logs whether TCPX plugin respects these variables.

### 3.4 Add Pre-flight Checks

Before running the benchmark, verify:
- IRQ affinity is set correctly (compare `/proc/irq/*/smp_affinity` to expected values)
- Environment variables are exported
- No conflicting CPU affinity settings (e.g., `taskset`, `numactl`)

**Script**: `p2p/tcpx/scripts/verify_bindings.sh`

---

## Step 4: Instrument and Test

### Objective
Run controlled experiments to measure the impact of IRQ/CPU bindings on performance.

### 4.1 Add Instrumentation

#### a) Extend benchmark logging
In `test_tcpx_perf_multi.cc`, add:
- Log environment variables at startup (`NCCL_GPUDIRECTTCPX_*`)
- Log actual thread CPU affinity (read from `/proc/self/task/<tid>/status`)
- Log IRQ counts before/after test (read `/proc/interrupts`)

#### b) Create a verbose mode
```bash
export UCCL_TCPX_VERBOSE=1
```
When enabled, dump:
- IRQ affinity for all gVNIC IRQs
- CPU binding ranges
- Thread-to-CPU mapping
- NIC-to-channel mapping

### 4.2 Test Matrix

Run the following test configurations and record results:

| Test ID | IRQ Affinity | Env Vars | Channels | Expected Outcome |
|---------|--------------|----------|----------|------------------|
| T1      | Default      | None     | 1        | Baseline (~2.75 GB/s) |
| T2      | Default      | NCCL-style | 1      | Check if env vars help |
| T3      | NCCL-matched | None     | 1        | Check if IRQ affinity helps |
| T4      | NCCL-matched | NCCL-style | 1      | Combined effect |
| T5      | NCCL-matched | NCCL-style | 2      | Multi-channel (if safe) |

**For each test**:
- Run 3 iterations for statistical significance
- Record: bandwidth (server/client), CPU usage, IRQ counts
- Save full logs

### 4.3 Collect Metrics

**Performance metrics**:
- Bandwidth per GPU (GB/s)
- Total aggregate bandwidth
- Latency (if measured)
- Request depth / outstanding chunks

**System metrics**:
- CPU utilization per core (from mpstat)
- IRQ counts per NIC (from /proc/interrupts delta)
- Context switches (from `pidstat` or `perf`)
- Cache misses, NUMA remote accesses (from `perf stat`)

**Comparison**:
- Compare each test to baseline (T1)
- Compare best P2P result to NCCL AllReduce

### 4.4 Validation Checks

After each test run:
- Verify no "rx no cmsg" errors
- Verify no GPU errors or hangs
- Check that IRQ affinity remained stable (didn't get reset)
- Confirm CPU cores in the binding range showed activity (from mpstat)

---

## Step 5: Iterate and Refine

### Objective
Based on test results, refine the binding strategy and identify remaining bottlenecks.

### 5.1 Analyze Results

**If performance improves**:
- Quantify the improvement (e.g., "20% bandwidth increase")
- Identify which change had the most impact (IRQ affinity vs env vars)
- Document the winning configuration

**If performance does NOT improve**:
- Check instrumentation data:
  - Are threads actually running on the bound cores? (verify with `ps -eLo pid,tid,psr,comm`)
  - Are IRQs being handled by the bound cores? (check /proc/interrupts delta)
  - Is CPU utilization high on the bound cores? (check mpstat)
- Look for other bottlenecks:
  - Queue depth too low (increase `UCCL_TCPX_WINDOW_SIZE`)
  - Chunk size suboptimal (sweep `UCCL_TCPX_CHUNK_BYTES`)
  - Flow steering not working (check dp-manager logs)
  - NUMA remote memory access (check `numastat`)

### 5.2 Refine Binding Strategy

**Potential adjustments**:

#### a) Expand or contract CPU core ranges
- If cores are underutilized, reduce the range (fewer cores, less contention)
- If cores are saturated, expand the range (more parallelism)

#### b) Separate TX and RX cores more strictly
- Ensure no overlap between TX and RX ranges
- Pin IRQ handlers to separate cores from TX/RX threads

#### c) Use physical cores only (disable hyperthreads)
- If hyperthreading causes contention, restrict to physical cores
- Example: cores 8-21 (physical) instead of 8-21,112-125 (physical + HT)

#### d) Per-channel affinity
- If using multiple channels, assign each channel to a different core subset
- Example: Channel 0 on cores 8-14, Channel 1 on cores 15-21

### 5.3 Investigate Queue Depth and Pipelining

If CPU binding doesn't help, the bottleneck may be elsewhere:

#### a) Increase request depth
```bash
export UCCL_TCPX_WINDOW_SIZE=32  # or 64
```
More outstanding requests → better pipeline utilization

#### b) Tune chunk size
```bash
export UCCL_TCPX_CHUNK_BYTES=1048576  # 1 MB
```
Larger chunks → fewer syscalls, better throughput

#### c) Enable multi-channel (if devmem conflict is resolved)
- Revisit multi-NIC approach with proper IRQ binding
- Hypothesis: with correct IRQ affinity, multi-channel may work

### 5.4 Profile with perf

If bottleneck is still unclear, use `perf` to profile:

```bash
# Profile the benchmark process
perf record -g -p <PID> -- sleep 30
perf report

# Look for:
# - High CPU time in specific functions (e.g., recvmsg, unpack kernel)
# - Cache misses
# - NUMA remote accesses
```

### 5.5 Compare with NCCL at Detailed Level

**Run NCCL with same parameters as P2P**:
- Single GPU pair (not AllReduce)
- Same data size (64 MB)
- Same number of iterations

**Use NCCL's sendrecv_perf**:
```bash
./run_nccl_test_tcpx.sh nccl 2 1 2 2  # sendrecv mode, 1 GPU per process
```

Compare:
- Bandwidth
- CPU usage pattern
- IRQ distribution
- Thread affinity (from NCCL logs)

---

## Step 6: Document and Decide

### 6.1 Create Summary Report

**File**: `p2p/tcpx/docs/IRQ_BINDING_RESULTS.md`

**Contents**:
- Summary of all tests run (table with results)
- Best configuration found
- Performance improvement (if any)
- Remaining gap to NCCL (if any)
- Lessons learned

### 6.2 Decision Point

Based on results, decide:

**Option A: IRQ binding significantly helps**
- Integrate the binding setup into the standard benchmark workflow
- Document the configuration
- Consider upstreaming to NIXL plugin

**Option B: IRQ binding has minimal impact**
- Document the negative result (important!)
- Focus investigation on other areas:
  - Multi-channel devmem conflict (Step 1 finding)
  - NCCL's Ring algorithm vs P2P
  - Kernel-level optimizations in NCCL

**Option C: Partial improvement, but gap remains**
- Keep the binding configuration
- Continue investigating other factors
- Consider hybrid approach (P2P for specific use cases, NCCL for others)

---

## Appendix: Quick Reference Commands

### Check IRQ affinity for a NIC
```bash
NIC=eth1
for irq in $(grep $NIC /proc/interrupts | awk '{print $1}' | tr -d ':'); do
  echo "IRQ $irq: $(cat /proc/irq/$irq/smp_affinity)"
done
```

### Set IRQ affinity (requires root)
```bash
# Example: pin IRQ 45 to cores 8-21 (bitmask calculation needed)
echo <bitmask> | sudo tee /proc/irq/45/smp_affinity
```

### Check thread CPU affinity
```bash
# For a running process
ps -eLo pid,tid,psr,comm | grep test_tcpx
# psr = processor (CPU core) the thread is currently on
```

### Monitor CPU usage in real-time
```bash
mpstat -P ALL 1
```

### Calculate CPU bitmask for core range
```bash
# Python one-liner
python3 -c "cores=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,112,113,114,115,116,117,118,119,120,121,122,123,124,125]; mask=sum(1<<c for c in cores); print(hex(mask))"
```

---

## Timeline Estimate

- **Step 1** (Data collection): 2-3 hours
- **Step 2** (Design): 1-2 hours
- **Step 3** (Implementation): 3-4 hours
- **Step 4** (Testing): 2-3 hours
- **Step 5** (Iteration): 2-4 hours (depends on findings)
- **Step 6** (Documentation): 1 hour

**Total**: ~11-17 hours (1.5 to 2 days of focused work)

---

## Success Criteria

1. **Baseline established**: Clear understanding of current IRQ/CPU configuration
2. **NCCL behavior documented**: Know exactly what NCCL does differently
3. **Binding implemented**: P2P benchmark uses NCCL-style bindings
4. **Performance measured**: Quantitative comparison before/after
5. **Decision made**: Clear next steps based on results

**Stretch goal**: Achieve >5 GB/s per GPU (2x improvement over baseline)

