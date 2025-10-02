# Multi-NIC Debugging Guide

**Problem**: Only eth1 is being used. eth2-4 are idle.

**Impact**: Bandwidth limited to ~3 GB/s (should be ~12 GB/s with 4 NICs).

**Priority**: 🔥 CRITICAL - This is the #1 blocker for performance.

---

## 🔍 Quick Diagnosis

### Step 1: Verify Environment Variables

```bash
cd /home/daniel/uccl/p2p/tcpx
./diagnose_multi_nic.sh
```

**Expected output**:
```
NCCL_GPUDIRECTTCPX_SOCKET_IFNAME = eth1,eth2,eth3,eth4
NCCL_GPUDIRECTTCPX_CTRL_DEV = eth0
```

---

### Step 2: Run Test with Debug Logs

**Server**:
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/debug_server.log
```

**Client**:
```bash
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix 2>&1 | tee logs/debug_client.log
```

---

### Step 3: Check TCPX Logs

```bash
# Look for TCPX configuration messages
grep -i "socket_ifname\|gpudirecttcpx" logs/debug_server.log | head -20

# Look for network interface detection
grep -i "eth1\|eth2\|eth3\|eth4" logs/debug_server.log | head -20

# Look for NCCL NET messages
grep "NCCL.*NET" logs/debug_server.log | head -20
```

**What to look for**:
```
NET_GPUDIRECTTCPX_SOCKET_IFNAME set to eth1,eth2,eth3,eth4
```

**If NOT found**: TCPX plugin is not reading the environment variable.

---

### Step 4: Monitor NIC Traffic

**During test, in another terminal**:
```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

**Expected**: All 4 NICs show traffic (~1-2 GB/s each).

**Current**: Only eth1 shows traffic.

---

## 🐛 Possible Causes and Solutions

### Cause 1: TCPX Plugin Not Reading Environment Variable

**Diagnosis**:
```bash
# Check if TCPX plugin is loaded
ldd ./tests/test_tcpx_perf | grep tcpx

# Check TCPX plugin version
ls -l /usr/local/tcpx/lib64/libnccl-net.so
```

**Solution**: Verify environment variable is exported BEFORE plugin loads.

**Test**:
```bash
# Export manually and run directly
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

./tests/test_tcpx_perf server 0
```

---

### Cause 2: NCCL_CROSS_NIC=0 Preventing Multi-NIC

**Current setting**: `NCCL_CROSS_NIC=0` (in `bench_p2p.sh`)

**What it does**: Prevents cross-NIC communication.

**Solution**: Try setting to 1.

**Steps**:
1. Edit `bench_p2p.sh` line 126:
   ```bash
   export NCCL_CROSS_NIC=1  # Was 0
   ```

2. Rerun test:
   ```bash
   ./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
   ```

3. Check NIC traffic:
   ```bash
   watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
   ```

---

### Cause 3: CPU Bindings Incorrect

**Current bindings** (in `bench_p2p.sh`):
```bash
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

**Diagnosis**: Check if CPU cores match your system.

**Steps**:
1. Check NUMA topology:
   ```bash
   lscpu | grep NUMA
   numactl --hardware
   ```

2. Check NIC to NUMA mapping:
   ```bash
   cat /sys/class/net/eth1/device/numa_node
   cat /sys/class/net/eth2/device/numa_node
   cat /sys/class/net/eth3/device/numa_node
   cat /sys/class/net/eth4/device/numa_node
   ```

3. If bindings don't match, try removing them:
   ```bash
   # Comment out in bench_p2p.sh
   # export NCCL_GPUDIRECTTCPX_TX_BINDINGS=...
   # export NCCL_GPUDIRECTTCPX_RX_BINDINGS=...
   ```

---

### Cause 4: TCPX Plugin Compiled with Wrong Adapter

**TCPX plugin can use two adapters**:
- **NCCL adapter**: Uses `NCCL_GPUDIRECTTCPX_*` env vars
- **Native adapter**: Uses `NET_TCPX_*` env vars

**Diagnosis**:
```bash
# Check which adapter is used
strings /usr/local/tcpx/lib64/libnccl-net.so | grep -E "NCCL_GPUDIRECTTCPX|NET_TCPX"
```

**Solution**: If using native adapter, change env vars:
```bash
export NET_TCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NET_TCPX_CTRL_DEV=eth0
```

---

### Cause 5: TCPX Only Uses First Interface in List

**Hypothesis**: TCPX plugin may only use the first interface in the comma-separated list.

**Test**: Try setting only eth2:
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2
./tests/test_tcpx_perf server 0
```

**Check**: Does eth2 show traffic now?

**If yes**: TCPX plugin is working but only uses first interface.

**Solution**: This may be a TCPX plugin limitation. Check TCPX documentation or source code.

---

## 🧪 Systematic Debugging Steps

### Test 1: Minimal Configuration

```bash
# Remove all TCPX env vars except essential ones
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

./tests/test_tcpx_perf server 0
```

**Check logs** for TCPX configuration messages.

---

### Test 2: Single NIC Test

```bash
# Test with only eth2
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2
./tests/test_tcpx_perf server 0
```

**Check**: Does eth2 show traffic?

**If yes**: TCPX is working, but multi-NIC is not.

---

### Test 3: Two NICs Test

```bash
# Test with eth1,eth2
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2
./tests/test_tcpx_perf server 0
```

**Check**: Do both eth1 and eth2 show traffic?

**If only eth1**: TCPX only uses first interface.

---

### Test 4: NCCL_CROSS_NIC=1

```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_CROSS_NIC=1  # Enable cross-NIC
./tests/test_tcpx_perf server 0
```

**Check**: Do all 4 NICs show traffic?

---

### Test 5: Check TCPX Source Code

```bash
cd /home/daniel/uccl/nccl-plugin-gpudirecttcpx
grep -r "SOCKET_IFNAME" src/
```

**Look for**: How TCPX parses the comma-separated list.

**Check**: Does it support multiple interfaces?

---

## 📊 Expected vs Actual

### Expected (with 4 NICs)

```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'

# Expected output:
eth1: 3.0 GB/s
eth2: 3.0 GB/s
eth3: 3.0 GB/s
eth4: 3.0 GB/s
Total: 12 GB/s
```

### Actual (current)

```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'

# Actual output:
eth1: 3.0 GB/s
eth2: 0.0 GB/s
eth3: 0.0 GB/s
eth4: 0.0 GB/s
Total: 3.0 GB/s
```

---

## 🎯 Next Steps

1. **Run Test 1** (minimal configuration) and check logs
2. **Run Test 2** (single NIC) to verify TCPX is working
3. **Run Test 3** (two NICs) to see if multi-NIC works at all
4. **Run Test 4** (NCCL_CROSS_NIC=1) to see if this helps
5. **Check TCPX source code** to understand multi-NIC support

---

## 📝 Report Findings

When reporting findings, include:

1. **Test number** (e.g., "Test 2: Single NIC")
2. **Command used**
3. **NIC traffic** (from `ifstat`)
4. **Relevant log lines** (from `grep`)
5. **Observations**

**Example**:
```
Test 2: Single NIC (eth2)

Command:
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2
./tests/test_tcpx_perf server 0

NIC traffic:
eth1: 0.0 GB/s
eth2: 3.0 GB/s  ← Working!
eth3: 0.0 GB/s
eth4: 0.0 GB/s

Logs:
NET_GPUDIRECTTCPX_SOCKET_IFNAME set to eth2

Observation:
TCPX is working and can use eth2. Multi-NIC may not be supported.
```

---

**Good luck debugging!** 🚀

