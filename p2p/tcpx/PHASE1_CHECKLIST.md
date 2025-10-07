# Phase 1 Testing Checklist

**Date**: 2025-10-07  
**Tester**: ___________  
**Nodes**: Node 0 (Server), Node 1 (Client)

---

## ✅ Pre-Test Verification

- [ ] Code compiled successfully
  ```bash
  cd /home/daniel/uccl/p2p/tcpx
  make test_tcpx_perf_orchestrator
  ```
  Expected: No errors

- [ ] Test scripts are executable
  ```bash
  ls -l test_phase1_4ch.sh verify_nic_distribution.sh
  ```
  Expected: `-rwxr-xr-x` (executable)

- [ ] Both nodes accessible
  ```bash
  # From Node 0
  ping <NODE1_IP>
  
  # From Node 1
  ping <NODE0_IP>
  ```
  Expected: Successful ping

---

## 🧪 Phase 1a: 4 Channels/GPU Test

### Step 1: Start Server (Node 0)
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh server
```

**Check for**:
- [ ] `Distributing 4 channels across 4 NICs (round-robin)` message for each GPU
- [ ] Channels assigned to netDev 0, 1, 2, 3 (all 4 NICs)
- [ ] `[GPU X] Listening on 4 channels` for all 8 GPUs
- [ ] Server waiting for bootstrap connections

**If server fails to start**: ___________________________________________

### Step 2: Start Client (Node 1)
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh client <NODE0_IP>
```

**Check for**:
- [ ] `Distributing 4 channels across 4 NICs (round-robin)` message for each GPU
- [ ] Channels assigned to netDev 0, 1, 2, 3 (all 4 NICs)
- [ ] `[GPU X] Connected 4 channels` for all 8 GPUs
- [ ] No connection errors

**If client fails to connect**: ___________________________________________

### Step 3: Verify Server Completion (Node 0)
**Check server terminal for**:
- [ ] `[GPU 0] Accepted 4 connections`
- [ ] `[GPU 1] Accepted 4 connections`
- [ ] `[GPU 2] Accepted 4 connections`
- [ ] `[GPU 3] Accepted 4 connections`
- [ ] `[GPU 4] Accepted 4 connections` ← **Critical: This was failing before**
- [ ] `[GPU 5] Accepted 4 connections` ← **Critical: This was failing before**
- [ ] `[GPU 6] Accepted 4 connections`
- [ ] `[GPU 7] Accepted 4 connections`
- [ ] `=== ALL GPUs READY (SERVER) ===`
- [ ] `Total channels: 32`

**If any GPU fails to accept**: ___________________________________________

### Step 4: Analyze Logs
```bash
cd /home/daniel/uccl/p2p/tcpx
./verify_nic_distribution.sh
```

**Expected Output**:
- [ ] `netDev 0 (eth1): 8 channels`
- [ ] `netDev 1 (eth2): 8 channels`
- [ ] `netDev 2 (eth3): 8 channels`
- [ ] `netDev 3 (eth4): 8 channels`
- [ ] `✅ GPU 0: Accepted 4 connections` (all 8 GPUs)
- [ ] `✅ No errors detected`

**Actual Output**:
```
netDev 0 (eth1): ___ channels
netDev 1 (eth2): ___ channels
netDev 2 (eth3): ___ channels
netDev 3 (eth4): ___ channels
```

**If distribution is uneven**: ___________________________________________

---

## 🚀 Phase 1b: 8 Channels/GPU Test (ONLY IF 1a PASSES)

### Step 1: Start Server (Node 0)
```bash
cd /home/daniel/uccl/p2p/tcpx
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh server
```

**Check for**:
- [ ] `Distributing 8 channels across 4 NICs (round-robin)` message
- [ ] `[GPU X] Listening on 8 channels` for all 8 GPUs
- [ ] Server waiting for bootstrap connections

### Step 2: Start Client (Node 1)
```bash
cd /home/daniel/uccl/p2p/tcpx
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh client <NODE0_IP>
```

**Check for**:
- [ ] `Distributing 8 channels across 4 NICs (round-robin)` message
- [ ] `[GPU X] Connected 8 channels` for all 8 GPUs
- [ ] No connection errors

### Step 3: Verify Server Completion (Node 0)
**Check server terminal for**:
- [ ] All 8 GPUs accepted 8 connections each
- [ ] `=== ALL GPUs READY (SERVER) ===`
- [ ] `Total channels: 64`

### Step 4: Analyze Logs
```bash
./verify_nic_distribution.sh
```

**Expected Output**:
- [ ] `netDev 0 (eth1): 16 channels`
- [ ] `netDev 1 (eth2): 16 channels`
- [ ] `netDev 2 (eth3): 16 channels`
- [ ] `netDev 3 (eth4): 16 channels`
- [ ] `✅ No errors detected`

**Actual Output**:
```
netDev 0 (eth1): ___ channels
netDev 1 (eth2): ___ channels
netDev 2 (eth3): ___ channels
netDev 3 (eth4): ___ channels
```

---

## 📊 Results Summary

### Phase 1a (4 channels/GPU):
- **Status**: [ ] PASS  [ ] FAIL
- **All GPUs accepted**: [ ] YES  [ ] NO
- **All NICs used**: [ ] YES  [ ] NO
- **Errors**: [ ] NONE  [ ] SOME (describe below)

**Notes**: ___________________________________________________________

### Phase 1b (8 channels/GPU):
- **Status**: [ ] PASS  [ ] FAIL  [ ] NOT TESTED
- **All GPUs accepted**: [ ] YES  [ ] NO
- **All NICs used**: [ ] YES  [ ] NO
- **Errors**: [ ] NONE  [ ] SOME (describe below)

**Notes**: ___________________________________________________________

---

## 🐛 Issues Encountered

### Issue 1:
**Description**: ___________________________________________________________

**Log Evidence**: ___________________________________________________________

**Resolution**: ___________________________________________________________

### Issue 2:
**Description**: ___________________________________________________________

**Log Evidence**: ___________________________________________________________

**Resolution**: ___________________________________________________________

---

## 📈 Performance Observations

### Bandwidth (if measured):
- **Phase 1a (4 ch/GPU)**: _______ GB/s
- **Phase 1b (8 ch/GPU)**: _______ GB/s
- **Baseline (old)**: 2.75 GB/s

### CPU Usage:
- **Observation**: ___________________________________________________________

### NIC Traffic (ethtool):
```bash
# Before test
sudo ethtool -S eth1 | grep rx_devmem_pkts
sudo ethtool -S eth2 | grep rx_devmem_pkts
sudo ethtool -S eth3 | grep rx_devmem_pkts
sudo ethtool -S eth4 | grep rx_devmem_pkts

# After test (compare deltas)
```

**Results**: ___________________________________________________________

---

## ✅ Sign-Off

### Phase 1a Test:
- **Tester**: ___________
- **Date**: ___________
- **Result**: [ ] PASS  [ ] FAIL
- **Signature**: ___________

### Phase 1b Test:
- **Tester**: ___________
- **Date**: ___________
- **Result**: [ ] PASS  [ ] FAIL
- **Signature**: ___________

---

## 🎯 Next Actions

### If Phase 1 PASSES:
- [ ] Proceed to Step 3 (Data Plane implementation)
- [ ] Consider Phase 2 (NUMA-aware selection) for optimization
- [ ] Measure bandwidth with actual data transfer

### If Phase 1 FAILS:
- [ ] Analyze failure logs
- [ ] Test with 2 channels/GPU
- [ ] Contact Google about TCPX plugin limits
- [ ] Consider alternative approaches

---

**Checklist Complete**: [ ] YES  [ ] NO  
**Ready for Next Step**: [ ] YES  [ ] NO

