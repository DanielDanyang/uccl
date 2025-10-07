# Step 3 Quick Start Guide

**Goal**: Test data transfer and measure bandwidth

---

## 🚀 Quick Commands

### On Server Node (Node 0):
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh server
```

### On Client Node (Node 1):
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_step3_bandwidth.sh client <SERVER_IP>
```

### Analyze Results:
```bash
cd /home/daniel/uccl/p2p/tcpx
./analyze_bandwidth.sh
```

---

## ✅ Success Indicators

### 1. Data Transfer Completes
```
[CLIENT] ===== Iteration 0 =====
[CLIENT] Iteration 0 completed in XXX ms, bandwidth: X.XX GB/s
...
[CLIENT] ===== Iteration 19 =====
[CLIENT] Iteration 19 completed in XXX ms, bandwidth: X.XX GB/s
```

### 2. Performance Summary
```
[CLIENT] ===== Performance Summary =====
[CLIENT] Average time: XXX ms
[CLIENT] Average bandwidth: X.XX GB/s  ← Should be >5 GB/s
[CLIENT] Total channels used: 32

[SERVER] ===== Performance Summary =====
[SERVER] Average time: XXX ms
[SERVER] Average bandwidth: X.XX GB/s  ← Should be >5 GB/s
[SERVER] Total channels used: 32
```

### 3. No Errors
```
✅ No errors detected
```

---

## 📊 Expected Results

### With 4 Channels/GPU (32 total):
```
Test size: 67108864 bytes (64 MB)
Iterations: 20
Chunk size: 524288 bytes (512 KB)
Total channels: 32

Average bandwidth: >5 GB/s  ← Target
```

### With 8 Channels/GPU (64 total):
```
Test size: 67108864 bytes (64 MB)
Iterations: 20
Chunk size: 524288 bytes (512 KB)
Total channels: 64

Average bandwidth: >10 GB/s  ← Target
```

---

## 🔧 Configuration Options

### Change Test Size:
```bash
# Test with 128MB
UCCL_TCPX_PERF_SIZE=$((128 * 1024 * 1024)) ./test_step3_bandwidth.sh server
```

### Change Iterations:
```bash
# Run 50 iterations for more stable results
UCCL_TCPX_PERF_ITERS=50 ./test_step3_bandwidth.sh server
```

### Change Channels:
```bash
# Test with 8 channels/GPU
UCCL_TCPX_NUM_CHANNELS=8 ./test_step3_bandwidth.sh server
```

### Combine Options:
```bash
# 8 channels, 128MB, 50 iterations
UCCL_TCPX_NUM_CHANNELS=8 \
UCCL_TCPX_PERF_SIZE=$((128 * 1024 * 1024)) \
UCCL_TCPX_PERF_ITERS=50 \
./test_step3_bandwidth.sh server
```

---

## ❌ Failure Indicators

### If You See This:
```
[ERROR] tcpx_isend failed (GPU X channel Y chunk Z)
[ERROR] tcpx_irecv failed (GPU X channel Y chunk Z)
```

**Action**:
1. Check logs: `cat logs/singleproc_*.log | grep ERROR`
2. Verify Phase 1 still works: `./test_phase1_4ch.sh server`
3. Try smaller test size: `UCCL_TCPX_PERF_SIZE=$((16 * 1024 * 1024)) ./test_step3_bandwidth.sh server`

---

## 📈 Performance Targets

### Baseline (Multi-Process)
- **Bandwidth**: 2.75 GB/s
- **Channels**: 8 (1 per GPU)

### Step 3 Targets
| Channels/GPU | Total Channels | Target BW | Status |
|--------------|----------------|-----------|--------|
| 4 | 32 | >5 GB/s | 🔄 To Test |
| 8 | 64 | >10 GB/s | ⏳ After 4ch |

### NCCL Reference
- **Bandwidth**: 19.176 GB/s
- **Goal**: Get within 20% (>15 GB/s)

---

## 🔍 Troubleshooting

### Problem: Bandwidth is low (<3 GB/s)
**Solution**:
1. Check NIC distribution: `./verify_nic_distribution.sh`
2. Check for errors: `grep ERROR logs/singleproc_*.log`
3. Try more channels: `UCCL_TCPX_NUM_CHANNELS=8 ./test_step3_bandwidth.sh server`

### Problem: "Connection refused"
**Solution**:
1. Ensure server started first
2. Wait 5 seconds after server starts
3. Check server IP is correct

### Problem: Test hangs
**Solution**:
1. Check both nodes are responsive
2. Kill and restart: `pkill -9 test_tcpx_perf_orchestrator`
3. Check firewall: `sudo iptables -L | grep 20000`

---

## 📝 Log Files

Logs are saved to: `p2p/tcpx/logs/singleproc_<role>_<timestamp>.log`

View latest logs:
```bash
# Server
tail -f logs/singleproc_server_*.log | grep -E "Iteration|bandwidth|ERROR"

# Client
tail -f logs/singleproc_client_*.log | grep -E "Iteration|bandwidth|ERROR"
```

---

## 🎯 Next Steps After Success

### If Bandwidth >5 GB/s:
1. ✅ Step 3 successful!
2. 🔄 Test with 8 channels/GPU
3. 📊 Compare with NCCL baseline
4. 🚀 Proceed to Step 4 (thread affinity) or Step 5 (instrumentation)

### If Bandwidth <5 GB/s but >2.75 GB/s:
1. ⚠️  Partial success
2. 🔍 Analyze bottlenecks
3. 🔄 Try more channels
4. 📊 Check CPU/NIC utilization

### If Bandwidth <2.75 GB/s:
1. ❌ Something wrong
2. 🔍 Check for errors in logs
3. 🔄 Verify Phase 1 still works
4. 📞 Debug or ask for help

---

**Ready to test!** Start with server, then client, then analyze.

