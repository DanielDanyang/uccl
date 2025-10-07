# Phase 1 Quick Start Guide

**Goal**: Test round-robin NIC distribution fix with 4 channels/GPU

---

## 🚀 Quick Commands

### On Server Node (Node 0):
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh server
```

### On Client Node (Node 1):
```bash
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh client <SERVER_IP>
```

### Verify Results:
```bash
cd /home/daniel/uccl/p2p/tcpx
./verify_nic_distribution.sh
```

---

## ✅ Success Indicators

Look for these in the output:

### 1. Round-Robin Distribution Message
```
[ChannelManager] GPU 0: Distributing 4 channels across 4 NICs (round-robin)
[ChannelManager] GPU 1: Distributing 4 channels across 4 NICs (round-robin)
...
```

### 2. All 4 NICs Used
```
[ChannelManager] Channel 0 → netDev 0 (eth1, ...)
[ChannelManager] Channel 1 → netDev 1 (eth2, ...)
[ChannelManager] Channel 2 → netDev 2 (eth3, ...)
[ChannelManager] Channel 3 → netDev 3 (eth4, ...)
```

### 3. All GPUs Accept Successfully
```
[GPU 0] Accepted 4 connections
[GPU 1] Accepted 4 connections
...
[GPU 7] Accepted 4 connections
```

### 4. No Errors
```
=== ALL GPUs READY (SERVER) ===
Total channels: 32
Architecture: Single process, all NICs shared
```

---

## ❌ Failure Indicators

### If You See This:
```
[ChannelManager] Failed to accept connection for channel X after 100 retries
[ERROR] GPU Y: server_accept_all failed
```

**Action**: 
1. Check `verify_nic_distribution.sh` output
2. Reduce to 2 channels/GPU: `UCCL_TCPX_NUM_CHANNELS=2 ./run_p2p_singleproc.sh server`
3. Report results

---

## 📊 Expected NIC Distribution (4 channels/GPU)

```
Total: 32 channels (8 GPUs × 4 channels)
Distribution: 8 channels per NIC

eth1 (netDev 0): 8 channels
eth2 (netDev 1): 8 channels
eth3 (netDev 2): 8 channels
eth4 (netDev 3): 8 channels
```

---

## 🔍 Troubleshooting

### Problem: "No such file or directory"
**Solution**: Make sure you're in `/home/daniel/uccl/p2p/tcpx`

### Problem: "Permission denied"
**Solution**: 
```bash
chmod +x test_phase1_4ch.sh
chmod +x verify_nic_distribution.sh
```

### Problem: Client can't connect
**Solution**: 
1. Check server is running first
2. Verify server IP is correct
3. Check firewall: `sudo iptables -L | grep 20000`

---

## 📈 Next Steps After Success

### If 4 channels/GPU works:
```bash
# Test with 8 channels/GPU
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh server
UCCL_TCPX_NUM_CHANNELS=8 ./run_p2p_singleproc.sh client <SERVER_IP>
./verify_nic_distribution.sh
```

Expected: 16 channels per NIC (still within limits)

---

## 📝 Log Files

Logs are saved to: `p2p/tcpx/logs/singleproc_<role>_<timestamp>.log`

View latest server log:
```bash
ls -t logs/singleproc_server_*.log | head -1 | xargs cat
```

View latest client log:
```bash
ls -t logs/singleproc_client_*.log | head -1 | xargs cat
```

---

**Ready to test!** Start with server, then client, then verify.

