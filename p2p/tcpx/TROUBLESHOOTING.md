# TCPX P2P Troubleshooting Guide

Common issues and solutions.

---

## 🔥 CRITICAL: Multi-NIC Not Working

### Symptom

Only eth1 shows traffic. eth2-4 are idle.

```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
# eth1: 3 GB/s, eth2-4: 0 GB/s
```

### Impact

Bandwidth limited to ~3 GB/s (should be ~12 GB/s with 4 NICs).

### Diagnosis

1. **Check environment variables**:
   ```bash
   ./diagnose_multi_nic.sh
   ```

2. **Enable TCPX debug logs** (already in `bench_p2p.sh`):
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=NET
   ```

3. **Run test and check logs**:
   ```bash
   ./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
   grep "SOCKET_IFNAME" logs/bench_server_*.log
   ```

4. **Look for**:
   ```
   NET_GPUDIRECTTCPX_SOCKET_IFNAME set to eth1,eth2,eth3,eth4
   ```

### Possible Causes

1. **TCPX plugin not reading env var**
   - Check plugin version
   - Check if env var is exported before plugin loads

2. **NCCL_CROSS_NIC=0 preventing multi-NIC**
   - Try setting `NCCL_CROSS_NIC=1` in `bench_p2p.sh`

3. **CPU bindings incorrect**
   - Check `NCCL_GPUDIRECTTCPX_TX_BINDINGS` and `NCCL_GPUDIRECTTCPX_RX_BINDINGS`

4. **TCPX plugin compiled with wrong adapter**
   - Check if plugin uses NCCL adapter or native adapter

### Solutions to Try

#### Solution 1: Verify Environment Variables

```bash
# Before running test
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# Run test
./tests/test_tcpx_perf server 0

# Check logs
grep "SOCKET_IFNAME" server.log
```

#### Solution 2: Try NCCL_CROSS_NIC=1

Edit `bench_p2p.sh`:
```bash
export NCCL_CROSS_NIC=1  # Was 0
```

#### Solution 3: Check TCPX Plugin

```bash
# Check plugin exists
ls -l /usr/local/tcpx/lib64/libnccl-net.so

# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | grep tcpx

# Check if plugin is loaded
ldd ./tests/test_tcpx_perf | grep tcpx
```

#### Solution 4: Simplify Configuration

Try with minimal configuration:
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
# Remove all other TCPX env vars

./tests/test_tcpx_perf server 0
```

---

## ⚠️ Build Errors

### Error: nvcc not found

```bash
# Check CUDA installation
which nvcc
nvcc --version

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
```

### Error: libnccl-net.so not found

```bash
# Check TCPX plugin
ls -l /usr/local/tcpx/lib64/libnccl-net.so

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/tcpx/lib64:$LD_LIBRARY_PATH
```

### Error: undefined reference to tcpx_*

```bash
# Check tcpx_impl.cc is being compiled
make clean && make test_tcpx_perf -j4

# Check Makefile includes tcpx_impl.cc
grep tcpx_impl Makefile
```

---

## ⚠️ Runtime Errors

### Error: Connection timeout

**Symptom**:
```
[ERROR] tcpx_connect failed: timeout
```

**Solutions**:

1. **Check network connectivity**:
   ```bash
   ping <SERVER_IP>
   ```

2. **Check firewall**:
   ```bash
   sudo iptables -L
   # Allow all traffic between nodes
   ```

3. **Increase timeout**:
   ```bash
   export NCCL_GPUDIRECTTCPX_PROGRAM_CONNECT_TIMEOUT_MS=60000  # 60 seconds
   ```

### Error: Unable to allocate requests

**Symptom**:
```
[ERROR] tcpx_irecv failed: rc=3 (unable to allocate requests)
```

**Cause**: Sliding window bug (should be fixed).

**Solution**: Make sure you're using the latest code with sliding window fix.

**Verify fix**:
```bash
grep -A 10 "Before tcpx_irecv, check if sliding window is full" tests/test_tcpx_perf.cc
```

Should show sliding window check BEFORE `tcpx_irecv()`.

### Error: CUDA out of memory

**Symptom**:
```
[ERROR] cudaMalloc failed: out of memory
```

**Solutions**:

1. **Reduce transfer size**:
   ```bash
   ./bench_p2p.sh server 0 --size=33554432  # 32 MB instead of 64 MB
   ```

2. **Reduce chunk size**:
   ```bash
   ./bench_p2p.sh server 0 --chunk=1048576  # 1 MB instead of 2 MB
   ```

3. **Check GPU memory**:
   ```bash
   nvidia-smi
   ```

---

## ⚠️ Performance Issues

### Server much faster than Client

**Symptom**: Server 21ms, Client 77ms (3.7× difference).

**Cause**: Client has smaller sliding window (12 vs 16).

**Solution**: Increase client sliding window.

**Steps**:
1. Edit `tests/test_tcpx_perf.cc` line ~993
2. Change `MAX_INFLIGHT_SEND = 12` to `MAX_INFLIGHT_SEND = 16`
3. Recompile: `make clean && make test_tcpx_perf -j4`

### Performance much slower than iperf3

**Symptom**: TCPX 3 GB/s, iperf3 7.5 GB/s.

**Causes**:

1. **Multi-NIC not working** (see above)
2. **Chunk size too small** (should be 2 MB)
3. **Debug logs enabled** (should be removed)

**Solutions**:

1. Fix multi-NIC (see above)
2. Increase chunk size:
   ```bash
   ./bench_p2p.sh server 0 --chunk=4194304  # 4 MB
   ```
3. Make sure debug logs are removed (check `tests/test_tcpx_perf.cc`)

### Kernel mode slower than D2D mode

**Symptom**: `--impl=kernel` is 100× slower than `--impl=d2d`.

**Cause**: CUDA stream/launcher created inside loop (should be fixed).

**Solution**: Make sure you're using the latest code.

**Verify fix**:
```bash
grep -B 5 "Create CUDA stream and launcher ONCE" tests/test_tcpx_perf.cc
```

Should show stream/launcher creation OUTSIDE the loop.

---

## 🔍 Debugging Tools

### Enable TCPX Debug Logs

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
```

### Monitor NIC Traffic

```bash
# Install ifstat
sudo apt-get install ifstat

# Monitor
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

### Profile with nsys

```bash
nsys profile -o tcpx_profile ./tests/test_tcpx_perf server 0
nsys-ui tcpx_profile.nsys-rep
```

### Check TCPX Configuration

```bash
./diagnose_multi_nic.sh
```

### Analyze Logs

```bash
# Performance summary
grep "Avg:" logs/bench_server_*.log

# Errors
grep "ERROR" logs/bench_server_*.log

# TCPX configuration
grep "NET_GPUDIRECTTCPX" logs/bench_server_*.log

# Chunk processing
grep "chunk_idx=" logs/bench_server_*.log | tail -10
```

---

## 📞 Getting Help

### Check Documentation

1. [QUICKSTART.md](QUICKSTART.md) - Quick start guide
2. [HANDOFF.md](HANDOFF.md) - Complete project overview
3. [docs/TEST_TCPX_PERF_EXPLAINED.md](docs/TEST_TCPX_PERF_EXPLAINED.md) - Detailed code explanation

### Check Logs

```bash
ls -lt logs/*.log | head -5
```

### Run Diagnostics

```bash
./diagnose_multi_nic.sh
```

---

## 📝 Reporting Issues

When reporting issues, include:

1. **Command used**:
   ```bash
   ./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
   ```

2. **Error message**:
   ```
   [ERROR] tcpx_irecv failed: rc=3
   ```

3. **Environment**:
   ```bash
   nvcc --version
   ls -l /usr/local/tcpx/lib64/libnccl-net.so
   echo $LD_LIBRARY_PATH
   ```

4. **Logs**:
   ```bash
   tail -50 logs/bench_server_*.log
   ```

5. **NIC traffic**:
   ```bash
   ifstat -i eth1,eth2,eth3,eth4 1 5
   ```

---

**Still stuck?** Check [HANDOFF.md](HANDOFF.md) for more details.

