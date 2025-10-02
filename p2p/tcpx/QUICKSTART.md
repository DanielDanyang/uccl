# TCPX P2P Quick Start Guide

Get up and running in 5 minutes.

---

## Prerequisites

- 2 nodes with H100 GPUs
- TCPX plugin installed
- 4 data NICs (eth1-4) + 1 control NIC (eth0)

---

## Step 1: Build

```bash
cd /home/daniel/uccl/p2p/tcpx
make clean && make test_tcpx_perf -j4
```

**Expected**: Compiles successfully with no errors.

---

## Step 2: Run Server

**On Node 1 (10.65.74.150)**:

```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**Expected output**:
```
=== TCPX P2P Benchmark ===
Role        : server
GPU ID      : 0
Ifaces      : eth1,eth2,eth3,eth4
...
[PERF] Iter 0 time=21.5ms
[PERF] Iter 1 time=20.8ms
...
[PERF] Avg: 21.0 ms, BW: 3.0 GB/s
```

---

## Step 3: Run Client

**On Node 2 (10.64.113.77)**:

```bash
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**Expected output**:
```
=== TCPX P2P Benchmark ===
Role        : client
Server IP   : 10.65.74.150
GPU ID      : 0
...
[PERF] Iter 0 time=78.2ms
[PERF] Iter 1 time=76.5ms
...
[PERF] Avg: 77.0 ms, BW: 0.8 GB/s
```

---

## Step 4: Verify Multi-NIC Usage

**On either node, during test**:

```bash
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'
```

**Expected**: All 4 NICs show traffic.

**Current Issue**: Only eth1 shows traffic. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Step 5: Check Results

```bash
# Server performance
grep "Avg:" logs/bench_server_*.log

# Client performance
grep "Avg:" logs/bench_client_*.log
```

**Current Performance**:
- Server: ~21 ms, ~3 GB/s
- Client: ~77 ms, ~1 GB/s

**Target Performance** (with 4 NICs):
- Server: <10 ms, >10 GB/s
- Client: <30 ms, >5 GB/s

---

## Common Options

### Change Transfer Size

```bash
# 128 MB instead of 64 MB
./bench_p2p.sh server 0 --size=134217728
```

### Change Chunk Size

```bash
# 4 MB chunks instead of 2 MB
./bench_p2p.sh server 0 --chunk=4194304
```

### Change Iterations

```bash
# 50 iterations instead of 20
./bench_p2p.sh server 0 --iters=50
```

### Use Different GPU

```bash
# GPU 1 instead of GPU 0
./bench_p2p.sh server 1
```

---

## Troubleshooting

### Build Fails

```bash
# Check CUDA installation
which nvcc
nvcc --version

# Check TCPX plugin
ls -l /usr/local/tcpx/lib64/libnccl-net.so
```

### Connection Timeout

```bash
# Check network connectivity
ping <SERVER_IP>

# Check firewall
sudo iptables -L
```

### Only eth1 Used

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for multi-NIC debugging.

---

## Next Steps

1. **Read**: [HANDOFF.md](HANDOFF.md) for complete project overview
2. **Debug**: Multi-NIC configuration (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md))
3. **Optimize**: Follow performance optimization roadmap in HANDOFF.md

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or [HANDOFF.md](HANDOFF.md).

