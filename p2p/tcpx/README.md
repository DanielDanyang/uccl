# TCPX P2P Performance Testing

GPU-to-GPU point-to-point performance testing using Google's TCPX plugin with GPU Direct TCPX (devmem-tcp).

## Quick Start

### Prerequisites
- Two H100 nodes with TCPX support
- Google nccl-plugin-gpudirecttcpx installed
- Multi-NIC environment (4 NICs recommended)
- Linux kernel with devmem-tcp support

### Build
```bash
cd /home/daniel/uccl/p2p/tcpx
make test_tcpx_perf -j4
```

### Run Single Test (64MB)

**Server (10.64.52.73):**
```bash
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**Client (10.64.113.74):**
```bash
./bench_p2p.sh client 10.64.52.73 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

### Run Full Sweep (4KB → 256MB)

**Server:**
```bash
./bench_p2p_sweep_server.sh 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
```

**Client:**
```bash
./bench_p2p_sweep_client.sh 10.64.52.73 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
```

Results: `logs/p2p_sweep_YYYYMMDD_HHMMSS.md`

---

## Architecture

### Key Components

1. **test_tcpx_perf.cc** - Main performance test
   - Server: Receives data with GPU kernel unpack (sliding window)
   - Client: Sends data (sliding window)
   
2. **unpack_kernels.cu** - GPU kernel for unpacking scattered packets
   - Copies scattered devmem-tcp buffers to contiguous GPU memory
   
3. **bench_p2p.sh** - Single test runner
   - Configures TCPX environment
   - Launches server/client with proper settings

### Sliding Window Design

Both server and client use sliding windows to avoid exhausting TCPX request pools:

**TCPX Request Pool Limit:**
- Each `tcpxComm` has **MAX_REQUESTS = 16** (hardcoded in TCPX plugin)
- Independent of `NCCL_NSOCKS_PERTHREAD` or `NCCL_SOCKET_NTHREADS`

**Server (Receiver):**
```cpp
constexpr int MAX_INFLIGHT = 16;  // Max concurrent recv requests
// Window: irecv → kernel → event → irecv_consumed
```

**Client (Sender):**
```cpp
constexpr int MAX_INFLIGHT_SEND = 12;  // Max concurrent send requests (< 16)
// Window: isend → test (auto-release when done)
```

**Why Sliding Window?**
- 64MB ÷ 512KB = 128 chunks
- Without sliding window: 128 requests needed → exhausts pool at chunk 17
- With sliding window: ≤ 12-16 requests in-flight → stable

---

## Performance Optimizations

### 1. Kernel Launch Optimization (100× speedup)

**Problem:** Original code created/destroyed stream and launcher per chunk
```cpp
// ❌ Wrong: 54ms overhead per chunk
for each chunk:
  cudaStreamCreate(&stream);
  UnpackLauncher launcher(stream);
  launcher.launchSync(desc);  // Synchronous!
  cudaStreamDestroy(stream);
```

**Solution:** Reuse stream and launcher, async launch
```cpp
// ✅ Correct: ~0.01ms per chunk
cudaStreamCreate(&stream);
UnpackLauncher* launcher = new UnpackLauncher(stream);
for each chunk:
  launcher->launch(desc);  // Async
cudaStreamSynchronize(stream);  // Sync once at end
```

**Result:** Kernel mode performance now matches D2D mode (~20 GB/s on 4 NICs)

### 2. Request Pool Management (Sliding Window)

**Problem:** Batch send/recv exhausts TCPX request pool
```cpp
// ❌ Wrong: 128 chunks → 128 requests → pool exhausted
for (128 chunks):
  tcpx_isend(...);  // Allocates request
  tcpx_test(...);   // Returns done=1, but request not immediately freed
```

**Solution:** Sliding window limits in-flight requests
```cpp
// ✅ Correct: ≤ 12 requests in-flight
if (pending_reqs.size() >= MAX_INFLIGHT_SEND) {
  wait_for_oldest();  // Drain one before issuing next
}
tcpx_isend(...);
pending_reqs.push_back(req);
```

**Result:** Stable operation for any message size

---

## Environment Variables

### TCPX Configuration
```bash
# Network interfaces (required)
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4

# Port range (required)
export NCCL_GPUDIRECTTCPX_PORT_BEGIN=50000
export NCCL_GPUDIRECTTCPX_PORT_END=60000

# Socket/thread config (optional, auto-detected for GCP)
# Note: These control parallel sockets, NOT request pool size
export NCCL_NSOCKS_PERTHREAD=1   # GCP default: 1
export NCCL_SOCKET_NTHREADS=6    # GCP default: 6

# Polling optimization (optional)
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=0
```

### Test Options
```bash
./bench_p2p.sh [server|client] [server_ip] [gpu_id] [options]

Options:
  --ifaces=LIST      Comma-separated NIC list (default: eth1,eth2,eth3,eth4)
  --size=BYTES       Total bytes per iteration (default: 67108864 = 64MB)
  --iters=N          Iterations (default: 20)
  --chunk=BYTES      Chunk size (default: 524288 = 512KB)
  --impl=kernel|d2d  Unpack implementation (default: kernel)
  --no-unix          Disable UNIX flow steering (default)
```

---

## Troubleshooting

### "unable to allocate requests"
**Cause:** Request pool exhausted (MAX_REQUESTS = 16 per comm)
**Solution:** Sliding window is already implemented in both server and client

### "rx no cmsg" errors
**Cause:** devmem-tcp not working
**Solution:** 
- Check kernel support: `dmesg | grep devmem`
- Verify NICs support devmem-tcp
- Use correct IP range (10.64.x.x for GCP)

### Low performance (<5 GB/s on 4 NICs)
**Possible causes:**
1. Kernel mode using synchronous launch → Check code uses async `launch()`
2. Single NIC only → Check `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME`
3. Small message size → Expected for <1MB messages

### Test hangs or times out
**Possible causes:**
1. Firewall blocking ports → Check port range 50000-60000
2. Wrong server IP → Verify 10.64.52.73 reachable
3. GPU not accessible → Check `nvidia-smi`

---

## File Structure

```
p2p/tcpx/
├── README.md                    # This file
├── Makefile                     # Build configuration
├── bench_p2p.sh                 # Single test runner
├── bench_p2p_sweep_*.sh         # Sweep test runners
├── tests/
│   ├── test_tcpx_perf.cc        # Main performance test
│   └── test_tcpx_transfer.cc    # Basic transfer test
├── device/
│   ├── unpack_kernels.cu        # GPU unpack kernel
│   └── unpack_launch.cu         # Kernel launcher
├── include/
│   ├── tcpx_interface.h         # TCPX API wrapper
│   └── rx_descriptor.h          # RX descriptor structures
├── logs/                        # Test logs
└── docs/                        # Additional documentation
    ├── PERF_DIARY.md            # Development history
    └── TCPX_LOGIC_MAPPING.md    # TCPX internals
```

---

## Performance Expectations

### 4-NIC Configuration (eth1-4, ~25 Gbps each)
| Message Size | Expected BW | Notes |
|--------------|-------------|-------|
| 4KB - 64KB   | 1-5 GB/s    | Small message overhead |
| 128KB - 1MB  | 5-15 GB/s   | Ramping up |
| 2MB - 64MB   | 15-20 GB/s  | Peak performance |
| 128MB+       | 18-22 GB/s  | Sustained peak |

**Theoretical max:** 4 × 25 Gbps = 100 Gbps = 12.5 GB/s
**Actual:** ~20 GB/s (160 Gbps) due to TCP overhead and multi-flow aggregation

---

## Development History

See `docs/PERF_DIARY.md` for complete development history including:
- Initial implementation and devmem-tcp setup
- Kernel performance debugging (100× speedup)
- Request pool exhaustion and sliding window solution
- TCPX internals analysis

---

## References

- [Google nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx)
- [NCCL Unpack Reference](https://github.com/NVIDIA/nccl/tree/master/src/device/network/unpack)
- Linux devmem-tcp kernel documentation

---

**Last Updated:** 2025-10-02  
**Status:** Production ready with sliding window optimization

