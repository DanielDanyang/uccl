# TCPX P2P Quick Reference

## One-Line Summary
GPU-to-GPU P2P performance testing using TCPX with sliding window optimization.

---

## Quick Commands

### Build
```bash
make test_tcpx_perf -j4
```

### Single Test (64MB)
```bash
# Server (10.64.52.73)
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix

# Client (10.64.113.74)
./bench_p2p.sh client 10.64.52.73 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

### Full Sweep (4KB → 256MB)
```bash
# Server
./bench_p2p_sweep_server.sh 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix

# Client
./bench_p2p_sweep_client.sh 10.64.52.73 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
```

---

## Key Facts

| Item | Value | Source |
|------|-------|--------|
| Request pool size | **16 per comm** | `work_queue.h:20` |
| Max in-flight (server) | 16 | `test_tcpx_perf.cc` |
| Max in-flight (client) | 12 | `test_tcpx_perf.cc` |
| Default chunk size | 512KB | `bench_p2p.sh` |
| GCP sockets | 6 (6 threads × 1) | `connect.cc` |

---

## Critical Rules

### ✅ DO
- Use sliding window for both send and recv
- Reuse CUDA streams and launchers
- Use async kernel launch (`launch()`, not `launchSync()`)
- Keep in-flight requests < 16

### ❌ DON'T
- Create/destroy streams in hot path (~4ms overhead)
- Use synchronous kernel launch (~48ms overhead)
- Batch all chunks without sliding window (exhausts pool)
- Assume NSOCKS/NTHREADS affects request pool size (it doesn't)

---

## Performance Targets (4-NIC)

| Size | Bandwidth | Notes |
|------|-----------|-------|
| 4KB | ~1 GB/s | Overhead dominated |
| 1MB | ~15 GB/s | Ramping up |
| 64MB | ~20 GB/s | Peak performance |

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "unable to allocate requests" | Pool exhausted | Sliding window (already implemented) |
| "rx no cmsg" | devmem-tcp not working | Check kernel, use 10.64.x.x IPs |
| Kernel 100× slower | Sync launch in loop | Use async launch (already fixed) |
| Test hangs | Firewall/wrong IP | Check ports 50000-60000, verify IPs |

---

## Environment Variables (Minimal)

```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_PORT_BEGIN=50000
export NCCL_GPUDIRECTTCPX_PORT_END=60000
```

---

## File Map

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `TECHNICAL_NOTES.md` | Detailed technical info |
| `QUICKREF.md` | This file |
| `CHANGELOG.md` | Change history |
| `tests/test_tcpx_perf.cc` | Main test program |
| `bench_p2p.sh` | Test runner |

---

## Next Steps After Fresh Clone

1. Read `README.md` (5 min)
2. Build: `make test_tcpx_perf -j4`
3. Run single test (see commands above)
4. Check logs in `logs/bench_*.log`
5. If issues, see `TECHNICAL_NOTES.md`

---

**Last Updated:** 2025-10-02

