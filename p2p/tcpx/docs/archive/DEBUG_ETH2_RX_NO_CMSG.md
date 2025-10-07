# Debug Report: eth2 "rx no cmsg" Issue

**Date**: 2025-10-06 (Initial), 2025-10-07 (Final Update)
**Status**: ✅ RESOLVED - Root cause identified and workaround implemented
**Priority**: CLOSED - Issue was due to loopback testing, not eth2-specific

---

## Executive Summary

**FINAL RESOLUTION (2025-10-07)**: The "rx no cmsg" error was NOT specific to eth2. The root cause was running both server and client on the **same machine** (loopback communication), which TCPX GPUDirect devmem does not support. When tests were run on **separate nodes**, all NICs (eth1, eth2, eth3, eth4) work correctly with devmem.

**Original Issue (2025-10-06)**: The TCPX P2P benchmark appeared to fail on eth2 with "rx no cmsg" error while eth1 worked. However, this was a testing methodology error, not a real eth2 issue.

**Key Learning**: TCPX devmem requires actual network communication between different machines. Loopback tests will always fail with "rx no cmsg" regardless of which NIC is used.

---

## Resolution Summary

### What Was Wrong
- Tests were run with both server and client on the **same node** (loopback)
- TCPX devmem path requires packets to traverse the physical network
- Loopback packets don't go through the NIC's devmem path, hence "no cmsg"

### What Fixed It
- Run server on Node 0, client on Node 1 (separate machines)
- All 4 NICs (eth1-4) now work correctly with devmem
- Verified by ethtool showing rx_devmem_pkts increasing on all NICs

### Current Status
- ✅ Single-NIC P2P benchmark works on all NICs (eth1, eth2, eth3, eth4)
- ✅ Performance: ~2.75 GB/s server, ~1.17 GB/s client per GPU
- ❌ Multi-NIC per GPU fails due to devmem resource conflicts (different issue)
- 🔍 Performance gap vs NCCL (18.7 GB/s) under investigation (IRQ binding hypothesis)

---

## Historical Context (Original Investigation)

The following sections document the original investigation before the root cause was discovered. Preserved for reference and learning.

---

## Environment Details

### Hardware Topology (GCP A3-high, 2 nodes)
- **Per node**: 8x H100 GPUs, 4x gVNIC (eth1-4)
- **NUMA mapping** (verified via PCI BDF):
  - eth1 (0000:06:00.0), eth2 (0000:0c:00.0) → NUMA 0 → GPU 0-3
  - eth3 (0000:86:00.0), eth4 (0000:8c:00.0) → NUMA 1 → GPU 4-7
- **Kernel versions**: Node0: 6.6.93+, Node1: 6.6.72+ (minor mismatch, not root cause)
- **gVNIC driver**: 1.3.3 (both nodes)

### Software Stack
- **NCCL Plugin**: NET/GPUDirectTCPX ver. 3.1.6._2023_09_27
- **Plugin path**: `/usr/local/tcpx/lib64/libnccl-net-tcpx.so` (symlinked from `/var/lib/tcpx/lib64`)
- **CUDA**: 12.x
- **TCPX services**: dp-manager running, UNIX sockets present at `/run/tcpx/`

---

## Problem Statement

### Symptom
When running P2P benchmark with eth2 in kernel-unpack mode:
- **Server side**: Fails with `[ERROR] process_recv_cmsg() rx no cmsg` (line 168 in server.log)
- **Client side**: Gets `Connection reset by peer` shortly after
- **ethtool counter**: `rx_devmem_pkts` on eth2 remains **0** during P2P test

### Working Cases
1. **eth1-only P2P test**: Works perfectly, rx_devmem_pkts increases
2. **eth2 with host-recv mode**: Works (bypasses devmem/cmsg path)
3. **NCCL AllReduce with eth1+eth2**: Works perfectly with both NICs showing millions of rx_devmem_pkts

### Evidence Table

| Test Scenario | eth1 rx_devmem_pkts | eth2 rx_devmem_pkts | Result |
|---------------|---------------------|---------------------|--------|
| P2P bench eth1-only | 83,018 | 0 | ✅ PASS |
| P2P bench eth2-only | 0 | 0 | ❌ FAIL (rx no cmsg) |
| P2P bench eth2 host-recv | 0 | 0 | ✅ PASS (no devmem) |
| NCCL AllReduce eth1+eth2 | 31,327,026 | 31,223,838 | ✅ PASS |

**Conclusion**: eth2's devmem/cmsg path is functional at system level, but not triggered in P2P benchmark.

---

## Verification Steps Completed

### ✅ 1. Plugin Version & Loading Path
**Verified**: Both P2P bench and NCCL test load the same plugin version.

```bash
# From logs:
# NCCL: NET/GPUDirectTCPX ver. 3.1.6._2023_09_27
# P2P:  NET/GPUDirectTCPX ver. 3.1.6._2023_09_27
grep -E 'Loading plugin:|NET/GPUDirectTCPX ver\.' logs/bench_*.log
```

**Status**: ✅ Confirmed identical

### ✅ 2. Server Receive Buffer Type
**Verified**: Server is using GPU memory with kernel unpack mode.

```bash
# Expected in server log:
# [PERF] Unpack impl: kernel
# [PERF][SERVER] Registering recv buffer: type=NCCL_PTR_CUDA
grep -E 'Unpack impl:|Registering recv buffer' logs/bench_server_*.log
```

**Status**: ✅ Confirmed CUDA buffer + kernel mode

### ✅ 3. Flow Steering Configuration
**Verified**: UNIX flow steering is enabled and configured.

```bash
# Expected in logs:
# Flow Steering Strategy: unix client
# using unix client prefix '/run/tcpx'
grep -E 'Flow Steering|unix client prefix' logs/bench_*.log
```

**Status**: ✅ Confirmed enabled (though actual rule creation logs may vary)

### ✅ 4. ethtool Counters During Test
**Verified**: rx_devmem_pkts is the authoritative indicator.

```bash
# Run during test:
watch -n 0.5 'ethtool -S eth2 | egrep "rx_devmem_pkts|rx_packets|tx_packets"'
```

**Result**: 
- NCCL test: rx_devmem_pkts increases to millions ✅
- P2P bench: rx_devmem_pkts stays at 0 ❌

---

## Configuration Alignment Attempts

### Attempt 1: Match NCCL Environment Variables
Modified `bench_p2p.sh` to align with `collective/rdma/run_nccl_test_tcpx.sh`:

```bash
# Added to bench_p2p.sh:
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
UNIX_PREFIX="/run/tcpx"  # default enabled
```

**Result**: No change, eth2 still fails with rx no cmsg

### Attempt 2: Force Plugin Path
Tried explicitly setting plugin path to match NCCL:

```bash
export UCCL_TCPX_PLUGIN_PATH="/var/lib/tcpx/lib64/libnccl-net-tcpx.so"
```

**Result**: File not found (plugin actually at `/usr/local/tcpx/lib64/` via symlink)

**Reverted to**: Default path `/usr/local/tcpx/lib64/libnccl-net-tcpx.so`

---

## Current Hypothesis

Since environment is proven functional (NCCL works), the issue likely lies in:

1. **Code path difference**: P2P benchmark may take a different code path through the TCPX plugin compared to NCCL, causing eth2 to not trigger devmem registration
2. **Initialization order**: Possible difference in how NICs are initialized/registered between single-NIC (eth1) and eth2
3. **Hidden plugin state**: The plugin may have internal state/caching that behaves differently for eth2 vs eth1
4. **Timing/race condition**: Flow steering rules may not be fully programmed before data transfer starts on eth2

**Key Observation**: The fact that eth1 works but eth2 doesn't, while NCCL uses both successfully, suggests the issue is in how our P2P test interacts with the plugin, not the plugin itself.

---

## Recommended Next Steps

### High Priority Debugging

1. **Compare plugin API call sequences**
   - Add verbose logging to `p2p/tcpx/tcpx_impl.cc` to trace all plugin API calls
   - Run eth1-only vs eth2-only and diff the call sequences
   - Look for differences in `ncclNetRegMr`, `ncclNetListen`, `ncclNetAccept` parameters

2. **Instrument plugin loading and NIC selection**
   - Add debug prints in `tcpx_get_properties()` and `tcpx_listen()` 
   - Verify device index mapping: does eth2 get correct dev_id?
   - Check if `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2` is properly parsed by plugin

3. **Capture plugin-side logs with maximum verbosity**
   ```bash
   export NCCL_DEBUG=TRACE
   export NCCL_DEBUG_SUBSYS=ALL
   export UCCL_TCPX_DEBUG=1
   # Re-run both eth1 and eth2 tests, compare logs
   ```

4. **Test intermediate configurations**
   - Try eth1+eth2 together in P2P bench (not just NCCL)
   - Try different GPU IDs with eth2 (GPU 1, 2, 3 instead of GPU 0)
   - Try eth3/eth4 to see if issue is specific to eth2 or affects all non-eth1 NICs

### Medium Priority Investigation

5. **Check for NIC-specific plugin quirks**
   - Review TCPX plugin source (if available) for eth1 vs eth2 special handling
   - Check if plugin has hardcoded assumptions about first NIC

6. **Verify flow steering rule creation timing**
   - Add timestamps to all flow steering related logs
   - Confirm rules are created BEFORE first data transfer attempt
   - Check `/run/tcpx/rx_rule_manager` socket interaction timing

7. **Memory registration differences**
   - Compare `ncclNetRegMr` calls for eth1 vs eth2
   - Verify GPU memory pointer and size are identical
   - Check if devmem registration succeeds but isn't used

### Low Priority (Workarounds)

8. **Force single-NIC mode per GPU**
   - Modify code to only use eth1 for GPU 0-3, eth3 for GPU 4-7
   - Skip eth2/eth4 entirely as temporary workaround

9. **Kernel version alignment**
   - Upgrade both nodes to same kernel (6.6.93+)
   - Unlikely to fix but eliminates one variable

---

## Key Files & Locations

### Source Code
- **Main benchmark script**: `p2p/tcpx/bench_p2p.sh`
- **Test program**: `p2p/tcpx/tests/test_tcpx_perf.cc`
- **TCPX wrapper**: `p2p/tcpx/tcpx_impl.cc` (plugin loading & API calls)
- **TCPX interface**: `p2p/tcpx/include/tcpx_interface.h`

### Working NCCL Reference
- **NCCL test script**: `collective/rdma/run_nccl_test_tcpx.sh` (KNOWN WORKING with eth2)

### Logs
- **P2P logs**: `p2p/tcpx/logs/bench_server_*.log`, `bench_client_*.log`
- **Info collection**: `p2p/tcpx/logs/infomation.log` (ethtool stats, PCI topology)
- **Historical logs**: `p2p/tcpx/server.log`, `client.log` (previous runs)

### Configuration
- **Network IPs**: `scripts/node_ips/tcpx.txt`
- **Plugin location**: `/usr/local/tcpx/lib64/libnccl-net-tcpx.so` → `/var/lib/tcpx/lib64/libnccl-net-tcpx.so`
- **UNIX sockets**: `/run/tcpx/` (dp-manager endpoints)

---

## Quick Reproduction Steps

### Terminal 1 (Server - Node 0)
```bash
cd /mnt/user_storage/uccl/p2p/tcpx
./bench_p2p.sh server 0 --ifaces=eth2 --iters=10 --size=67108864
# Watch for "rx no cmsg" error in logs/bench_server_*.log
```

### Terminal 2 (Client - Node 1)
```bash
cd /mnt/user_storage/uccl/p2p/tcpx
./bench_p2p.sh client <NODE0_ETH0_IP> 0 --ifaces=eth2 --iters=10 --size=67108864
# Watch for "Connection reset by peer"
```

### Terminal 3 (Monitor - Either Node)
```bash
# Before and during test:
watch -n 0.5 'ethtool -S eth2 | egrep "rx_devmem_pkts|rx_packets"'
# rx_devmem_pkts should increase if devmem path is active (it doesn't for P2P)
```

### Working Comparison (eth1)
```bash
# Server:
./bench_p2p.sh server 0 --ifaces=eth1 --iters=10 --size=67108864
# Client:
./bench_p2p.sh client <NODE0_ETH0_IP> 0 --ifaces=eth1 --iters=10 --size=67108864
# This works perfectly, rx_devmem_pkts increases on eth1
```

---

## Important Notes for Next Developer

1. **Do NOT assume environment issues**: The environment is proven working via NCCL tests. Focus on code path differences.

2. **ethtool is ground truth**: `rx_devmem_pkts` counter is the authoritative indicator of whether devmem/cmsg path is active. Logs can be misleading.

3. **Plugin is a black box**: We don't have TCPX plugin source. Debug by observing API call patterns and comparing working (eth1) vs broken (eth2) cases.

4. **Flow steering is external**: We configure it via env vars, but actual rule programming is done by plugin + dp-manager. Our code doesn't implement flow steering logic.

5. **Symlink setup required**: The plugin at `/usr/local/tcpx/lib64/` is a symlink to `/var/lib/tcpx/lib64/`. This is intentional due to container path constraints.

6. **Two-node setup required**: This is a P2P test, needs both nodes running simultaneously. Use `scripts/node_ips/tcpx.txt` for IP addresses.

---

## Contact & Handoff

**Original Developer**: Ray  
**Collaborators**: Anyscale, Character AI  
**GCP Support**: Confirmed TCPX unpack kernel copies scattered buffers from devmem-tcp cmsg  

**Last Updated**: 2025-10-06  
**Next Review**: When new developer starts investigation

