# AI Assistant Handoff Prompt

**Purpose**: This prompt allows a new AI assistant (without conversation history) to immediately continue debugging the eth2 "rx no cmsg" issue.

---

## Context Injection Prompt

Copy and paste this entire section to a new AI assistant:

```
I'm working on a NIXL-TCPX plugin for GCP A3-high instances (2 nodes, 8x H100 GPUs per node, 4x gVNIC per node). The project uses Google's nccl-plugin-gpudirecttcpx APIs to implement GPU-to-GPU P2P communication over TCPX (GPUDirect over TCP with devmem-tcp kernel API).

CRITICAL ISSUE: The P2P benchmark works perfectly on eth1 but fails on eth2 with "rx no cmsg" error, even though NCCL AllReduce tests successfully use BOTH eth1 and eth2 with GPUDirect TCPX devmem (proven by ethtool showing millions of rx_devmem_pkts on both NICs).

KEY FACTS:
1. Environment is VERIFIED WORKING - NCCL uses eth2 successfully with devmem
2. Hardware topology is correct: eth1/eth2 on NUMA0 for GPU0-3, eth3/eth4 on NUMA1 for GPU4-7
3. Same TCPX plugin version (3.1.6._2023_09_27) used by both NCCL and P2P bench
4. eth1-only P2P test: WORKS (rx_devmem_pkts increases)
5. eth2-only P2P test: FAILS with "rx no cmsg" (rx_devmem_pkts stays 0)
6. eth2 with host-recv mode: WORKS (bypasses devmem/cmsg)
7. Plugin loading path, flow steering config, and buffer types all verified correct

HYPOTHESIS: The issue is in how our P2P benchmark code interacts with the TCPX plugin for eth2, NOT an environmental/system issue. There's a code path difference between eth1 and eth2 in our implementation or plugin interaction.

WORKSPACE: /home/daniel/uccl
KEY FILES:
- Debug report: p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md (READ THIS FIRST)
- P2P benchmark: p2p/tcpx/bench_p2p.sh
- Test program: p2p/tcpx/tests/test_tcpx_perf.cc
- TCPX wrapper: p2p/tcpx/tcpx_impl.cc
- Working NCCL reference: collective/rdma/run_nccl_test_tcpx.sh

IMMEDIATE TASK: Read the debug report, then help me identify why eth2 doesn't trigger the devmem/cmsg path in our P2P code when eth1 does, given that both NICs work perfectly in NCCL tests.

START BY: Reading p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md for full context.
```

---

## Quick Start Commands for New AI

After injecting the context above, the AI can immediately run:

### 1. Read the full debug report
```
view p2p/tcpx/docs/DEBUG_ETH2_RX_NO_CMSG.md
```

### 2. Examine the P2P test code
```
view p2p/tcpx/tests/test_tcpx_perf.cc
view p2p/tcpx/tcpx_impl.cc
```

### 3. Compare with working NCCL script
```
view collective/rdma/run_nccl_test_tcpx.sh
```

### 4. Check recent logs
```
view p2p/tcpx/logs/bench_server_*.log
view p2p/tcpx/logs/bench_client_*.log
```

### 5. Search for relevant code patterns
```
grep-search --query "ncclNetListen|ncclNetAccept|ncclNetRegMr" --directory /home/daniel/uccl/p2p/tcpx
```

---

## Expected AI Workflow

1. **Read DEBUG_ETH2_RX_NO_CMSG.md** - Get full context
2. **Analyze code paths** - Compare eth1 vs eth2 execution through tcpx_impl.cc
3. **Instrument code** - Add debug logging to trace plugin API calls
4. **Run controlled experiments** - eth1 vs eth2 with verbose logging
5. **Identify divergence point** - Find where eth2 path differs from eth1
6. **Propose fix** - Based on root cause analysis

---

## Key Debugging Principles

- **ethtool rx_devmem_pkts is ground truth**: If it's 0, devmem path isn't active
- **NCCL proves environment works**: Don't waste time on system-level debugging
- **Plugin is black box**: Debug by observing API call patterns, not internals
- **Compare working vs broken**: eth1 (works) vs eth2 (fails) in same codebase
- **Verify assumptions**: Check device index mapping, NIC name parsing, etc.

---

## Common Pitfalls to Avoid

❌ **Don't** assume it's a kernel/driver/dp-manager issue (NCCL proves otherwise)  
❌ **Don't** try to modify TCPX plugin (we don't have source, treat as black box)  
❌ **Don't** focus on flow steering implementation (it's external, we just configure it)  
❌ **Don't** ignore ethtool counters (they're authoritative)  

✅ **Do** compare eth1 vs eth2 code paths in our wrapper  
✅ **Do** add verbose logging to trace plugin API calls  
✅ **Do** verify device index and NIC name mapping  
✅ **Do** check initialization order and timing  

---

## Success Criteria

The issue is resolved when:
1. `./bench_p2p.sh server 0 --ifaces=eth2` completes without "rx no cmsg" error
2. `ethtool -S eth2` shows `rx_devmem_pkts` increasing during eth2 P2P test
3. Both eth1 and eth2 work identically in P2P benchmark (matching NCCL behavior)

---

## Additional Context

### Project Background
- **Goal**: Implement NIXL-TCPX plugin using Google's nccl-plugin-gpudirecttcpx APIs
- **Partners**: Anyscale, Character AI
- **Environment**: GCP A3-high (H100), TCPX-only (no RDMA)
- **Constraint**: Treat net_tcpx.h as black box, use send/recv APIs directly

### Technical Details
- **GPUDirect TCPX**: Zero-copy GPU-to-GPU over TCP via devmem-tcp kernel API
- **devmem-tcp**: Kernel provides cmsg with scattered buffer descriptors for DMA
- **Unpack kernel**: CUDA kernel copies scattered devmem buffers to contiguous GPU memory
- **Flow steering**: UNIX socket-based traffic steering via dp-manager (external service)

### Network Topology
- **Node 0**: eth0 (ctrl), eth1/eth2 (NUMA0/GPU0-3), eth3/eth4 (NUMA1/GPU4-7)
- **Node 1**: Same layout
- **IPs**: See scripts/node_ips/tcpx.txt

---

## Reproduction Commands

### Failing Case (eth2)
```bash
# Terminal 1 (Node 0):
cd /mnt/user_storage/uccl/p2p/tcpx
./bench_p2p.sh server 0 --ifaces=eth2 --iters=10 --size=67108864

# Terminal 2 (Node 1):
./bench_p2p.sh client <NODE0_ETH0_IP> 0 --ifaces=eth2 --iters=10 --size=67108864

# Terminal 3 (Either node):
watch -n 0.5 'ethtool -S eth2 | grep rx_devmem_pkts'
# Observe: stays at 0 (WRONG)
```

### Working Case (eth1)
```bash
# Same commands but --ifaces=eth1
# Observe: rx_devmem_pkts increases (CORRECT)
```

### Working NCCL Reference (eth2)
```bash
cd /mnt/user_storage/uccl/collective/rdma
./run_nccl_test_tcpx.sh
# Uses both eth1 and eth2 successfully
# Observe: rx_devmem_pkts increases on BOTH NICs
```

---

## File Structure Reference

```
/home/daniel/uccl/
├── p2p/tcpx/
│   ├── bench_p2p.sh              # Main benchmark script
│   ├── tests/
│   │   └── test_tcpx_perf.cc     # P2P test program
│   ├── tcpx_impl.cc              # TCPX plugin wrapper (KEY FILE)
│   ├── include/
│   │   └── tcpx_interface.h      # API definitions
│   ├── logs/                     # Test logs
│   │   ├── bench_server_*.log
│   │   ├── bench_client_*.log
│   │   └── infomation.log        # ethtool stats, topology
│   └── docs/
│       ├── DEBUG_ETH2_RX_NO_CMSG.md  # Full debug report (READ FIRST)
│       └── AI_HANDOFF_PROMPT.md      # This file
├── collective/rdma/
│   └── run_nccl_test_tcpx.sh     # Working NCCL reference
└── scripts/node_ips/
    └── tcpx.txt                  # Network configuration
```

---

## Environment Variables Reference

### TCPX Configuration (from bench_p2p.sh)
```bash
NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=/run/tcpx
NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;..."
NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;..."
```

### Plugin Loading (from tcpx_impl.cc)
```bash
UCCL_TCPX_PLUGIN_PATH=/usr/local/tcpx/lib64/libnccl-net-tcpx.so  # default
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:...
```

### Debug Logging
```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=ENV,NET,INIT
UCCL_TCPX_DEBUG=1
```

---

## Last Known State

- **Date**: 2025-10-06
- **Status**: Issue reproduced consistently, environment verified, root cause unknown
- **Next Step**: Instrument tcpx_impl.cc to trace plugin API calls for eth1 vs eth2
- **Blocker**: Need to identify why eth2 doesn't trigger devmem registration in P2P code

---

## Questions to Guide Investigation

1. Does `tcpx_get_properties()` return different values for eth1 vs eth2?
2. Is the device index mapping correct when `--ifaces=eth2` is specified?
3. Does `ncclNetListen()` get called with the same parameters for both NICs?
4. Is there a timing difference in flow steering rule creation?
5. Does the plugin have internal state that differs between first NIC (eth1) and second NIC (eth2)?
6. Are there any hardcoded assumptions about NIC order or names in our wrapper code?

---

## Expected Time to Resolution

- **With focused debugging**: 2-4 hours (add logging, compare traces, identify divergence)
- **With code fix**: +1-2 hours (implement fix, test, verify)
- **Total estimate**: 3-6 hours for experienced developer with AI assistance

---

## Success Indicators During Debug

- [ ] Can reproduce issue consistently (DONE)
- [ ] Have verbose logs for both eth1 (working) and eth2 (failing) (TODO)
- [ ] Identified first point of divergence in code path (TODO)
- [ ] Understand why divergence causes devmem to not activate (TODO)
- [ ] Have proposed fix with clear rationale (TODO)
- [ ] Fix tested and verified with ethtool counters (TODO)

---

## Final Notes

This is a **code path issue**, not an environmental issue. The smoking gun is that NCCL uses eth2 successfully with the same plugin, same hardware, same kernel. Our P2P code does something different that prevents eth2's devmem path from activating.

Focus on **comparative analysis** (eth1 vs eth2 in our code) rather than absolute debugging (trying to understand TCPX plugin internals).

Good luck! 🚀

