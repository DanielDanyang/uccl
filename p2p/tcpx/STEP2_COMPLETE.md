# Step 2: Control Plane Refactor - COMPLETE

**Date**: 2025-10-07  
**Status**: ✅ Control plane implemented and compiled

---

## ✅ Completed Tasks

### 1. Single-Process Launcher Script

**File**: `run_p2p_singleproc.sh`

**Key Features**:
- Launches **1 process per node** (vs 8 processes in old version)
- Manages all 8 GPUs in single process
- **All 4 NICs available** to all GPUs: `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4`
- 8 channels per GPU (configurable via `UCCL_TCPX_NUM_CHANNELS`)
- Bootstrap port range: `20000-20007` (one per GPU)

**Usage**:
```bash
# Server
./run_p2p_singleproc.sh server

# Client
./run_p2p_singleproc.sh client <server_ip>
```

### 2. Single-Process Orchestrator

**File**: `tests/test_tcpx_perf_orchestrator.cc`

**Architecture**:
```
Single Process
├── GPU 0 (ChannelManager, 8 channels)
├── GPU 1 (ChannelManager, 8 channels)
├── GPU 2 (ChannelManager, 8 channels)
├── GPU 3 (ChannelManager, 8 channels)
├── GPU 4 (ChannelManager, 8 channels)
├── GPU 5 (ChannelManager, 8 channels)
├── GPU 6 (ChannelManager, 8 channels)
└── GPU 7 (ChannelManager, 8 channels)

Total: 64 channels, all 4 NICs shared
```

**Execution Flow**:

**Server**:
1. Initialize all 8 GPUs (CUDA contexts, buffers)
2. Listen on all GPUs (each GPU's ChannelManager listens)
3. Bootstrap handshake (send handles to client, one connection per GPU)
4. Accept connections (all channels for all GPUs)
5. Register memory (all GPUs, all channels)

**Client**:
1. Initialize all 8 GPUs
2. Bootstrap handshake (receive handles from server)
3. Connect to server (all channels for all GPUs)
4. Register memory (all GPUs, all channels)

### 3. Bootstrap Strategy

**Port Allocation**:
```
GPU 0: port 20000
GPU 1: port 20001
GPU 2: port 20002
GPU 3: port 20003
GPU 4: port 20004
GPU 5: port 20005
GPU 6: port 20006
GPU 7: port 20007
```

**Per-GPU Bootstrap**:
- Each GPU has its own bootstrap connection
- Server sends all channel handles for that GPU in one message
- Client receives all handles and creates ChannelManager

**Sequencing**:
- Sequential execution (no concurrent listen/accept issues)
- All GPUs listen → all GPUs bootstrap → all GPUs accept → all GPUs register

---

## 📊 Comparison: Multi-Process vs Single-Process

| Aspect | Multi-Process (Old) | Single-Process (New) |
|--------|---------------------|----------------------|
| Processes per node | 8 | 1 |
| GPUs per process | 1 | 8 |
| Channels per GPU | 1 | 8 (configurable) |
| NICs per GPU | 1 (fixed mapping) | 4 (all available) |
| Total channels | 8 | 64 |
| Devmem conflicts | ❌ Yes | ✅ No |
| NIC sharing | ❌ No | ✅ Yes |
| Bootstrap ports | 20000-20007 | 20000-20007 |

---

## 🔧 Key Implementation Details

### NIC Configuration

**Old (Multi-Process)**:
```bash
# GPU 0-1 → eth1 only
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1

# GPU 2-3 → eth2 only
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth2

# etc.
```

**New (Single-Process)**:
```bash
# All GPUs → all NICs
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
```

### Memory Allocation

**4KB Alignment** (required by devmem-tcp):
```cpp
CUdeviceptr d_base, d_aligned;
cuMemAlloc(&d_base, size + 4096);
uintptr_t addr = (uintptr_t)d_base;
addr = (addr + 4095) & ~4095;  // 4KB align
void* gpu_buf = (void*)addr;
```

### ChannelManager Usage

**Per-GPU ChannelManager**:
```cpp
for (int gpu_id = 0; gpu_id < 8; gpu_id++) {
    ChannelManager* mgr = new ChannelManager(num_channels, gpu_id);
    
    // Server
    mgr->server_listen_all(handles);
    mgr->server_accept_all();
    mgr->register_memory(gpu_buf, size, NCCL_PTR_CUDA, true);
    
    // Client
    mgr->client_connect_all(handles);
    mgr->register_memory(gpu_buf, size, NCCL_PTR_CUDA, false);
}
```

---

## 📝 Files Created/Modified

### New Files
- `run_p2p_singleproc.sh` (140 lines) - Single-process launcher
- `tests/test_tcpx_perf_orchestrator.cc` (300 lines) - Orchestrator program
- `STEP2_COMPLETE.md` (this file)

### Modified Files
- `Makefile` (+8 lines) - Added test_tcpx_perf_orchestrator target

### Build Status
✅ **Compiled successfully**

```bash
$ make test_tcpx_perf_orchestrator
Building test_tcpx_perf_orchestrator...
/usr/local/cuda/bin/nvcc ... -o tests/test_tcpx_perf_orchestrator ...
```

---

## 🧪 How to Test

### Basic Test (Channel Creation + Memory Registration)

```bash
# Server (Node 0)
cd /home/daniel/uccl/p2p/tcpx
./run_p2p_singleproc.sh server

# Client (Node 1)
./run_p2p_singleproc.sh client <NODE0_IP>
```

**Expected Output**:
```
=== Single-Process P2P Orchestrator ===
Role: server
GPUs: 8
Channels per GPU: 8
Bootstrap port base: 20000
=======================================

[INFO] TCPX devices: 4
[INFO] All GPUs can use all 4 NICs (single-process architecture)

[GPU 0] Initializing...
[GPU 0] Buffer allocated: 0x...
...
[GPU 7] Buffer allocated: 0x...

[INFO] All GPUs initialized

[SERVER] Step 1: Listening on all GPUs...
[GPU 0] Listening on 8 channels
...
[GPU 7] Listening on 8 channels

[SERVER] Step 2: Bootstrap handshake...
[GPU 0] Sent 8 handles
...
[GPU 7] Sent 8 handles

[SERVER] Step 3: Accepting connections...
[GPU 0] Accepted 8 connections
...
[GPU 7] Accepted 8 connections

[SERVER] Step 4: Registering memory...
[GPU 0] Registered memory on 8 channels
...
[GPU 7] Registered memory on 8 channels

=== ALL GPUs READY (SERVER) ===
Total channels: 64
Architecture: Single process, all NICs shared
```

---

## 🎯 Next Steps

### Step 3: Data Plane Upgrade (Not Yet Started)

**Tasks**:
1. Add actual data transfer logic to orchestrator
2. Implement round-robin channel selection
3. Add sliding window flow control
4. Implement unpack kernel integration

**Estimated Time**: 2-3 days

### Step 4: Thread Affinity (Not Yet Started)

**Checkpoint First**:
- Run prototype with only env vars
- Check if TCPX plugin auto-binds threads
- Decide: env vars only OR manual pthread_setaffinity_np()

**Estimated Time**: 0.5-1 day

### Step 5: Instrumentation (Not Yet Started)

**Tasks**:
- Per-NIC traffic verification (ethtool)
- Per-channel traffic counters
- Channel distribution logging

**Estimated Time**: 0.5-1 day

### Step 6: Validation (Not Yet Started)

**Tasks**:
- Bandwidth testing
- CPU usage profiling
- Multi-NIC verification

**Estimated Time**: 1-2 days

---

## 📈 Progress Status

| Step | Status | Time Spent | Remaining |
|------|--------|------------|-----------|
| Step 2.5: Devmem Validation | ✅ Complete | 1 day | - |
| **Step 2: Control Plane** | ✅ **Complete** | **0.5 day** | **-** |
| Step 3: Data Plane | ⏳ Not Started | - | 2-3 days |
| Step 4: Thread Affinity | ⏳ Not Started | - | 0.5-1 day |
| Step 5: Instrumentation | ⏳ Not Started | - | 0.5-1 day |
| Step 6: Validation | ⏳ Not Started | - | 1-2 days |

**Total Progress**: 2/6 steps complete (33%)  
**Estimated Remaining**: 4-7 days work, 6-10 days calendar

---

**Status**: ✅ Step 2 complete, ready to test on GCP  
**Next Action**: Test orchestrator on GCP nodes, then proceed to Step 3

