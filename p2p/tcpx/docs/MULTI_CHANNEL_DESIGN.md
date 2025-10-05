# Multi-Channel TCPX Design Document

**Date**: 2025-10-05  
**Status**: 🚧 Design Phase  
**Goal**: Enable multi-NIC utilization by creating multiple TCPX channels

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Solution Design](#solution-design)
4. [Code Organization](#code-organization)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)

---

## Problem Statement

### Current Behavior

- **Only eth1 has traffic**, eth2-4 are idle
- **Bandwidth limited to ~3 GB/s** (should be ~12 GB/s with 4 NICs)
- **Single TCPX connection** created per test run

### Evidence

```bash
# Network traffic monitoring
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'

# Current:
eth1: 3.0 GB/s  ← Only this one is used
eth2: 0.0 GB/s
eth3: 0.0 GB/s
eth4: 0.0 GB/s
```

---

## Root Cause Analysis

### Key Findings

1. **Single Connection Created**
   - `tests/test_tcpx_perf.cc:239/921` only call `tcpx_listen/tcpx_connect_v5` **once**
   - Uses `gpu_id` as the `dev` argument (typically 0)
   - Result: Always opens a single TCPX comm that maps to the first NIC (eth1)

2. **Wrong dev Parameter Mapping**
   - TCPX plugin enumerates **NICs, not GPUs** (`nccl-plugin-gpudirecttcpx/src/net_tcpx.cc:698`)
   - `kTcpxNetIfs` is filled from `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4`
   - `dev=0` → `kTcpxSocketDevs[0]` → eth1
   - `dev=1` → `kTcpxSocketDevs[1]` → eth2
   - Passing GPU IDs to `tcpx_*` is the **wrong mapping**

3. **Bootstrap Protocol Limitation**
   - Currently sends exactly **one 128-byte handle** (`tests/test_tcpx_perf.cc:258-271 / 900-909`)
   - Cannot establish multiple comms even if we requested them

4. **Single-Comm Data Path**
   - Sliding-window logic (`tests/test_tcpx_perf.cc:420-474`)
   - Memory registration (`tests/test_tcpx_perf.cc:365-418, 965-1016`)
   - All built around a **single** `recv_comm/send_comm`
   - Data path cannot fan out to additional NICs

### NCCL's Approach

From `thirdparty/nccl/src/transport/net.cc`:

```cpp
// NCCL creates multiple channels, each with its own netDev
static ncclResult_t sendSetup(..., int channelId, ...) {
  // Get netDev based on channelId (not GPU ID!)
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, 
                               peerInfo->rank, &netId, &req.netDev, &proxyRank));
  
  // Each channel calls listen/connect with different netDev
  NCCLCHECK(proxyState->ncclNet->listen(req.netDev, respBuff, &resources->netListenComm));
}
```

**Key insight**: NCCL uses `channelId` to determine `netDev`, creating multiple independent connections.

---

## Solution Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Test Program                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Channel 0   │  │  Channel 1   │  │  Channel 2   │  ...     │
│  │  (eth1)      │  │  (eth2)      │  │  (eth3)      │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ listen_comm  │  │ listen_comm  │  │ listen_comm  │          │
│  │ recv_comm    │  │ recv_comm    │  │ recv_comm    │          │
│  │ mhandle      │  │ mhandle      │  │ mhandle      │          │
│  │ sliding_win  │  │ sliding_win  │  │ sliding_win  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
    ┌────────┐        ┌────────┐        ┌────────┐
    │  eth1  │        │  eth2  │        │  eth3  │
    └────────┘        └────────┘        └────────┘
```

### Core Concepts

1. **Channel**: An independent TCPX connection using a specific NIC
   - Each channel has its own `listen_comm`, `recv_comm`, `send_comm`
   - Each channel maps to a specific `netDev` (NIC index) resolved from
     `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME`
   - Channels operate independently with their own sliding windows

2. **Round-Robin Data Distribution**
   ```
   Chunk 0  → Channel 0 (eth1)
   Chunk 1  → Channel 1 (eth2)
   Chunk 2  → Channel 2 (eth3)
   Chunk 3  → Channel 3 (eth4)
   Chunk 4  → Channel 0 (eth1)  ← Wrap around
   ...
   ```

3. **Shared GPU Memory**
   - All channels share the same GPU buffer
   - Each channel registers the same memory region
   - Chunks write to different offsets in the shared buffer

### NCCL Reference Points

- `thirdparty/nccl/src/transport/net.cc:566-642` shows how NCCL derives
  `netDev` per channel (`req->netDev`) using topology helpers and then calls
  `ncclNet->listen` / `ncclNet->connect` with that device. Our plan mirrors
  this by resolving TCPX devices from `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME` and
  binding each channel to a deterministic device index.
- `net.cc:626` highlights the one-listen-per-channel pattern; we will follow
  the same handshake by exporting multiple `ncclNetHandle_v7` structs through
  the bootstrap socket, prefixed with a channel count (matching NCCL’s
  multi-channel negotiation).
- `net.cc:688+` keeps inflight bookkeeping per connection. Our per-channel
  sliding window manager aims to emulate the same limits (16 inflight) so that
  TCPX request pressure matches the NCCL plugin expectations.

### Data Structures

```cpp
// Per-channel resources (similar to NCCL's sendNetResources/recvNetResources)
struct ChannelResources {
  // Channel identification
  int channel_id;           // 0, 1, 2, 3, ...
  int net_dev;              // NIC index (0→eth1, 1→eth2, ...)
  
  // TCPX connection handles
  void* listen_comm;        // Server: listen comm
  void* recv_comm;          // Server: receive comm
  void* send_comm;          // Client: send comm
  void* recv_dev_handle;    // Server: device handle
  void* send_dev_handle;    // Client: device handle
  
  // Memory registration
  void* mhandle;            // Memory handle for this channel
  
  // Sliding window (per-channel, max 16 in-flight)
  std::vector<void*> pending_reqs;
  std::vector<int> pending_chunk_indices;
  std::vector<cudaEvent_t> events;  // Server only
  
  // Statistics
  uint64_t bytes_transferred;
  int chunks_processed;
};

// Global channel manager
struct ChannelManager {
  int num_channels;
  std::vector<ChannelResources> channels;
  
  // Shared GPU memory
  CUdeviceptr d_buffer;
  size_t buffer_size;
  
  // Bootstrap connection
  int bootstrap_fd;
};
```

### Bootstrap Handshake (NCCL-style)

```
uint32_t channel_count;
ncclNetHandle_v7 handles[channel_count];

// Server
channel_count = num_channels;
write(client_fd, &channel_count, sizeof(channel_count));
write(client_fd, handles, channel_count * sizeof(ncclNetHandle_v7));

// Client
read(server_fd, &channel_count, sizeof(channel_count));
handles.resize(channel_count);
read(server_fd, handles.data(), channel_count * sizeof(ncclNetHandle_v7));
```

This mirrors the NCCL handshake between proxy threads (`net.cc:626`) and avoids
per-handle TCP round-trips.

---

## Code Organization

### Problem with Current Structure

`tests/test_tcpx_perf.cc` is **1139 lines** and contains:
- Bootstrap logic
- Connection setup
- Memory management
- Sliding window logic
- Data transfer
- Performance measurement
- Cleanup

**This is too monolithic for multi-channel support.**

### Proposed New Structure

```
p2p/tcpx/
├── include/
│   ├── tcpx_interface.h          # Existing: TCPX plugin wrapper
│   ├── tcpx_structs.h            # Existing: TCPX structures
│   ├── rx_descriptor.h           # Existing: RX descriptor builder
│   ├── channel_manager.h         # NEW: Multi-channel management
│   └── bootstrap.h               # NEW: Bootstrap protocol
│
├── src/
│   ├── tcpx_impl.cc              # Existing: TCPX plugin implementation
│   ├── channel_manager.cc        # NEW: Channel lifecycle management
│   ├── bootstrap.cc              # NEW: Bootstrap handshake
│   └── sliding_window.cc         # NEW: Sliding window logic
│
├── tests/
│   ├── test_tcpx_perf.cc         # REFACTOR: Main test (simplified)
│   ├── test_tcpx_transfer.cc     # Existing: Basic transfer test
│   └── test_multi_channel.cc     # NEW: Multi-channel specific test
│
└── device/
    ├── unpack_kernels.cu         # Existing: GPU unpack kernels
    ├── unpack_launch.cu          # Existing: Kernel launcher
    └── unpack_launch.h           # Existing: Kernel launcher header
```

> **Adoption note**: We can stage the refactor by first embedding the helper
> classes inside `tests/test_tcpx_perf.cc` (anonymous namespace) and extracting
> them into dedicated files once the NCCL-style multi-channel path is stable.

### Module Responsibilities

#### 1. `channel_manager.h/cc` (NEW)

**Purpose**: Manage lifecycle of multiple TCPX channels

```cpp
class ChannelManager {
public:
  // Initialize with number of channels
  ChannelManager(int num_channels, int gpu_id);
  
  // Server: Create listen comms for all channels
  int server_listen_all(std::vector<ncclNetHandle_v7>& handles);
  
  // Server: Accept connections for all channels
  int server_accept_all();
  
  // Client: Connect to all channels
  int client_connect_all(const std::vector<ncclNetHandle_v7>& handles);
  
  // Memory registration (shared buffer across all channels)
  int register_memory(void* buffer, size_t size, int ptr_type);
  
  // Get channel for a given chunk index (round-robin)
  ChannelResources& get_channel_for_chunk(int chunk_idx);
  
  // Cleanup
  ~ChannelManager();
  
private:
  std::vector<ChannelResources> channels_;
  int num_channels_;
  int gpu_id_;
};
```

#### 2. `bootstrap.h/cc` (NEW)

**Purpose**: Handle bootstrap handshake for multiple channels

```cpp
// Server: Create bootstrap server and wait for client
int bootstrap_server_create(int port, int* server_fd);

// Server: Send multiple handles to client
int bootstrap_server_send_handles(int client_fd, 
                                   const std::vector<ncclNetHandle_v7>& handles);

// Client: Connect to bootstrap server
int bootstrap_client_connect(const char* server_ip, int port, int* client_fd);

// Client: Receive multiple handles from server
int bootstrap_client_recv_handles(int server_fd, 
                                   std::vector<ncclNetHandle_v7>& handles);
```

#### 3. `sliding_window.cc` (NEW)

**Purpose**: Per-channel sliding window management

```cpp
class SlidingWindow {
public:
  SlidingWindow(int max_inflight);
  
  // Check if window is full
  bool is_full() const;
  
  // Add a new request to the window
  void add_request(void* request, int chunk_idx, cudaEvent_t event = nullptr);
  
  // Wait for oldest request and remove it
  int wait_and_release_oldest(void* comm, bool is_recv);
  
  // Drain all pending requests
  int drain_all(void* comm, bool is_recv);
  
private:
  int max_inflight_;
  std::vector<void*> pending_reqs_;
  std::vector<int> pending_indices_;
  std::vector<cudaEvent_t> events_;
};
```

#### 4. `test_tcpx_perf.cc` (REFACTORED)

**Purpose**: Main test program (simplified with new modules)

```cpp
int main(int argc, char** argv) {
  // Parse arguments
  bool is_server = ...;
  int gpu_id = ...;
  int num_channels = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 4);
  
  // Create channel manager
  ChannelManager channel_mgr(num_channels, gpu_id);
  
  if (is_server) {
    // Server flow
    std::vector<ncclNetHandle_v7> handles;
    channel_mgr.server_listen_all(handles);
    
    int bootstrap_fd = bootstrap_server_create(kBootstrapPort);
    bootstrap_server_send_handles(bootstrap_fd, handles);
    
    channel_mgr.server_accept_all();
    channel_mgr.register_memory(recv_buf, size, NCCL_PTR_CUDA);
    
    run_server_benchmark(channel_mgr, ...);
    
  } else {
    // Client flow
    int bootstrap_fd = bootstrap_client_connect(server_ip, kBootstrapPort);
    
    std::vector<ncclNetHandle_v7> handles;
    bootstrap_client_recv_handles(bootstrap_fd, handles);
    
    channel_mgr.client_connect_all(handles);
    channel_mgr.register_memory(send_buf, size, NCCL_PTR_CUDA);
    
    run_client_benchmark(channel_mgr, ...);
  }
}
```

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1)

**Goal**: Create supporting helpers without breaking existing tests

**Tasks**:
1. ⬜ Create `include/channel_manager.h`
2. ⬜ Create `src/channel_manager.cc`
3. ⬜ Create `include/bootstrap.h`
4. ⬜ Create `src/bootstrap.cc`
5. ⬜ Create `include/sliding_window.h`
6. ⬜ Create `src/sliding_window.cc`
7. ⬜ Update `Makefile` to compile new modules
8. ⬜ Write unit tests for each module

**Deliverables**:
- New helpers compile successfully
- Unit tests pass
- Existing `test_tcpx_perf` still works (unchanged)

### Phase 2: Multi-Channel Support (Week 2)

**Goal**: Implement multi-channel logic in ChannelManager using NCCL-style mapping

**Tasks**:
1. ⬜ Implement `ChannelManager::server_listen_all()`
2. ⬜ Implement `ChannelManager::server_accept_all()`
3. ⬜ Implement `ChannelManager::client_connect_all()`
4. ⬜ Implement `ChannelManager::register_memory()` (shared buffer)
5. ⬜ Implement `ChannelManager::get_channel_for_chunk()` (round-robin)
6. ⬜ Implement bootstrap protocol for multiple handles (count + handles)
7. ⬜ Write integration tests
8. ⬜ Log channel→NIC mapping using `tcpx_get_properties` for visibility

**Deliverables**:
- Can establish 4 TCPX connections
- Each connection uses different NIC
- Bootstrap protocol works for N handles

### Phase 3: Refactor test_tcpx_perf (Week 3)

**Goal**: Integrate multi-channel support into main test with minimal churn

**Tasks**:
1. ⬜ Refactor server benchmark loop to use ChannelManager
2. ⬜ Refactor client benchmark loop to use ChannelManager
3. ⬜ Update sliding window logic to be per-channel
4. ⬜ Add `UCCL_TCPX_NUM_CHANNELS` environment variable (default = #netDevs)
5. ⬜ Ensure backward compatibility (num_channels=1)
6. ⬜ Update documentation

**Deliverables**:
- `test_tcpx_perf` uses multi-channel by default
- Can run with 1, 2, 4, or 8 channels
- Performance scales linearly with channel count

### Phase 4: Testing & Optimization (Week 4)

**Goal**: Verify multi-NIC utilization and optimize performance

**Tasks**:
1. ⬜ Run with `ifstat` to verify all NICs have traffic
2. ⬜ Benchmark with 1, 2, 4 channels
3. ⬜ Verify 4× bandwidth improvement
4. ⬜ Profile with `nsys` to find bottlenecks
5. ⬜ Optimize chunk distribution strategy
6. ⬜ Add error handling for channel failures
7. ⬜ Update all documentation

**Deliverables**:
- All 4 NICs show traffic
- Bandwidth: 12 GB/s (4× improvement)
- Comprehensive test coverage
- Updated documentation

---

## Testing Strategy

### Unit Tests

```bash
# Test bootstrap protocol
./tests/test_bootstrap

# Test channel manager
./tests/test_channel_manager

# Test sliding window
./tests/test_sliding_window
```

### Integration Tests

```bash
# Test with 1 channel (backward compatibility)
UCCL_TCPX_NUM_CHANNELS=1 ./bench_p2p.sh server 0

# Test with 2 channels
UCCL_TCPX_NUM_CHANNELS=2 ./bench_p2p.sh server 0

# Test with 4 channels (default)
UCCL_TCPX_NUM_CHANNELS=4 ./bench_p2p.sh server 0
```

### Performance Validation

```bash
# Monitor NIC traffic during test
watch -n 1 'ifstat -i eth1,eth2,eth3,eth4'

# Expected with 4 channels:
# eth1: 3.0 GB/s
# eth2: 3.0 GB/s
# eth3: 3.0 GB/s
# eth4: 3.0 GB/s
# Total: 12 GB/s
```

### Error Scenarios

1. **Fewer NICs than channels**: Gracefully reduce channel count
2. **Channel connection failure**: Fail entire test (simple approach)
3. **Channel data transfer failure**: Fail entire test
4. **Mismatched channel count**: Server/client negotiation

---

## Configuration

### Environment Variables

```bash
# Number of channels (default: 4)
export UCCL_TCPX_NUM_CHANNELS=4

# Existing variables (unchanged)
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
export UCCL_TCPX_PERF_SIZE=67108864
export UCCL_TCPX_CHUNK_BYTES=2097152
```

> Align `UCCL_TCPX_NUM_CHANNELS` with NCCL’s `NCCL_MAX_NCHANNELS`/`NCCL_MIN_NCHANNELS`
> used in `collective/rdma/run_nccl_test_tcpx.sh` (currently 8) when we want to
> stress-test parity with the NCCL plugin.

### Auto-Detection

```cpp
// Auto-detect available NICs
int num_nics = tcpx_get_device_count();
int num_channels = getEnvInt("UCCL_TCPX_NUM_CHANNELS", num_nics);

if (num_channels > num_nics) {
  std::cerr << "Warning: UCCL_TCPX_NUM_CHANNELS (" << num_channels 
            << ") > available NICs (" << num_nics << ")" << std::endl;
  num_channels = num_nics;
}
```

---

## Expected Results

### Performance Metrics

| Metric | Before (1 channel) | After (4 channels) | Improvement |
|--------|-------------------|-------------------|-------------|
| Server Bandwidth | 3 GB/s | 12 GB/s | **4×** |
| Client Bandwidth | 1 GB/s | 4 GB/s | **4×** |
| Server Latency | 21 ms | ~20 ms | ~1× |
| Client Latency | 77 ms | ~75 ms | ~1× |
| eth1 Traffic | 3 GB/s | 3 GB/s | 1× |
| eth2 Traffic | 0 GB/s | 3 GB/s | **∞** |
| eth3 Traffic | 0 GB/s | 3 GB/s | **∞** |
| eth4 Traffic | 0 GB/s | 3 GB/s | **∞** |

### Success Criteria

- ✅ All 4 NICs show traffic in `ifstat`
- ✅ Total bandwidth ≥ 10 GB/s (target: 12 GB/s)
- ✅ Backward compatible with `UCCL_TCPX_NUM_CHANNELS=1`
- ✅ No performance regression in single-channel mode
- ✅ Code is modular and maintainable

---

## Next Steps

1. **Review this design document** with the team
2. **Get approval** on the architecture and code organization
3. **Start Phase 1** implementation
4. **Iterate** based on testing results

---

**Questions? Feedback?** Please discuss before implementation begins.
