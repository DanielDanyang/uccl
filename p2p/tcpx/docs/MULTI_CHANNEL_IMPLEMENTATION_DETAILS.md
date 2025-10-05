# Multi-Channel Implementation Details

**Date**: 2025-10-05  
**Companion to**: MULTI_CHANNEL_DESIGN.md

This document provides detailed implementation specifications for each module.

---

## Table of Contents

1. [ChannelManager Module](#channelmanager-module)
2. [Bootstrap Module](#bootstrap-module)
3. [SlidingWindow Module](#slidingwindow-module)
4. [Data Flow](#data-flow)
5. [Error Handling](#error-handling)
6. [Memory Layout](#memory-layout)

---

## ChannelManager Module

### Header: `include/channel_manager.h`

```cpp
#pragma once

#include <array>
#include <string>
#include <vector>
#include <cuda.h>
#include "tcpx_interface.h"

// Forward declarations
struct ncclNetHandle_v7;
class SlidingWindow;

// Per-channel resources
struct ChannelResources {
  int channel_id;                    // Logical channel index (0..N-1)
  int net_dev;                       // TCPX device index resolved from iface list

  // Connection handles
  void* listen_comm;                 // Server-side only
  void* recv_comm;                   // Server-side only
  void* send_comm;                   // Client-side only

  // TCPX device handle storage (required 16-byte alignment)
  alignas(16) std::array<uint8_t, 512> recv_dev_handle_storage;
  alignas(16) std::array<uint8_t, 512> send_dev_handle_storage;
  void* recv_dev_handle;             // Points into recv_dev_handle_storage
  void* send_dev_handle;             // Points into send_dev_handle_storage

  // Memory registration
  void* mhandle;

  // Sliding window helper (one per comm to mirror NCCL's inflight tracking)
  SlidingWindow* sliding_window;

  // Statistics / debugging
  uint64_t bytes_transferred;
  int chunks_processed;
};

class ChannelManager {
public:
  // Constructor
  // @param num_channels: Number of channels to create
  // @param gpu_id: GPU device ID (for CUDA context)
  ChannelManager(int num_channels, int gpu_id);
  
  // Destructor
  ~ChannelManager();
  
  // Get number of channels
  int get_num_channels() const { return num_channels_; }
  
  // Get channel by index
  ChannelResources& get_channel(int idx);
  
  // Get channel for a chunk (round-robin)
  ChannelResources& get_channel_for_chunk(int chunk_idx);
  
  // ========================================================================
  // Server-side methods
  // ========================================================================
  
  // Create listen comms for all channels
  // @param handles: Output vector of handles (one per channel)
  // @return 0 on success, -1 on error
  int server_listen_all(std::vector<ncclNetHandle_v7>& handles);
  
  // Accept connections for all channels
  // @return 0 on success, -1 on error
  int server_accept_all();
  
  // ========================================================================
  // Client-side methods
  // ========================================================================
  
  // Connect to all channels
  // @param handles: Input vector of handles (one per channel)
  // @return 0 on success, -1 on error
  int client_connect_all(const std::vector<ncclNetHandle_v7>& handles);
  
  // ========================================================================
  // Memory management
  // ========================================================================
  
  // Register memory for all channels (shared buffer)
  // @param buffer: GPU or host buffer pointer
  // @param size: Buffer size in bytes
  // @param ptr_type: NCCL_PTR_CUDA or NCCL_PTR_HOST
  // @param is_recv: true for recv (server), false for send (client)
  // @return 0 on success, -1 on error
  int register_memory(void* buffer, size_t size, int ptr_type, bool is_recv);
  
  // Deregister memory for all channels
  // @param is_recv: true for recv (server), false for send (client)
  // @return 0 on success, -1 on error
  int deregister_memory(bool is_recv);
  
  // ========================================================================
  // Cleanup
  // ========================================================================
  
  // Close all connections
  // @param is_recv: true for recv (server), false for send (client)
  void close_all(bool is_recv);
  
private:
  int num_channels_;
  int gpu_id_;
  std::vector<ChannelResources> channels_;
  static std::vector<int> resolve_net_devices(int num_channels);
  
  // Disable copy
  ChannelManager(const ChannelManager&) = delete;
  ChannelManager& operator=(const ChannelManager&) = delete;
};
```

### Implementation: `src/channel_manager.cc`

```cpp
#include "channel_manager.h"
#include "sliding_window.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

// NCCL handle structure
struct ncclNetHandle_v7 {
  char data[128];
};

ChannelManager::ChannelManager(int num_channels, int gpu_id)
    : num_channels_(num_channels), gpu_id_(gpu_id) {
  channels_.resize(num_channels_);

  // Resolve TCPX device order from NCCL_GPUDIRECTTCPX_SOCKET_IFNAME (matches NCCL logic around
  // thirdparty/nccl/src/transport/net.cc:566 where channelId -> netDev).
  std::vector<int> net_dev_order = resolve_net_devices(num_channels_);

  for (int i = 0; i < num_channels_; ++i) {
    channels_[i].channel_id = i;
    channels_[i].net_dev = net_dev_order[i];
    channels_[i].listen_comm = nullptr;
    channels_[i].recv_comm = nullptr;
    channels_[i].send_comm = nullptr;
    channels_[i].recv_dev_handle = channels_[i].recv_dev_handle_storage.data();
    channels_[i].send_dev_handle = channels_[i].send_dev_handle_storage.data();
    channels_[i].mhandle = nullptr;
    channels_[i].sliding_window = new SlidingWindow(16);  // Mirror TCPX MAX_REQUESTS
    channels_[i].bytes_transferred = 0;
    channels_[i].chunks_processed = 0;
  }

  std::cout << "[ChannelManager] Created " << num_channels_
            << " channels for GPU " << gpu_id_ << std::endl;
}

// Optional: after construction, iterate over `channels_` and call
// `tcpx_get_properties(ch.net_dev, &props)` so we can log
// `[ChannelManager] channel 0 -> eth1 (pci=...)`, matching NCCL's verbose output.

ChannelManager::~ChannelManager() {
  for (auto& ch : channels_) {
    if (ch.sliding_window) {
      delete ch.sliding_window;
      ch.sliding_window = nullptr;
    }
  }
}

ChannelResources& ChannelManager::get_channel(int idx) {
  if (idx < 0 || idx >= num_channels_) {
    std::cerr << "[ChannelManager] Invalid channel index: " << idx << std::endl;
    return channels_[0];  // Fallback
  }
  return channels_[idx];
}

// Helper: build channel->netDev mapping identical to NCCL's ncclTopoGetNetDev logic.
std::vector<int> ChannelManager::resolve_net_devices(int num_channels) {
  std::vector<int> devices;

  // 1. Read iface list from NCCL_GPUDIRECTTCPX_SOCKET_IFNAME (same env NCCL uses).
  std::string iface_list = get_env("NCCL_GPUDIRECTTCPX_SOCKET_IFNAME", "eth1,eth2,eth3,eth4");
  std::vector<std::string> ifaces = split_list(iface_list);

  // 2. Query TCPX plugin for enumerated devices; they are ordered exactly as the list was
  //    parsed inside the plugin (see nccl-plugin-gpudirecttcpx/src/connect.cc around line 905).
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    throw std::runtime_error("tcpx_get_device_count returned 0; cannot build mapping");
  }

  // 3. Map channel i -> plugin device that matches iface[i % ifaces.size()].
  for (int i = 0; i < num_channels; ++i) {
    const std::string& target_iface = ifaces[i % ifaces.size()];
    int dev = lookup_tcpx_device_index(target_iface, device_count);
    devices.push_back(dev);
  }

  return devices;
}

// Helper utilities (to implement alongside the manager):
// - std::string get_env(const char* name, const std::string& def);
// - std::vector<std::string> split_list(const std::string& csv);
// - int lookup_tcpx_device_index(const std::string& iface, int device_count);
// The lookup helper will call tcpx_get_properties(dev, &props) and match props.name with iface,
// mirroring NCCL's ncclTopoGetNetDev behaviour.

ChannelResources& ChannelManager::get_channel_for_chunk(int chunk_idx) {
  int channel_idx = chunk_idx % num_channels_;
  return channels_[channel_idx];
}

int ChannelManager::server_listen_all(std::vector<ncclNetHandle_v7>& handles) {
  handles.resize(num_channels_);
  
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Listening on netDev=" << ch.net_dev << std::endl;
    
    if (tcpx_listen(ch.net_dev, &handles[i], &ch.listen_comm) != 0) {
      std::cerr << "[ChannelManager] tcpx_listen failed for channel " 
                << ch.channel_id << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": listen_comm=" << ch.listen_comm << std::endl;
  }
  
  return 0;
}

int ChannelManager::server_accept_all() {
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    std::cout << "[ChannelManager] Channel " << ch.channel_id
              << ": Accepting connection..." << std::endl;

    ch.recv_dev_handle_storage.fill(0);
    ch.recv_dev_handle = ch.recv_dev_handle_storage.data();

    // Retry accept (client may not have connected yet)
    constexpr int kMaxRetries = 100;
    for (int attempt = 0; attempt < kMaxRetries; attempt++) {
      int rc = tcpx_accept_v5(ch.listen_comm, &ch.recv_comm, &ch.recv_dev_handle);
      if (rc != 0) {
        std::cerr << "[ChannelManager] tcpx_accept_v5 failed for channel " 
                  << ch.channel_id << ", rc=" << rc << std::endl;
        return -1;
      }
      
      if (ch.recv_comm) {
        std::cout << "[ChannelManager] Channel " << ch.channel_id 
                  << ": Connection accepted, recv_comm=" << ch.recv_comm << std::endl;
        break;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (!ch.recv_comm) {
      std::cerr << "[ChannelManager] Failed to accept connection for channel " 
                << ch.channel_id << " after retries" << std::endl;
      return -1;
    }
  }
  
  return 0;
}

int ChannelManager::client_connect_all(const std::vector<ncclNetHandle_v7>& handles) {
  if (handles.size() != static_cast<size_t>(num_channels_)) {
    std::cerr << "[ChannelManager] Handle count mismatch: expected " 
              << num_channels_ << ", got " << handles.size() << std::endl;
    return -1;
  }
  
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Connecting to netDev=" << ch.net_dev << std::endl;
    
    ch.send_dev_handle_storage.fill(0);
    ch.send_dev_handle = ch.send_dev_handle_storage.data();

    // Make a mutable copy of the handle
    ncclNetHandle_v7 handle_copy = handles[i];
    
    if (tcpx_connect_v5(ch.net_dev, &handle_copy, &ch.send_comm, &ch.send_dev_handle) != 0 
        || !ch.send_comm) {
      std::cerr << "[ChannelManager] tcpx_connect_v5 failed for channel " 
                << ch.channel_id << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Connected, send_comm=" << ch.send_comm << std::endl;
  }
  
  return 0;
}

int ChannelManager::register_memory(void* buffer, size_t size, int ptr_type, bool is_recv) {
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    void* comm = is_recv ? ch.recv_comm : ch.send_comm;
    
    if (!comm) {
      std::cerr << "[ChannelManager] Channel " << ch.channel_id 
                << ": comm is null, cannot register memory" << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Registering memory, ptr=" << buffer 
              << ", size=" << size << std::endl;
    
    if (tcpx_reg_mr(comm, buffer, size, ptr_type, &ch.mhandle) != 0) {
      std::cerr << "[ChannelManager] tcpx_reg_mr failed for channel " 
                << ch.channel_id << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Memory registered, mhandle=" << ch.mhandle << std::endl;
  }
  
  return 0;
}

int ChannelManager::deregister_memory(bool is_recv) {
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    void* comm = is_recv ? ch.recv_comm : ch.send_comm;
    
    if (comm && ch.mhandle) {
      tcpx_dereg_mr(comm, ch.mhandle);
      ch.mhandle = nullptr;
    }
  }
  
  return 0;
}

void ChannelManager::close_all(bool is_recv) {
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    
    if (is_recv) {
      if (ch.recv_comm) {
        tcpx_close_recv(ch.recv_comm);
        ch.recv_comm = nullptr;
      }
      if (ch.listen_comm) {
        tcpx_close_listen(ch.listen_comm);
        ch.listen_comm = nullptr;
      }
    } else {
      if (ch.send_comm) {
        tcpx_close_send(ch.send_comm);
        ch.send_comm = nullptr;
      }
    }
  }
}
```

---

## Bootstrap Module

### Header: `include/bootstrap.h`

```cpp
#pragma once

#include <vector>

// Forward declaration
struct ncclNetHandle_v7;

// Bootstrap protocol constants
constexpr int kBootstrapPort = 12347;
constexpr size_t kHandleBytes = 128;

// ============================================================================
// Server-side functions
// ============================================================================

// Create bootstrap server and wait for client connection
// @param port: TCP port to listen on
// @param client_fd: Output parameter for connected client socket
// @return 0 on success, -1 on error
int bootstrap_server_create(int port, int* client_fd);

// Send multiple handles to client. Protocol mirrors NCCL's handshake:
//   1) uint32_t channel count
//   2) count×128-byte handle payload
// @param client_fd: Connected client socket
// @param handles: Vector of handles to send
// @return 0 on success, -1 on error
int bootstrap_server_send_handles(int client_fd,
                                   const std::vector<ncclNetHandle_v7>& handles);

// ============================================================================
// Client-side functions
// ============================================================================

// Connect to bootstrap server
// @param server_ip: Server IP address
// @param port: TCP port to connect to
// @param server_fd: Output parameter for connected server socket
// @return 0 on success, -1 on error
int bootstrap_client_connect(const char* server_ip, int port, int* server_fd);

// Receive multiple handles from server (consumes the same format as above).
// @param server_fd: Connected server socket
// @param handles: Output vector of handles (resized to negotiated count)
// @return 0 on success, -1 on error
int bootstrap_client_recv_handles(int server_fd,
                                   std::vector<ncclNetHandle_v7>& handles);
```

### Implementation: `src/bootstrap.cc`

```cpp
int bootstrap_server_send_handles(int client_fd,
                                   const std::vector<ncclNetHandle_v7>& handles) {
  uint32_t count = handles.size();
  if (send_all(client_fd, &count, sizeof(count)) != sizeof(count)) return -1;
  if (count == 0) return 0;
  return send_all(client_fd, handles.data(), count * sizeof(ncclNetHandle_v7)) ==
         static_cast<ssize_t>(count * sizeof(ncclNetHandle_v7)) ? 0 : -1;
}

int bootstrap_client_recv_handles(int server_fd,
                                   std::vector<ncclNetHandle_v7>& handles) {
  uint32_t count = 0;
  if (recv_all(server_fd, &count, sizeof(count)) != sizeof(count)) return -1;
  handles.resize(count);
  if (count == 0) return 0;
  return recv_all(server_fd, handles.data(), count * sizeof(ncclNetHandle_v7)) ==
         static_cast<ssize_t>(count * sizeof(ncclNetHandle_v7)) ? 0 : -1;
}
```

`send_all`/`recv_all` are thin wrappers over `send`/`recv` that loop until the
requested byte count is transferred (identical to the single-handle helper we
already use today).

---

## SlidingWindow Module

### Header: `include/sliding_window.h`

```cpp
#pragma once

#include <vector>
#include <cuda_runtime.h>

class SlidingWindow {
public:
  // Constructor
  // @param max_inflight: Maximum number of in-flight requests (typically 16)
  explicit SlidingWindow(int max_inflight);
  
  // Destructor
  ~SlidingWindow();
  
  // Check if window is full
  bool is_full() const;
  
  // Get current window size
  int size() const;
  
  // Add a new request to the window
  // @param request: TCPX request handle
  // @param chunk_idx: Chunk index for tracking
  // @param event: CUDA event (optional, for server recv)
  void add_request(void* request, int chunk_idx, cudaEvent_t event = nullptr);
  
  // Wait for oldest request and remove it
  // @param comm: TCPX comm handle
  // @param is_recv: true for recv (server), false for send (client)
  // @return 0 on success, -1 on error
  int wait_and_release_oldest(void* comm, bool is_recv);
  
  // Drain all pending requests
  // @param comm: TCPX comm handle
  // @param is_recv: true for recv (server), false for send (client)
  // @return 0 on success, -1 on error
  int drain_all(void* comm, bool is_recv);
  
  // Clear all requests (without waiting)
  void clear();
  
private:
  int max_inflight_;
  std::vector<void*> pending_reqs_;
  std::vector<int> pending_indices_;
  std::vector<cudaEvent_t> events_;
};
```

---

## Data Flow

### Server Receive Flow (Multi-Channel)

```
┌─────────────────────────────────────────────────────────────┐
│ For each chunk (0 to total_chunks-1):                       │
│                                                              │
│  1. Select channel: ch = chunk_idx % num_channels           │
│                                                              │
│  2. Check sliding window:                                   │
│     if (ch.sliding_window->is_full()) {                     │
│       ch.sliding_window->wait_and_release_oldest(           │
│         ch.recv_comm, true);                                │
│     }                                                        │
│                                                              │
│  3. Issue irecv:                                            │
│     tcpx_irecv(ch.recv_comm, 1, recv_data, recv_sizes,      │
│                recv_tags, &ch.mhandle, &request);           │
│                                                              │
│  4. Poll for completion:                                    │
│     while (!done) {                                         │
│       tcpx_test(request, &done, &size);                     │
│     }                                                        │
│                                                              │
│  5. Launch unpack kernel:                                   │
│     cudaEvent_t event;                                      │
│     cudaEventCreate(&event);                                │
│     launch_unpack_kernel(...);                              │
│     cudaEventRecord(event, stream);                         │
│                                                              │
│  6. Add to sliding window:                                  │
│     ch.sliding_window->add_request(request, chunk_idx,      │
│                                    event);                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

> **Note**: The same `ch.net_dev` assignment is reused on the client so both
> sides pick identical NICs for each channel (no negotiation magic needed).

> **Note**: `ch.net_dev` is the TCPX device index resolved up-front via
> `resolve_net_devices`, matching NCCL’s channel→netDev mapping logic so that
> channel 0 consistently binds to `eth1`, channel 1 to `eth2`, and so on.

### Client Send Flow (Multi-Channel)

```
┌─────────────────────────────────────────────────────────────┐
│ For each chunk (0 to total_chunks-1):                       │
│                                                              │
│  1. Select channel: ch = chunk_idx % num_channels           │
│                                                              │
│  2. Check sliding window:                                   │
│     if (ch.sliding_window->is_full()) {                     │
│       ch.sliding_window->wait_and_release_oldest(           │
│         ch.send_comm, false);                               │
│     }                                                        │
│                                                              │
│  3. Issue isend:                                            │
│     tcpx_isend(ch.send_comm, send_data, chunk_size,         │
│                tag, ch.mhandle, &request);                  │
│                                                              │
│  4. Add to sliding window:                                  │
│     ch.sliding_window->add_request(request, chunk_idx);     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### Strategy: Fail Fast

Per the requirements, we use a simple error handling strategy:
- **Any channel failure → entire test fails**
- No partial degradation or retry logic
- Clear error messages indicating which channel failed

### Error Scenarios

1. **tcpx_listen fails for a channel**
   ```cpp
   if (tcpx_listen(ch.net_dev, &handle, &ch.listen_comm) != 0) {
     std::cerr << "[ERROR] Channel " << ch.channel_id 
               << ": tcpx_listen failed for netDev=" << ch.net_dev << std::endl;
     return -1;  // Fail entire test
   }
   ```

2. **tcpx_accept fails for a channel**
   ```cpp
   if (!ch.recv_comm) {
     std::cerr << "[ERROR] Channel " << ch.channel_id 
               << ": Failed to accept connection" << std::endl;
     return -1;  // Fail entire test
   }
   ```

3. **Memory registration fails**
   ```cpp
   if (tcpx_reg_mr(...) != 0) {
     std::cerr << "[ERROR] Channel " << ch.channel_id 
               << ": Memory registration failed" << std::endl;
     return -1;  // Fail entire test
   }
   ```

---

## Memory Layout

### Shared Buffer Approach (Option A - Selected)

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory (64 MB)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  All channels share the same buffer                         │
│  Each chunk writes to: buffer + (chunk_idx * chunk_size)    │
│                                                              │
│  Channel 0 (eth1): Chunks 0, 4, 8, 12, ...                  │
│  Channel 1 (eth2): Chunks 1, 5, 9, 13, ...                  │
│  Channel 2 (eth3): Chunks 2, 6, 10, 14, ...                 │
│  Channel 3 (eth4): Chunks 3, 7, 11, 15, ...                 │
│                                                              │
│  All channels register the same buffer with tcpx_reg_mr()   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Advantages**:
- Simple memory management
- No data copying between channel buffers
- Efficient memory usage

**Implementation**:
```cpp
// Allocate shared buffer
CUdeviceptr d_buffer;
cuMemAlloc(&d_buffer, total_size);

// Register with all channels
for (int i = 0; i < num_channels; i++) {
  tcpx_reg_mr(channels[i].recv_comm, (void*)d_buffer, 
              total_size, NCCL_PTR_CUDA, &channels[i].mhandle);
}

// Each chunk writes to its offset
for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
  size_t offset = chunk_idx * chunk_size;
  void* chunk_ptr = (void*)(d_buffer + offset);
  
  // Use chunk_ptr for this chunk's data
}
```

---

**Next**: See `MULTI_CHANNEL_DESIGN.md` for high-level architecture and implementation plan.
