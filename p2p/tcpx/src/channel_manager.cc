/**
 * @file channel_manager.cc
 * @brief Implementation of multi-channel TCPX connection manager
 */

#include "channel_manager.h"
#include "sliding_window.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

ChannelManager::ChannelManager(int num_channels, int gpu_id)
    : num_channels_(num_channels), gpu_id_(gpu_id) {

  // Validate against actual TCPX device count
  int tcpx_dev_count = tcpx_get_device_count();
  if (tcpx_dev_count < 0) {
    std::cerr << "[ChannelManager] Failed to get TCPX device count" << std::endl;
    num_channels_ = 0;
    return;
  }

  if (num_channels_ > tcpx_dev_count) {
    std::cerr << "[ChannelManager] Warning: Requested " << num_channels_
              << " channels but only " << tcpx_dev_count
              << " TCPX devices available. Clamping to " << tcpx_dev_count << std::endl;
    num_channels_ = tcpx_dev_count;
  }

  channels_.resize(num_channels_);

  // Initialize each channel
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];

    ch.channel_id = i;

    // Map channel_id to TCPX device index
    // TCPX enumerates devices according to NCCL_GPUDIRECTTCPX_SOCKET_IFNAME
    // Query actual device properties to verify mapping
    ch.net_dev = i;

    // Query device properties to get actual NIC name
    struct tcpx_net_properties props;
    if (tcpx_get_properties(i, &props) == 0 && props.name) {
      std::cout << "[ChannelManager] Channel " << i << " → netDev " << i
                << " (" << props.name << ", " << props.speed << " Mbps)" << std::endl;
    } else {
      std::cout << "[ChannelManager] Channel " << i << " → netDev " << i
                << " (properties unavailable)" << std::endl;
    }

    ch.listen_comm = nullptr;
    ch.recv_comm = nullptr;
    ch.send_comm = nullptr;

    // Initialize device handle pointers to point into storage
    ch.recv_dev_handle = ch.recv_dev_handle_storage.data();
    ch.send_dev_handle = ch.send_dev_handle_storage.data();
    std::memset(ch.recv_dev_handle_storage.data(), 0, ch.recv_dev_handle_storage.size());
    std::memset(ch.send_dev_handle_storage.data(), 0, ch.send_dev_handle_storage.size());

    ch.mhandle = nullptr;
    ch.sliding_window = new SlidingWindow(16);  // Max 16 in-flight per TCPX comm
    ch.bytes_transferred = 0;
    ch.chunks_processed = 0;
  }

  std::cout << "[ChannelManager] Created " << num_channels_
            << " channels for GPU " << gpu_id_
            << " (TCPX devices: " << tcpx_dev_count << ")" << std::endl;
}

ChannelManager::~ChannelManager() {
  for (auto& ch : channels_) {
    if (ch.sliding_window) {
      delete ch.sliding_window;
      ch.sliding_window = nullptr;
    }
  }
}

ChannelResources& ChannelManager::get_channel(int idx) {
  // CRITICAL: Check for empty vector first (e.g., local runs without TCPX)
  if (channels_.empty()) {
    std::cerr << "[ChannelManager] FATAL: No channels available (TCPX not initialized)" << std::endl;
    std::cerr << "[ChannelManager] This usually means TCPX plugin is not loaded or no devices found" << std::endl;
    // Cannot return a reference to non-existent element - this is a programming error
    std::abort();  // Fail fast to make misuse obvious
  }

  if (idx < 0 || idx >= num_channels_) {
    std::cerr << "[ChannelManager] ERROR: Invalid channel index " << idx
              << " (valid range: 0-" << (num_channels_ - 1) << ")" << std::endl;
    std::cerr << "[ChannelManager] Returning channel 0 as fallback" << std::endl;
    return channels_[0];  // Safe fallback since we know channels_ is not empty
  }

  return channels_[idx];
}

ChannelResources& ChannelManager::get_channel_for_chunk(int chunk_idx) {
  // CRITICAL: Check for empty vector first
  if (channels_.empty()) {
    std::cerr << "[ChannelManager] FATAL: No channels available for chunk " << chunk_idx << std::endl;
    std::abort();  // Fail fast
  }

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
                << ch.channel_id << ", netDev=" << ch.net_dev << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": listen_comm=" << ch.listen_comm << std::endl;
  }
  
  std::cout << "[ChannelManager] All " << num_channels_ 
            << " channels listening successfully" << std::endl;
  return 0;
}

int ChannelManager::server_accept_all() {
  constexpr int kMaxRetries = 100;
  constexpr int kRetryDelayMs = 100;
  
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Accepting connection..." << std::endl;
    
    // Retry accept (client may not have connected yet)
    bool accepted = false;
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
        accepted = true;
        break;
      }
      
      // Client hasn't connected yet, retry
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
    }
    
    if (!accepted) {
      std::cerr << "[ChannelManager] Failed to accept connection for channel " 
                << ch.channel_id << " after " << kMaxRetries << " retries" << std::endl;
      return -1;
    }
  }
  
  std::cout << "[ChannelManager] All " << num_channels_ 
            << " channels accepted successfully" << std::endl;
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
    
    // Make a mutable copy of the handle (tcpx_connect_v5 may modify it)
    ncclNetHandle_v7 handle_copy = handles[i];
    
    if (tcpx_connect_v5(ch.net_dev, &handle_copy, &ch.send_comm, &ch.send_dev_handle) != 0 
        || !ch.send_comm) {
      std::cerr << "[ChannelManager] tcpx_connect_v5 failed for channel " 
                << ch.channel_id << ", netDev=" << ch.net_dev << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Connected, send_comm=" << ch.send_comm << std::endl;
  }
  
  std::cout << "[ChannelManager] All " << num_channels_ 
            << " channels connected successfully" << std::endl;
  return 0;
}

int ChannelManager::register_memory(void* buffer, size_t size, int ptr_type, bool is_recv) {
  const char* type_str = (ptr_type == NCCL_PTR_CUDA) ? "CUDA" : "HOST";
  const char* role_str = is_recv ? "recv" : "send";
  
  std::cout << "[ChannelManager] Registering " << type_str << " memory for " 
            << role_str << ": ptr=" << buffer << ", size=" << size << std::endl;
  
  for (int i = 0; i < num_channels_; i++) {
    ChannelResources& ch = channels_[i];
    void* comm = is_recv ? ch.recv_comm : ch.send_comm;
    
    if (!comm) {
      std::cerr << "[ChannelManager] Channel " << ch.channel_id 
                << ": comm is null, cannot register memory" << std::endl;
      return -1;
    }
    
    if (tcpx_reg_mr(comm, buffer, size, ptr_type, &ch.mhandle) != 0) {
      std::cerr << "[ChannelManager] tcpx_reg_mr failed for channel " 
                << ch.channel_id << std::endl;
      return -1;
    }
    
    std::cout << "[ChannelManager] Channel " << ch.channel_id 
              << ": Memory registered, mhandle=" << ch.mhandle << std::endl;
  }
  
  std::cout << "[ChannelManager] All " << num_channels_ 
            << " channels registered memory successfully" << std::endl;
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
  
  std::cout << "[ChannelManager] All channels deregistered memory" << std::endl;
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
  
  std::cout << "[ChannelManager] All channels closed" << std::endl;
}

