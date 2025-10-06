/**
 * @file channel_manager.cc
 * @brief Implementation of multi-channel TCPX connection manager
 */

#include "channel_manager.h"
#include "sliding_window.h"
#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>
#include <chrono>

namespace {

std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

bool read_gpu_pci_bdf(int gpu_id, std::string& bdf_out) {
  std::ifstream file("/proc/driver/nvidia/gpus/" + std::to_string(gpu_id) + "/information");
  if (!file.is_open()) return false;
  std::string line;
  while (std::getline(file, line)) {
    auto pos = line.find("Bus Location");
    if (pos == std::string::npos) continue;
    auto colon = line.find(':', pos);
    if (colon == std::string::npos) continue;
    std::string bdf = trim(line.substr(colon + 1));
    if (bdf.size() >= 12) bdf = bdf.substr(bdf.size() - 12);  // Keep xxxx:yy:zz.z
    if (!bdf.empty()) {
      bdf_out = bdf;
      return true;
    }
  }
  return false;
}

std::string canonical_path(const std::string& path) {
  if (path.empty()) return path;
  std::error_code ec;
  auto p = std::filesystem::path(path);
  auto canonical = std::filesystem::canonical(p, ec);
  if (ec) return path;
  return canonical.string();
}

std::vector<std::string> extract_pci_segments(const std::string& path) {
  std::vector<std::string> segments;
  if (path.empty()) return segments;
  std::stringstream ss(path);
  std::string item;
  while (std::getline(ss, item, '/')) {
    if (item.find(':') != std::string::npos && item.size() >= 7) {
      segments.push_back(item);
    }
  }
  return segments;
}

int compute_pci_score(const std::vector<std::string>& gpu_path,
                      const std::vector<std::string>& nic_path) {
  if (gpu_path.empty() || nic_path.empty()) return -1000;
  size_t min_len = std::min(gpu_path.size(), nic_path.size());
  size_t common = 0;
  for (size_t i = 0; i < min_len; ++i) {
    if (gpu_path[i] == nic_path[i]) {
      ++common;
    } else {
      break;
    }
  }
  if (common == 0) return -1000;
  int distance = static_cast<int>(gpu_path.size() + nic_path.size() - 2 * common);
  return static_cast<int>(common * 100 - distance);
}

}  // namespace

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

  std::string gpu_bdf;
  std::vector<std::string> gpu_pci_segments;
  if (read_gpu_pci_bdf(gpu_id_, gpu_bdf)) {
    std::string gpu_sysfs = canonical_path("/sys/bus/pci/devices/" + gpu_bdf);
    gpu_pci_segments = extract_pci_segments(gpu_sysfs);
    std::cout << "[ChannelManager] GPU " << gpu_id_ << " PCI BDF " << gpu_bdf
              << " (" << gpu_sysfs << ")" << std::endl;
  } else {
    std::cerr << "[ChannelManager] Warning: Unable to determine PCI path for GPU "
              << gpu_id_ << ". Falling back to naive NIC ordering." << std::endl;
  }

  struct Candidate {
    int dev = -1;
    tcpx_net_properties props{};
    std::vector<std::string> pci_segments;
    int score = -1000;
    bool cuda_supported = false;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(tcpx_dev_count);
  for (int dev = 0; dev < tcpx_dev_count; ++dev) {
    tcpx_net_properties props{};
    if (tcpx_get_properties(dev, &props) != 0) continue;

    Candidate cand;
    cand.dev = dev;
    cand.props = props;
    cand.cuda_supported = (props.ptr_support & NCCL_PTR_CUDA) != 0;
    std::string nic_path = props.pci_path ? props.pci_path : "";
    if (!nic_path.empty()) {
      nic_path = canonical_path(nic_path);
      cand.pci_segments = extract_pci_segments(nic_path);
      cand.score = compute_pci_score(gpu_pci_segments, cand.pci_segments);
    }
    candidates.push_back(cand);
  }

  if (candidates.empty()) {
    std::cerr << "[ChannelManager] No TCPX devices available" << std::endl;
    num_channels_ = 0;
    return;
  }

  std::vector<Candidate> sorted = candidates;
  std::sort(sorted.begin(), sorted.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score == b.score) return a.dev < b.dev;
    return a.score > b.score;
  });

  std::vector<Candidate> selected;
  selected.reserve(num_channels_);
  for (const auto& cand : sorted) {
    if (!cand.cuda_supported) continue;
    if (!gpu_pci_segments.empty() && cand.score < 0) continue;
    selected.push_back(cand);
    if ((int)selected.size() == num_channels_) break;
  }

  if (selected.empty()) {
    std::cerr << "[ChannelManager] Warning: No GPU-direct capable NICs detected for GPU "
              << gpu_id_ << ". Falling back to first enumerated NIC." << std::endl;
    selected.push_back(sorted.front());
  }

  if ((int)selected.size() < num_channels_) {
    std::cerr << "[ChannelManager] Warning: Requested " << num_channels_
              << " channels but only " << selected.size()
              << " GPU-direct NICs matched this GPU. Reducing channel count." << std::endl;
    num_channels_ = selected.size();
  }

  channels_.resize(num_channels_);

  for (int i = 0; i < num_channels_; ++i) {
    ChannelResources& ch = channels_[i];
    const auto& cand = selected[i];

    ch.channel_id = i;
    ch.net_dev = cand.dev;

    const char* nic_name = cand.props.name ? cand.props.name : "unknown";
    const char* nic_pci = cand.props.pci_path ? cand.props.pci_path : "";
    std::cout << "[ChannelManager] Channel " << i << " → netDev " << cand.dev
              << " (" << nic_name << ", PCI=" << nic_pci
              << ", score=" << cand.score << ")" << std::endl;

    ch.listen_comm = nullptr;
    ch.recv_comm = nullptr;
    ch.send_comm = nullptr;

    ch.recv_dev_handle = ch.recv_dev_handle_storage.data();
    ch.send_dev_handle = ch.send_dev_handle_storage.data();
    std::memset(ch.recv_dev_handle_storage.data(), 0, ch.recv_dev_handle_storage.size());
    std::memset(ch.send_dev_handle_storage.data(), 0, ch.send_dev_handle_storage.size());

    ch.mhandle = nullptr;
    ch.sliding_window = new SlidingWindow(16);
    ch.bytes_transferred = 0;
    ch.chunks_processed = 0;
  }

  std::cout << "[ChannelManager] Created " << num_channels_
            << " channel(s) for GPU " << gpu_id_
            << " (TCPX devices available: " << tcpx_dev_count << ")" << std::endl;
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
    std::cerr << "[ChannelManager] FATAL: Invalid channel index " << idx
              << " (valid range: 0-" << (num_channels_ - 1) << ")" << std::endl;
    std::cerr << "[ChannelManager] This indicates a configuration bug (env asked for more channels than available)" << std::endl;
    std::cerr << "[ChannelManager] Aborting to make the bug obvious instead of silently using channel 0" << std::endl;
    std::abort();  // Fail fast instead of masking the bug
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
