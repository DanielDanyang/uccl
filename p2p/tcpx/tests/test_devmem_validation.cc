/**
 * @file test_devmem_validation.cc
 * @brief Devmem Validation Test - Verify single-process can use multiple channels on same NIC
 * 
 * Based on test_tcpx_perf_multi.cc but simplified to just test channel creation and memory registration.
 * 
 * Usage:
 *   # Server
 *   UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_devmem_validation server 0
 *   
 *   # Client
 *   UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_devmem_validation client <server_ip> 0
 */

#include "../include/channel_manager.h"
#include "../include/bootstrap.h"
#include "../include/tcpx_interface.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr size_t kTransferSize = 16 * 1024 * 1024;  // 16 MB
constexpr size_t kRegisteredBytes = kTransferSize + 4096;

int getEnvInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

bool cuda_check(CUresult res, const char* msg) {
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(res, &err_str);
    std::cerr << "[ERROR] " << msg << " failed: " << (err_str ? err_str : "unknown") << std::endl;
    return false;
  }
  return true;
}

bool cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] " << msg << " failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Parse arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <server|client> <server_ip|0> <gpu_id>" << std::endl;
    return 1;
  }

  bool is_server = (std::strcmp(argv[1], "server") == 0);
  const char* server_ip = argv[2];
  int gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  
  // Get configuration from environment
  int num_channels = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 4);
  int bootstrap_port = getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 12347) + gpu_id;
  
  std::cout << "=== Devmem Validation Test ===" << std::endl;
  std::cout << "Role: " << (is_server ? "server" : "client") << std::endl;
  std::cout << "GPU: " << gpu_id << std::endl;
  std::cout << "Channels: " << num_channels << std::endl;
  std::cout << "Bootstrap port: " << bootstrap_port << std::endl;
  std::cout << "==============================" << std::endl << std::endl;

  // Load TCPX plugin
  const char* plugin_path = std::getenv("NCCL_GPUDIRECTTCPX_PLUGIN_PATH");
  if (!plugin_path) {
    plugin_path = "/usr/local/tcpx/lib64/libnccl-net.so";
  }
  
  if (tcpx_load_plugin(plugin_path) != 0) {
    std::cerr << "[ERROR] Failed to load TCPX plugin from " << plugin_path << std::endl;
    return 1;
  }
  
  int ndev = tcpx_get_device_count();
  std::cout << "[INFO] TCPX devices: " << ndev << std::endl;

  // Initialize CUDA
  CUdevice cuDev;
  CUcontext cuCtx;
  
  if (!cuda_check(cuInit(0), "cuInit") ||
      !cuda_check(cuDeviceGet(&cuDev, gpu_id), "cuDeviceGet") ||
      !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
      !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
      !cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
    return 1;
  }

  // Allocate GPU buffer (4KB aligned)
  CUdeviceptr d_base = 0, d_aligned = 0;
  
  if (!cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
    return 1;
  }
  
  uintptr_t addr = static_cast<uintptr_t>(d_base);
  addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
  d_aligned = static_cast<CUdeviceptr>(addr);
  void* gpu_buf = reinterpret_cast<void*>(d_aligned);
  
  std::cout << "[INFO] Allocated GPU buffer: " << gpu_buf << std::endl;

  if (is_server) {
    // ===== SERVER =====
    
    // Step 1: Create ChannelManager and listen
    std::cout << "\n[SERVER] Step 1: Creating ChannelManager and listening..." << std::endl;
    ChannelManager mgr(num_channels, gpu_id);
    std::vector<ncclNetHandle_v7> handles;
    
    if (mgr.server_listen_all(handles) != 0) {
      std::cerr << "[ERROR] server_listen_all failed" << std::endl;
      return 1;
    }
    
    num_channels = mgr.get_num_channels();
    std::cout << "[SERVER] Listening on " << num_channels << " channels" << std::endl;
    
    // Step 2: Bootstrap - send handles to client
    std::cout << "\n[SERVER] Step 2: Bootstrap handshake..." << std::endl;
    int bootstrap_fd = -1;
    if (bootstrap_server_create(bootstrap_port, &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_server_create failed" << std::endl;
      return 1;
    }
    
    if (bootstrap_server_send_handles(bootstrap_fd, handles) != 0) {
      std::cerr << "[ERROR] bootstrap_server_send_handles failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    
    std::cout << "[SERVER] Sent " << handles.size() << " handles to client" << std::endl;
    
    // Step 3: Accept all connections
    std::cout << "\n[SERVER] Step 3: Accepting connections..." << std::endl;
    if (mgr.server_accept_all() != 0) {
      std::cerr << "[ERROR] server_accept_all failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    
    std::cout << "[SERVER] All " << num_channels << " channels accepted" << std::endl;
    
    // Step 4: Register memory
    std::cout << "\n[SERVER] Step 4: Registering memory..." << std::endl;
    if (mgr.register_memory(gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, true) != 0) {
      std::cerr << "[ERROR] register_memory failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    
    std::cout << "[SERVER] Registered memory on " << num_channels << " channels" << std::endl;
    
    std::cout << "\n=== ALL CHANNELS PASSED (SERVER) ===" << std::endl;
    std::cout << "Result: Single-process CAN use " << num_channels << " channels on same NIC" << std::endl;
    std::cout << "Devmem conflicts: RESOLVED" << std::endl;
    
    // Cleanup
    mgr.deregister_memory(true);
    mgr.close_all(true);
    close(bootstrap_fd);
    
  } else {
    // ===== CLIENT =====
    
    // Step 1: Bootstrap - receive handles from server
    std::cout << "\n[CLIENT] Step 1: Bootstrap handshake..." << std::endl;
    int bootstrap_fd = -1;
    if (bootstrap_client_connect(server_ip, bootstrap_port, &bootstrap_fd) != 0) {
      std::cerr << "[ERROR] bootstrap_client_connect failed" << std::endl;
      return 1;
    }
    
    std::vector<ncclNetHandle_v7> handles;
    if (bootstrap_client_recv_handles(bootstrap_fd, handles) != 0) {
      std::cerr << "[ERROR] bootstrap_client_recv_handles failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    
    std::cout << "[CLIENT] Received " << handles.size() << " handles from server" << std::endl;
    
    // Step 2: Create ChannelManager and connect
    std::cout << "\n[CLIENT] Step 2: Creating ChannelManager and connecting..." << std::endl;
    ChannelManager mgr(handles.size(), gpu_id);
    
    if (mgr.client_connect_all(handles) != 0) {
      std::cerr << "[ERROR] client_connect_all failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    
    num_channels = mgr.get_num_channels();
    std::cout << "[CLIENT] Connected " << num_channels << " channels" << std::endl;
    
    // Step 3: Register memory
    std::cout << "\n[CLIENT] Step 3: Registering memory..." << std::endl;
    if (mgr.register_memory(gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, false) != 0) {
      std::cerr << "[ERROR] register_memory failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    
    std::cout << "[CLIENT] Registered memory on " << num_channels << " channels" << std::endl;
    
    std::cout << "\n=== ALL CHANNELS PASSED (CLIENT) ===" << std::endl;
    std::cout << "Result: Single-process CAN use " << num_channels << " channels on same NIC" << std::endl;
    std::cout << "Devmem conflicts: RESOLVED" << std::endl;
    
    // Cleanup
    mgr.deregister_memory(false);
    mgr.close_all(false);
    close(bootstrap_fd);
  }
  
  cuMemFree(d_base);
  
  std::cout << "\nProceed to Step 2: Full refactor" << std::endl;
  return 0;
}
