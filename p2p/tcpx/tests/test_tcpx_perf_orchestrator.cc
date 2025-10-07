/**
 * @file test_tcpx_perf_orchestrator.cc
 * @brief Single-process orchestrator for 8 GPUs
 * 
 * Architecture:
 * - 1 process per node
 * - Manages all 8 GPUs
 * - Each GPU can use multiple channels
 * - All 4 NICs available to all GPUs (no devmem conflicts)
 * 
 * Usage:
 *   # Server
 *   UCCL_TCPX_NUM_CHANNELS=8 ./tests/test_tcpx_perf_orchestrator server
 *   
 *   # Client
 *   UCCL_TCPX_NUM_CHANNELS=8 ./tests/test_tcpx_perf_orchestrator client <server_ip>
 */

#include "../include/channel_manager.h"
#include "../include/bootstrap.h"
#include "../include/tcpx_interface.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr int kNumGPUs = 8;
constexpr size_t kTransferSize = 64 * 1024 * 1024;  // 64 MB per GPU
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

struct GPUContext {
  int gpu_id;
  CUdevice cuDev;
  CUcontext cuCtx;
  CUdeviceptr d_base;
  void* gpu_buf;
  ChannelManager* mgr;
  int num_channels;
  int bootstrap_port;
  std::vector<ncclNetHandle_v7> handles;  // Cache handles from server_listen_all

  GPUContext() : gpu_id(-1), cuDev(0), cuCtx(nullptr), d_base(0),
                 gpu_buf(nullptr), mgr(nullptr), num_channels(0), bootstrap_port(0) {}

  ~GPUContext() {
    if (mgr) delete mgr;
    if (d_base) cuMemFree(d_base);
    // Release CUDA primary context (avoid leak)
    if (cuCtx) {
      cuDevicePrimaryCtxRelease(cuDev);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <server|client> [server_ip]" << std::endl;
    return 1;
  }

  bool is_server = (std::strcmp(argv[1], "server") == 0);
  const char* server_ip = (argc > 2) ? argv[2] : nullptr;
  
  if (!is_server && !server_ip) {
    std::cerr << "[ERROR] Client mode requires server_ip" << std::endl;
    return 1;
  }
  
  // Configuration
  int num_channels_per_gpu = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 8);
  int bootstrap_port_base = getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 20000);
  
  std::cout << "=== Single-Process P2P Orchestrator ===" << std::endl;
  std::cout << "Role: " << (is_server ? "server" : "client") << std::endl;
  std::cout << "GPUs: " << kNumGPUs << std::endl;
  std::cout << "Channels per GPU: " << num_channels_per_gpu << std::endl;
  std::cout << "Bootstrap port base: " << bootstrap_port_base << std::endl;
  std::cout << "=======================================" << std::endl << std::endl;

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
  std::cout << "[INFO] All GPUs can use all " << ndev << " NICs (single-process architecture)" << std::endl << std::endl;

  // Initialize CUDA for all GPUs
  if (!cuda_check(cuInit(0), "cuInit")) {
    return 1;
  }

  // Create contexts for all GPUs
  std::vector<GPUContext> gpus(kNumGPUs);
  
  for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
    GPUContext& ctx = gpus[gpu_id];
    ctx.gpu_id = gpu_id;
    ctx.num_channels = num_channels_per_gpu;
    ctx.bootstrap_port = bootstrap_port_base + gpu_id;
    
    std::cout << "[GPU " << gpu_id << "] Initializing..." << std::endl;
    
    if (!cuda_check(cuDeviceGet(&ctx.cuDev, gpu_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&ctx.cuCtx, ctx.cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
      return 1;
    }
    
    // Allocate GPU buffer (4KB aligned)
    if (!cuda_check(cuMemAlloc(&ctx.d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      return 1;
    }
    
    uintptr_t addr = static_cast<uintptr_t>(ctx.d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    ctx.gpu_buf = reinterpret_cast<void*>(addr);
    
    std::cout << "[GPU " << gpu_id << "] Buffer allocated: " << ctx.gpu_buf << std::endl;
    
    // Create ChannelManager
    ctx.mgr = new ChannelManager(ctx.num_channels, gpu_id);
  }
  
  std::cout << "\n[INFO] All GPUs initialized" << std::endl << std::endl;

  if (is_server) {
    // ===== SERVER: Manage all 8 GPUs =====
    
    std::cout << "[SERVER] Step 1: Listening on all GPUs..." << std::endl;

    // Listen on all GPUs and cache handles
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      if (ctx.mgr->server_listen_all(ctx.handles) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": server_listen_all failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Listening on " << ctx.mgr->get_num_channels()
                << " channels (cached " << ctx.handles.size() << " handles)" << std::endl;
    }

    std::cout << "\n[SERVER] Step 2: Bootstrap handshake..." << std::endl;

    // Bootstrap: send cached handles to client (one connection per GPU)
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      int bootstrap_fd = -1;
      if (bootstrap_server_create(ctx.bootstrap_port, &bootstrap_fd) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_server_create failed" << std::endl;
        return 1;
      }

      // Reuse cached handles (no duplicate listen)
      if (bootstrap_server_send_handles(bootstrap_fd, ctx.handles) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_server_send_handles failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }

      close(bootstrap_fd);
      std::cout << "[GPU " << gpu_id << "] Sent " << ctx.handles.size() << " handles" << std::endl;
    }
    
    std::cout << "\n[SERVER] Step 3: Accepting connections..." << std::endl;
    
    // Accept all connections
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];
      
      if (ctx.mgr->server_accept_all() != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": server_accept_all failed" << std::endl;
        return 1;
      }
      
      std::cout << "[GPU " << gpu_id << "] Accepted " << ctx.mgr->get_num_channels() << " connections" << std::endl;
    }
    
    std::cout << "\n[SERVER] Step 4: Registering memory..." << std::endl;
    
    // Register memory on all GPUs
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];
      
      if (!cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent")) {
        return 1;
      }
      
      if (ctx.mgr->register_memory(ctx.gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, true) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": register_memory failed" << std::endl;
        return 1;
      }
      
      std::cout << "[GPU " << gpu_id << "] Registered memory on " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }
    
    std::cout << "\n=== ALL GPUs READY (SERVER) ===" << std::endl;
    std::cout << "Total channels: " << kNumGPUs * num_channels_per_gpu << std::endl;
    std::cout << "Architecture: Single process, all NICs shared" << std::endl;
    
    // Cleanup
    for (auto& ctx : gpus) {
      ctx.mgr->deregister_memory(true);
      ctx.mgr->close_all(true);
    }
    
  } else {
    // ===== CLIENT: Manage all 8 GPUs =====
    
    std::cout << "[CLIENT] Step 1: Bootstrap handshake..." << std::endl;
    
    // Bootstrap: receive handles from server
    std::vector<std::vector<ncclNetHandle_v7>> all_handles(kNumGPUs);
    
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];
      
      int bootstrap_fd = -1;
      if (bootstrap_client_connect(server_ip, ctx.bootstrap_port, &bootstrap_fd) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_client_connect failed" << std::endl;
        return 1;
      }
      
      if (bootstrap_client_recv_handles(bootstrap_fd, all_handles[gpu_id]) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_client_recv_handles failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      
      close(bootstrap_fd);
      std::cout << "[GPU " << gpu_id << "] Received " << all_handles[gpu_id].size() << " handles" << std::endl;
    }
    
    std::cout << "\n[CLIENT] Step 2: Connecting to server..." << std::endl;
    
    // Connect all GPUs
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];
      
      if (ctx.mgr->client_connect_all(all_handles[gpu_id]) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": client_connect_all failed" << std::endl;
        return 1;
      }
      
      std::cout << "[GPU " << gpu_id << "] Connected " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }
    
    std::cout << "\n[CLIENT] Step 3: Registering memory..." << std::endl;
    
    // Register memory on all GPUs
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];
      
      if (!cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent")) {
        return 1;
      }
      
      if (ctx.mgr->register_memory(ctx.gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, false) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": register_memory failed" << std::endl;
        return 1;
      }
      
      std::cout << "[GPU " << gpu_id << "] Registered memory on " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }
    
    std::cout << "\n=== ALL GPUs READY (CLIENT) ===" << std::endl;
    std::cout << "Total channels: " << kNumGPUs * num_channels_per_gpu << std::endl;
    std::cout << "Architecture: Single process, all NICs shared" << std::endl;
    
    // Wait for server
    std::cout << "\n[CLIENT] Waiting 5 seconds for server..." << std::endl;
    sleep(5);
    
    // Cleanup
    for (auto& ctx : gpus) {
      ctx.mgr->deregister_memory(false);
      ctx.mgr->close_all(false);
    }
  }
  
  std::cout << "\n[INFO] Test completed successfully" << std::endl;
  std::cout << "Next: Add actual data transfer and performance measurement" << std::endl;
  
  return 0;
}

