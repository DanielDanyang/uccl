/**
 * @file test_tcpx_perf_orchestrator.cc
 * @brief Single-process orchestrator for 8 GPUs - Core of single-process architecture
 *
 * PURPOSE:
 * ========
 * This is the main test program for the single-process P2P architecture refactor.
 * It demonstrates that a SINGLE process can manage ALL 8 GPUs and share ALL 4 NICs
 * without devmem conflicts (which plagued the old multi-process architecture).
 *
 * ARCHITECTURE COMPARISON:
 * ========================
 *
 * OLD (Multi-Process):
 *   Node
 *   ├── Process 0 (GPU 0, eth1 only, 1 channel)  ← devmem conflict
 *   ├── Process 1 (GPU 1, eth1 only, 1 channel)  ← devmem conflict
 *   ├── Process 2 (GPU 2, eth2 only, 1 channel)
 *   ...
 *   └── Process 7 (GPU 7, eth4 only, 1 channel)
 *
 *   Problem: Multiple processes cannot share NICs due to devmem-tcp limitations
 *   Result: Each GPU stuck with 1 NIC, low bandwidth
 *
 * NEW (Single-Process):
 *   Node
 *   └── Single Process
 *       ├── GPU 0 (8 channels, all 4 NICs available)  ← No conflict!
 *       ├── GPU 1 (8 channels, all 4 NICs available)
 *       ├── GPU 2 (8 channels, all 4 NICs available)
 *       ...
 *       └── GPU 7 (8 channels, all 4 NICs available)
 *
 *   Benefit: All GPUs can use all NICs, higher bandwidth potential
 *   Total: 64 channels (8 GPUs × 8 channels), 4 NICs shared
 *
 * EXECUTION FLOW:
 * ===============
 *
 * Server Side:
 *   1. Initialize all 8 GPUs (CUDA contexts, allocate buffers)
 *   2. Listen on all channels (ChannelManager.server_listen_all)
 *   3. Bootstrap handshake (send channel handles to client)
 *   4. Accept connections from client
 *   5. Register GPU memory for RDMA (tcpx_reg_mr)
 *   6. [Future] Receive data and measure performance
 *
 * Client Side:
 *   1. Initialize all 8 GPUs
 *   2. Bootstrap handshake (receive channel handles from server)
 *   3. Connect to server channels
 *   4. Register GPU memory for RDMA
 *   5. [Future] Send data and measure performance
 *
 * KEY DESIGN DECISIONS:
 * =====================
 *
 * 1. Per-GPU ChannelManager:
 *    - Each GPU has its own ChannelManager instance
 *    - Manages multiple channels (default: 8) for that GPU
 *    - Handles TCPX listen/connect/accept for all channels
 *
 * 2. Bootstrap Strategy:
 *    - One bootstrap connection per GPU (ports 20000-20007)
 *    - Each connection sends ALL channel handles for that GPU
 *    - Avoids the "one bootstrap per channel" overhead
 *
 * 3. Handle Caching:
 *    - Handles from server_listen_all() are cached in GPUContext
 *    - Reused during bootstrap (no duplicate listen calls)
 *    - Prevents resource leaks and "already listening" errors
 *
 * 4. Sequential Execution:
 *    - All GPUs listen → all GPUs bootstrap → all GPUs accept
 *    - Avoids race conditions from concurrent listen/accept
 *    - Simpler to debug than fully concurrent approach
 *
 * CURRENT STATUS:
 * ===============
 * This version only establishes channels and registers memory.
 * Actual data transfer and performance measurement will be added in Step 3.
 *
 * USAGE:
 * ======
 *   # Server (Node 0)
 *   UCCL_TCPX_NUM_CHANNELS=8 ./tests/test_tcpx_perf_orchestrator server
 *
 *   # Client (Node 1)
 *   UCCL_TCPX_NUM_CHANNELS=8 ./tests/test_tcpx_perf_orchestrator client <server_ip>
 *
 * ENVIRONMENT VARIABLES:
 * ======================
 *   UCCL_TCPX_NUM_CHANNELS         - Channels per GPU (default: 8)
 *   UCCL_TCPX_BOOTSTRAP_PORT_BASE  - Base port for bootstrap (default: 20000)
 *   NCCL_GPUDIRECTTCPX_SOCKET_IFNAME - NICs to use (should be "eth1,eth2,eth3,eth4")
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
#include <chrono>
#include <thread>

namespace {

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr int kNumGPUs = 8;  // A3-high instances have 8× H100 GPUs per node
constexpr size_t kDefaultTransferSize = 64 * 1024 * 1024;  // 64 MB per GPU (default)
constexpr size_t kMaxTransferSize = 256 * 1024 * 1024;  // 256 MB max per GPU
constexpr size_t kRegisteredBytes = kMaxTransferSize + 4096;  // Extra space for alignment

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Get integer from environment variable with default fallback
 * @param name Environment variable name
 * @param def Default value if not set
 * @return Parsed integer or default
 */
int getEnvInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

/**
 * @brief Get size_t from environment variable with default fallback
 * @param name Environment variable name
 * @param def Default value if not set
 * @return Parsed size_t or default
 */
size_t getEnvSize(const char* name, size_t def) {
  const char* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

/**
 * @brief Check CUDA Driver API result and print error if failed
 * @param res CUDA result code
 * @param msg Context message for error
 * @return true if success, false if failed
 */
bool cuda_check(CUresult res, const char* msg) {
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(res, &err_str);
    std::cerr << "[ERROR] " << msg << " failed: " << (err_str ? err_str : "unknown") << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief Check CUDA Runtime API result and print error if failed
 * @param err CUDA error code
 * @param msg Context message for error
 * @return true if success, false if failed
 */
bool cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] " << msg << " failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

// ============================================================================
// GPU CONTEXT STRUCTURE
// ============================================================================

/**
 * @brief Per-GPU context holding all resources for one GPU
 *
 * This structure encapsulates everything needed to manage one GPU:
 * - CUDA context and device handles
 * - GPU memory buffer (4KB aligned for devmem-tcp)
 * - ChannelManager for TCPX channels
 * - Bootstrap configuration
 * - Cached channel handles (to avoid duplicate listen calls)
 *
 * Lifecycle:
 * 1. Constructor: Initialize to default values
 * 2. Main: Allocate CUDA resources, create ChannelManager
 * 3. Destructor: Clean up all resources (memory, context, manager)
 */
struct GPUContext {
  // GPU identification
  int gpu_id;                  // GPU index (0-7)

  // CUDA resources
  CUdevice cuDev;              // CUDA device handle
  CUcontext cuCtx;             // CUDA context (retained primary context)
  CUdeviceptr d_base;          // Base GPU memory allocation
  void* gpu_buf;               // 4KB-aligned GPU buffer pointer

  // TCPX channel management
  ChannelManager* mgr;         // Manages all channels for this GPU
  int num_channels;            // Number of channels (e.g., 8)

  // Bootstrap configuration
  int bootstrap_port;          // Port for bootstrap handshake (20000 + gpu_id)

  // Handle caching (CRITICAL: prevents duplicate listen calls)
  std::vector<ncclNetHandle_v7> handles;  // Cached from server_listen_all()

  /**
   * @brief Default constructor - initialize all fields to safe defaults
   */
  GPUContext() : gpu_id(-1), cuDev(0), cuCtx(nullptr), d_base(0),
                 gpu_buf(nullptr), mgr(nullptr), num_channels(0), bootstrap_port(0) {}

  /**
   * @brief Destructor - clean up all resources
   *
   * Order matters:
   * 1. Delete ChannelManager (closes TCPX channels)
   * 2. Free GPU memory
   * 3. Release CUDA primary context (matches cuDevicePrimaryCtxRetain)
   */
  ~GPUContext() {
    if (mgr) delete mgr;
    if (d_base) cuMemFree(d_base);
    // CRITICAL: Release primary context to avoid leak
    // Every cuDevicePrimaryCtxRetain() must have a matching Release()
    if (cuCtx) {
      cuDevicePrimaryCtxRelease(cuDev);
    }
  }
};

}  // namespace

// ============================================================================
// MAIN FUNCTION
// ============================================================================

/**
 * @brief Main entry point for single-process orchestrator
 *
 * This function orchestrates the entire P2P setup for all 8 GPUs:
 * 1. Parse command-line arguments (server vs client)
 * 2. Load TCPX plugin
 * 3. Initialize all 8 GPUs (CUDA contexts, buffers, ChannelManagers)
 * 4. Execute server or client flow
 * 5. Clean up resources
 *
 * @param argc Argument count
 * @param argv Arguments: <server|client> [server_ip]
 * @return 0 on success, 1 on failure
 */
int main(int argc, char** argv) {
  // ========================================
  // Parse Arguments
  // ========================================

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

  // ========================================
  // Configuration from Environment
  // ========================================

  int num_channels_per_gpu = getEnvInt("UCCL_TCPX_NUM_CHANNELS", 8);
  int bootstrap_port_base = getEnvInt("UCCL_TCPX_BOOTSTRAP_PORT_BASE", 20000);

  std::cout << "=== Single-Process P2P Orchestrator ===" << std::endl;
  std::cout << "Role: " << (is_server ? "server" : "client") << std::endl;
  std::cout << "GPUs: " << kNumGPUs << std::endl;
  std::cout << "Channels per GPU: " << num_channels_per_gpu << std::endl;
  std::cout << "Bootstrap port base: " << bootstrap_port_base << std::endl;
  std::cout << "=======================================" << std::endl << std::endl;

  // ========================================
  // Load TCPX Plugin
  // ========================================

  // The TCPX plugin provides the GPUDirect-TCPX network interface
  // It must be loaded before any TCPX operations
  const char* plugin_path = std::getenv("NCCL_GPUDIRECTTCPX_PLUGIN_PATH");
  if (!plugin_path) {
    plugin_path = "/usr/local/tcpx/lib64/libnccl-net.so";
  }

  if (tcpx_load_plugin(plugin_path) != 0) {
    std::cerr << "[ERROR] Failed to load TCPX plugin from " << plugin_path << std::endl;
    return 1;
  }

  // Query available TCPX devices (NICs)
  // In single-process architecture, all NICs are visible to all GPUs
  int ndev = tcpx_get_device_count();
  std::cout << "[INFO] TCPX devices: " << ndev << std::endl;
  std::cout << "[INFO] All GPUs can use all " << ndev << " NICs (single-process architecture)" << std::endl << std::endl;

  // ========================================
  // Initialize CUDA
  // ========================================

  // Initialize CUDA Driver API (required for cuMemAlloc, cuCtxSetCurrent, etc.)
  if (!cuda_check(cuInit(0), "cuInit")) {
    return 1;
  }

  // ========================================
  // Create GPU Contexts
  // ========================================

  // Allocate context structures for all 8 GPUs
  // These will be populated in the loop below
  std::vector<GPUContext> gpus(kNumGPUs);
  
  // ========================================
  // Initialize Each GPU
  // ========================================

  // For each GPU, we need to:
  // 1. Get CUDA device handle
  // 2. Retain primary context (for CUDA operations)
  // 3. Allocate GPU memory (4KB aligned for devmem-tcp)
  // 4. Create ChannelManager (for TCPX channels)

  for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
    GPUContext& ctx = gpus[gpu_id];
    ctx.gpu_id = gpu_id;
    ctx.num_channels = num_channels_per_gpu;
    ctx.bootstrap_port = bootstrap_port_base + gpu_id;  // e.g., 20000, 20001, ...

    std::cout << "[GPU " << gpu_id << "] Initializing..." << std::endl;

    // Get CUDA device handle
    if (!cuda_check(cuDeviceGet(&ctx.cuDev, gpu_id), "cuDeviceGet")) {
      return 1;
    }

    // Retain primary context (CRITICAL: must be released in destructor)
    if (!cuda_check(cuDevicePrimaryCtxRetain(&ctx.cuCtx, ctx.cuDev), "cuDevicePrimaryCtxRetain")) {
      return 1;
    }

    // Set current context (for subsequent CUDA calls)
    if (!cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent")) {
      return 1;
    }

    // Set device for Runtime API calls
    if (!cuda_check(cudaSetDevice(gpu_id), "cudaSetDevice")) {
      return 1;
    }

    // Allocate GPU buffer with extra space for alignment
    // devmem-tcp requires 4KB (4096 byte) alignment
    if (!cuda_check(cuMemAlloc(&ctx.d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      return 1;
    }

    // Align to 4KB boundary
    // Formula: (addr + 4095) & ~4095 rounds up to next 4KB boundary
    uintptr_t addr = static_cast<uintptr_t>(ctx.d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    ctx.gpu_buf = reinterpret_cast<void*>(addr);

    std::cout << "[GPU " << gpu_id << "] Buffer allocated: " << ctx.gpu_buf << std::endl;

    // Create ChannelManager for this GPU
    // This will manage all TCPX channels for this GPU
    ctx.mgr = new ChannelManager(ctx.num_channels, gpu_id);
  }

  std::cout << "\n[INFO] All GPUs initialized" << std::endl << std::endl;

  // ========================================
  // SERVER or CLIENT Flow
  // ========================================

  if (is_server) {
    // ========================================================================
    // SERVER FLOW
    // ========================================================================

    // The server flow has 4 main steps:
    // 1. Listen on all channels (create listen_comm for each channel)
    // 2. Bootstrap handshake (send channel handles to client)
    // 3. Accept connections (wait for client to connect)
    // 4. Register memory (prepare GPU buffers for RDMA)

    // ========================================
    // Step 1: Listen on All Channels
    // ========================================

    std::cout << "[SERVER] Step 1: Listening on all GPUs..." << std::endl;

    // For each GPU, call server_listen_all() to:
    // - Create listen_comm for each channel
    // - Generate handles that client will use to connect
    // - Cache handles in GPUContext (CRITICAL: avoid duplicate listen)

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // CRITICAL: Cache handles in ctx.handles to avoid duplicate listen
      // Calling server_listen_all() twice would leak listen_comm descriptors
      if (ctx.mgr->server_listen_all(ctx.handles) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": server_listen_all failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Listening on " << ctx.mgr->get_num_channels()
                << " channels (cached " << ctx.handles.size() << " handles)" << std::endl;
    }

    // ========================================
    // Step 2: Bootstrap Handshake
    // ========================================

    std::cout << "\n[SERVER] Step 2: Bootstrap handshake..." << std::endl;

    // Bootstrap is a simple TCP connection used to exchange channel handles
    // Strategy: One bootstrap connection per GPU
    // - Port: 20000 + gpu_id (e.g., GPU 0 → 20000, GPU 1 → 20001)
    // - Payload: All channel handles for that GPU (e.g., 8 handles)
    //
    // Alternative (not used): One bootstrap per channel would require 64 connections

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Create bootstrap server socket and wait for client
      int bootstrap_fd = -1;
      if (bootstrap_server_create(ctx.bootstrap_port, &bootstrap_fd) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_server_create failed" << std::endl;
        return 1;
      }

      // Send cached handles to client
      // CRITICAL: Use ctx.handles (cached from Step 1), not a new listen call
      if (bootstrap_server_send_handles(bootstrap_fd, ctx.handles) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_server_send_handles failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }

      close(bootstrap_fd);
      std::cout << "[GPU " << gpu_id << "] Sent " << ctx.handles.size() << " handles" << std::endl;
    }
    
    // ========================================
    // Step 3: Accept Connections
    // ========================================

    std::cout << "\n[SERVER] Step 3: Accepting connections..." << std::endl;

    // Now that client has the handles, it will connect to each channel
    // We need to accept those connections
    // ChannelManager.server_accept_all() handles retry logic internally

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Accept all channel connections for this GPU
      // This creates recv_comm for each channel
      if (ctx.mgr->server_accept_all() != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": server_accept_all failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Accepted " << ctx.mgr->get_num_channels() << " connections" << std::endl;
    }

    // ========================================
    // Step 4: Register Memory
    // ========================================

    std::cout << "\n[SERVER] Step 4: Registering memory..." << std::endl;

    // Register GPU buffers with TCPX for RDMA (zero-copy transfers)
    // This calls tcpx_reg_mr() for each channel
    // The registered memory can then be used for irecv operations

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Set CUDA context for this GPU (required for memory operations)
      if (!cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent")) {
        return 1;
      }

      // Register memory for receiving
      // is_send=true means this is the receiving side
      if (ctx.mgr->register_memory(ctx.gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, true) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": register_memory failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Registered memory on " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }

    // ========================================
    // Server Ready
    // ========================================

    // Calculate actual total channels
    int total_channels_ready = 0;
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      total_channels_ready += gpus[gpu_id].mgr->get_num_channels();
    }

    std::cout << "\n=== ALL GPUs READY (SERVER) ===" << std::endl;
    std::cout << "Total channels: " << total_channels_ready << std::endl;
    std::cout << "Architecture: Single process, all NICs shared" << std::endl;

    // ========================================
    // Step 4: Data Receive (NEW in Step 3)
    // ========================================

    std::cout << "\n[SERVER] Step 4: Starting data receive..." << std::endl;

    // Get test parameters from environment
    size_t test_size_per_gpu = getEnvSize("UCCL_TCPX_PERF_SIZE", kDefaultTransferSize);
    int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 20);
    size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES", 524288);  // 512KB default
    const int kTransferTag = 42;

    // Validate test_size fits in registered buffer
    if (test_size_per_gpu > kMaxTransferSize) {
      std::cerr << "[ERROR] UCCL_TCPX_PERF_SIZE (" << test_size_per_gpu
                << ") exceeds max buffer size (" << kMaxTransferSize << ")" << std::endl;
      return 1;
    }

    // Calculate actual total channels (handle potential mismatches)
    int total_channels = 0;
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      total_channels += gpus[gpu_id].mgr->get_num_channels();
    }

    std::cout << "[SERVER] Test size per GPU: " << test_size_per_gpu << " bytes ("
              << (test_size_per_gpu / (1024 * 1024)) << " MB)" << std::endl;
    std::cout << "[SERVER] Total test size: " << (test_size_per_gpu * kNumGPUs) << " bytes ("
              << (test_size_per_gpu * kNumGPUs / (1024 * 1024)) << " MB)" << std::endl;
    std::cout << "[SERVER] Iterations: " << iterations << std::endl;
    std::cout << "[SERVER] Chunk size: " << chunk_bytes << " bytes" << std::endl;
    std::cout << "[SERVER] Total channels: " << total_channels << std::endl;

    double total_time_ms = 0.0;

    // Track all pending requests
    struct PendingRecv {
      void* request;
      int gpu_id;
      int channel_id;
      int chunk_idx;
    };
    std::vector<PendingRecv> pending_recvs;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "\n[SERVER] ===== Iteration " << iter << " =====" << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      pending_recvs.clear();

      // Phase 1: Post all receives (async)
      int global_chunk_idx = 0;
      for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
        GPUContext& ctx = gpus[gpu_id];
        int num_channels = ctx.mgr->get_num_channels();

        // Defensive: skip GPU if no channels (NIC probe failure, etc.)
        if (num_channels == 0) {
          std::cerr << "[WARNING] GPU " << gpu_id << " has 0 channels, skipping" << std::endl;
          continue;
        }

        size_t offset = 0;
        int local_chunk_idx = 0;

        while (offset < test_size_per_gpu) {
          size_t this_chunk = std::min(chunk_bytes, test_size_per_gpu - offset);

          // Round-robin across channels within this GPU
          int channel_local_id = local_chunk_idx % num_channels;
          ChannelResources& ch = ctx.mgr->get_channel(channel_local_id);

          // Calculate destination pointer within this GPU's buffer
          void* dst_ptr = reinterpret_cast<void*>(
              reinterpret_cast<uintptr_t>(ctx.gpu_buf) + offset);

          // Unique tag for this chunk (must match client)
          int tag = kTransferTag + iter * 100000 + gpu_id * 10000 + local_chunk_idx;

          // Post receive (async)
          void* recv_data[1] = {dst_ptr};
          int recv_sizes[1] = {static_cast<int>(this_chunk)};
          int recv_tags[1] = {tag};
          void* recv_mhandles[1] = {ch.mhandle};
          void* recv_request = nullptr;

          if (tcpx_irecv(ch.recv_comm, 1, recv_data, recv_sizes, recv_tags,
                         recv_mhandles, &recv_request) != 0) {
            std::cerr << "[ERROR] tcpx_irecv failed (GPU " << gpu_id
                      << " channel " << channel_local_id << " chunk " << local_chunk_idx << ")" << std::endl;
            return 1;
          }

          // Track this request
          pending_recvs.push_back({recv_request, gpu_id, channel_local_id, local_chunk_idx});

          offset += this_chunk;
          local_chunk_idx++;
          global_chunk_idx++;
        }
      }

      std::cout << "[SERVER] Posted " << pending_recvs.size() << " async receives" << std::endl;

      // Phase 2: Wait for all receives to complete
      for (auto& pending : pending_recvs) {
        int done = 0, received_size = 0;
        while (!done) {
          tcpx_test(pending.request, &done, &received_size);
          if (!done) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
        }

        // Mark as consumed
        GPUContext& ctx = gpus[pending.gpu_id];
        ChannelResources& ch = ctx.mgr->get_channel(pending.channel_id);
        tcpx_irecv_consumed(ch.recv_comm, 1, pending.request);
      }

      auto end = std::chrono::high_resolution_clock::now();
      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += elapsed_ms;

      // Calculate bandwidth: total bytes transferred / time
      size_t total_bytes = test_size_per_gpu * kNumGPUs;
      double bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
      std::cout << "[SERVER] Iteration " << iter << " completed in " << elapsed_ms
                << " ms, bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    }

    double avg_time_ms = total_time_ms / iterations;
    size_t total_bytes = test_size_per_gpu * kNumGPUs;
    double avg_bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);

    std::cout << "\n[SERVER] ===== Performance Summary =====" << std::endl;
    std::cout << "[SERVER] Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "[SERVER] Average bandwidth: " << avg_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "[SERVER] Total channels used: " << kNumGPUs * num_channels_per_gpu << std::endl;

    // ========================================
    // Cleanup
    // ========================================

    for (auto& ctx : gpus) {
      ctx.mgr->deregister_memory(true);
      ctx.mgr->close_all(true);
    }
    
  } else {
    // ========================================================================
    // CLIENT FLOW
    // ========================================================================

    // The client flow has 3 main steps:
    // 1. Bootstrap handshake (receive channel handles from server)
    // 2. Connect to server channels
    // 3. Register memory (prepare GPU buffers for RDMA)

    // ========================================
    // Step 1: Bootstrap Handshake
    // ========================================

    std::cout << "[CLIENT] Step 1: Bootstrap handshake..." << std::endl;

    // Connect to server's bootstrap sockets and receive channel handles
    // One connection per GPU (ports 20000-20007)
    // Each connection receives all channel handles for that GPU

    std::vector<std::vector<ncclNetHandle_v7>> all_handles(kNumGPUs);

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Connect to server's bootstrap socket
      int bootstrap_fd = -1;
      if (bootstrap_client_connect(server_ip, ctx.bootstrap_port, &bootstrap_fd) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_client_connect failed" << std::endl;
        return 1;
      }

      // Receive all channel handles for this GPU
      if (bootstrap_client_recv_handles(bootstrap_fd, all_handles[gpu_id]) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": bootstrap_client_recv_handles failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }

      close(bootstrap_fd);
      std::cout << "[GPU " << gpu_id << "] Received " << all_handles[gpu_id].size() << " handles" << std::endl;
    }
    
    // ========================================
    // Step 2: Connect to Server
    // ========================================

    std::cout << "\n[CLIENT] Step 2: Connecting to server..." << std::endl;

    // Use the received handles to connect to server's channels
    // ChannelManager.client_connect_all() calls tcpx_connect_v5() for each channel

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Connect all channels for this GPU
      // This creates send_comm for each channel
      if (ctx.mgr->client_connect_all(all_handles[gpu_id]) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": client_connect_all failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Connected " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }

    // ========================================
    // Step 3: Register Memory
    // ========================================

    std::cout << "\n[CLIENT] Step 3: Registering memory..." << std::endl;

    // Register GPU buffers with TCPX for RDMA
    // This calls tcpx_reg_mr() for each channel
    // The registered memory can then be used for isend operations

    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      GPUContext& ctx = gpus[gpu_id];

      // Set CUDA context for this GPU
      if (!cuda_check(cuCtxSetCurrent(ctx.cuCtx), "cuCtxSetCurrent")) {
        return 1;
      }

      // Register memory for sending
      // is_send=false means this is the sending side
      if (ctx.mgr->register_memory(ctx.gpu_buf, kRegisteredBytes, NCCL_PTR_CUDA, false) != 0) {
        std::cerr << "[ERROR] GPU " << gpu_id << ": register_memory failed" << std::endl;
        return 1;
      }

      std::cout << "[GPU " << gpu_id << "] Registered memory on " << ctx.mgr->get_num_channels() << " channels" << std::endl;
    }

    // ========================================
    // Client Ready
    // ========================================

    // Calculate actual total channels
    int total_channels_ready = 0;
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      total_channels_ready += gpus[gpu_id].mgr->get_num_channels();
    }

    std::cout << "\n=== ALL GPUs READY (CLIENT) ===" << std::endl;
    std::cout << "Total channels: " << total_channels_ready << std::endl;
    std::cout << "Architecture: Single process, all NICs shared" << std::endl;

    // ========================================
    // Step 4: Data Transfer (NEW in Step 3)
    // ========================================

    std::cout << "\n[CLIENT] Step 4: Starting data transfer..." << std::endl;

    // Get test parameters from environment
    size_t test_size_per_gpu = getEnvSize("UCCL_TCPX_PERF_SIZE", kDefaultTransferSize);
    int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 20);
    size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES", 524288);  // 512KB default
    const int kTransferTag = 42;

    // Validate test_size fits in registered buffer
    if (test_size_per_gpu > kMaxTransferSize) {
      std::cerr << "[ERROR] UCCL_TCPX_PERF_SIZE (" << test_size_per_gpu
                << ") exceeds max buffer size (" << kMaxTransferSize << ")" << std::endl;
      return 1;
    }

    // Calculate actual total channels (handle potential mismatches)
    int total_channels = 0;
    for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
      total_channels += gpus[gpu_id].mgr->get_num_channels();
    }

    std::cout << "[CLIENT] Test size per GPU: " << test_size_per_gpu << " bytes ("
              << (test_size_per_gpu / (1024 * 1024)) << " MB)" << std::endl;
    std::cout << "[CLIENT] Total test size: " << (test_size_per_gpu * kNumGPUs) << " bytes ("
              << (test_size_per_gpu * kNumGPUs / (1024 * 1024)) << " MB)" << std::endl;
    std::cout << "[CLIENT] Iterations: " << iterations << std::endl;
    std::cout << "[CLIENT] Chunk size: " << chunk_bytes << " bytes" << std::endl;
    std::cout << "[CLIENT] Total channels: " << total_channels << std::endl;

    // Wait for server to be ready and post receives
    std::cout << "\n[CLIENT] Waiting 10 seconds for server to post receives..." << std::endl;
    sleep(10);

    double total_time_ms = 0.0;

    // Track all pending requests
    struct PendingSend {
      void* request;
      int gpu_id;
      int channel_id;
      int chunk_idx;
    };
    std::vector<PendingSend> pending_sends;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "\n[CLIENT] ===== Iteration " << iter << " =====" << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      pending_sends.clear();

      // Phase 1: Post all sends (async)
      int global_chunk_idx = 0;
      for (int gpu_id = 0; gpu_id < kNumGPUs; gpu_id++) {
        GPUContext& ctx = gpus[gpu_id];
        int num_channels = ctx.mgr->get_num_channels();

        // Defensive: skip GPU if no channels (NIC probe failure, etc.)
        if (num_channels == 0) {
          std::cerr << "[WARNING] GPU " << gpu_id << " has 0 channels, skipping" << std::endl;
          continue;
        }

        size_t offset = 0;
        int local_chunk_idx = 0;

        while (offset < test_size_per_gpu) {
          size_t this_chunk = std::min(chunk_bytes, test_size_per_gpu - offset);

          // Round-robin across channels within this GPU
          int channel_local_id = local_chunk_idx % num_channels;
          ChannelResources& ch = ctx.mgr->get_channel(channel_local_id);

          // Calculate source pointer within this GPU's buffer
          void* src_ptr = reinterpret_cast<void*>(
              reinterpret_cast<uintptr_t>(ctx.gpu_buf) + offset);

          // Unique tag for this chunk (must match server)
          int tag = kTransferTag + iter * 100000 + gpu_id * 10000 + local_chunk_idx;

          // Send the chunk (async)
          void* send_request = nullptr;
          if (tcpx_isend(ch.send_comm, src_ptr, static_cast<int>(this_chunk),
                         tag, ch.mhandle, &send_request) != 0) {
            std::cerr << "[ERROR] tcpx_isend failed (GPU " << gpu_id
                      << " channel " << channel_local_id << " chunk " << local_chunk_idx << ")" << std::endl;
            return 1;
          }

          // Track this request
          pending_sends.push_back({send_request, gpu_id, channel_local_id, local_chunk_idx});

          offset += this_chunk;
          local_chunk_idx++;
          global_chunk_idx++;
        }
      }

      std::cout << "[CLIENT] Posted " << pending_sends.size() << " async sends" << std::endl;

      // Phase 2: Wait for all sends to complete
      for (auto& pending : pending_sends) {
        int done = 0, sent_size = 0;
        while (!done) {
          tcpx_test(pending.request, &done, &sent_size);
          if (!done) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
        }
      }

      auto end = std::chrono::high_resolution_clock::now();
      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += elapsed_ms;

      // Calculate bandwidth: total bytes transferred / time
      size_t total_bytes = test_size_per_gpu * kNumGPUs;
      double bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
      std::cout << "[CLIENT] Iteration " << iter << " completed in " << elapsed_ms
                << " ms, bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    }

    double avg_time_ms = total_time_ms / iterations;
    size_t total_bytes = test_size_per_gpu * kNumGPUs;
    double avg_bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);

    std::cout << "\n[CLIENT] ===== Performance Summary =====" << std::endl;
    std::cout << "[CLIENT] Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "[CLIENT] Average bandwidth: " << avg_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "[CLIENT] Total channels used: " << kNumGPUs * num_channels_per_gpu << std::endl;

    // ========================================
    // Cleanup
    // ========================================

    for (auto& ctx : gpus) {
      ctx.mgr->deregister_memory(false);
      ctx.mgr->close_all(false);
    }
  }

  // ========================================
  // Test Complete
  // ========================================

  std::cout << "\n[INFO] Test completed successfully" << std::endl;
  std::cout << "Next: Add actual data transfer and performance measurement (Step 3)" << std::endl;

  return 0;
}

