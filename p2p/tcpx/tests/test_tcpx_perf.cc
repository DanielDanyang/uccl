/**
 * @file test_tcpx_perf.cc
 * @brief TCPX GPU-to-GPU performance benchmark
 *
 * Based on test_tcpx_transfer.cc - uses SAME logic with iterations and timing.
 *
 * Usage:
 *   UCCL_TCPX_PERF_SIZE=4194304 ./tests/test_tcpx_perf server 0
 *   UCCL_TCPX_PERF_SIZE=4194304 ./tests/test_tcpx_perf client <server_ip> 0
 */

#include "../include/tcpx_interface.h"
#include "../include/tcpx_structs.h"
#include "../include/rx_descriptor.h"
#include "../device/unpack_launch.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <algorithm>
#include <vector>


namespace {
constexpr size_t kHandleBytes = 128;
struct ncclNetHandle_v7 { char data[kHandleBytes]; };
constexpr int kBootstrapPort = 12347;
constexpr int kTransferTag = 99;
constexpr size_t kMaxSize = 256 * 1024 * 1024;
constexpr size_t kRegisteredBytes = kMaxSize + 4096;

int getEnvInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

size_t getEnvSize(const char* name, size_t def) {
  const char* v = std::getenv(name);
  return v ? static_cast<size_t>(std::atoll(v)) : def;
}

// SAME as test_tcpx_transfer - returns client_fd (already accepted)
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) return -1;
  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(kBootstrapPort);
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(listen_fd);
    return -1;
  }
  if (listen(listen_fd, 1) < 0) {
    close(listen_fd);
    return -1;
  }
  // IMPORTANT: accept here and return client_fd
  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);
  return client_fd;
}

int connect_bootstrap(const char* ip) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(ip, &addr.sin_addr);
  for (int i = 0; i < 30; ++i) {
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  close(fd);
  return -1;
}
}  // namespace

int main(int argc, char** argv) {
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);
  // Enable wrapper debug logs unless user overrides
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  if (!std::getenv("UCCL_TCPX_LAUNCH_DEBUG")) setenv("UCCL_TCPX_LAUNCH_DEBUG", "0", 0);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <server|client> <gpu_id|server_ip> [gpu_id]" << std::endl;
    return 1;
  }

  bool is_server = (std::string(argv[1]) == "server");
  int gpu_id = 0;
  std::string server_ip;

  if (is_server) {
    gpu_id = std::atoi(argv[2]);
  } else {
    server_ip = argv[2];
    gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  }

  size_t test_size = getEnvSize("UCCL_TCPX_PERF_SIZE", 4 * 1024 * 1024);
  int iterations = getEnvInt("UCCL_TCPX_PERF_ITERS", 10);
  // Honor NCCL-style chunk size to mirror NCCL tests (fallback to 512KB)
  size_t chunk_bytes = getEnvSize("UCCL_TCPX_CHUNK_BYTES", getEnvSize("NCCL_P2P_NET_CHUNKSIZE", 512 * 1024));
  if (test_size > kMaxSize) test_size = kMaxSize;

  std::cout << "[PERF] Mode: " << (is_server ? "SERVER" : "CLIENT") << std::endl;
  std::cout << "[PERF] GPU: " << gpu_id << std::endl;
  std::cout << "[PERF] Size: " << (test_size / 1024 / 1024) << " MB" << std::endl;
  std::cout << "[PERF] Iterations: " << iterations << std::endl;

  int ndev = tcpx_get_device_count();
  if (ndev <= 0 || gpu_id >= ndev) {
    std::cerr << "[ERROR] Invalid GPU" << std::endl;
    return 1;
  }

  if (is_server) {
    ncclNetHandle_v7 handle{};
    void* listen_comm = nullptr;
    if (tcpx_listen(gpu_id, &handle, &listen_comm) != 0) {
      std::cerr << "[ERROR] tcpx_listen failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Waiting for client..." << std::endl;

    // IMPORTANT: create_bootstrap_server already does accept() internally
    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cerr << "[ERROR] bootstrap server failed" << std::endl;
      return 1;
    }

    std::cout << "[PERF] Bootstrap connection established, sending handle" << std::endl;

    // Send handle to client (SAME as transfer - with loop)
    size_t total_sent = 0;
    while (total_sent < kHandleBytes) {
      ssize_t sent = send(bootstrap_fd, handle.data + total_sent, kHandleBytes - total_sent, 0);
      if (sent <= 0) {
        std::cerr << "[ERROR] Failed to send handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_sent += static_cast<size_t>(sent);
    }

    // Accept TCPX connection (SAME as transfer - with retry)
    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    int attempts = 0;
    constexpr int kMaxRetries = 100;
    while (attempts < kMaxRetries) {
      int rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cerr << "[ERROR] tcpx_accept_v5 returned rc=" << rc << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      if (recv_comm) break;
      ++attempts;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cerr << "[ERROR] Failed to obtain recv_comm after retries" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // Select unpack implementation (kernel | d2d | host)
    const char* impl_env = std::getenv("UCCL_TCPX_UNPACK_IMPL");
    std::string impl = impl_env ? std::string(impl_env) : std::string("kernel");
    std::transform(impl.begin(), impl.end(), impl.begin(), ::tolower);
    std::cout << "[PERF] Unpack impl: " << impl << std::endl;


    std::cout << "[PERF] TCPX connection established" << std::endl;

    // Host-recv debug switch (align with transfer test)
    bool use_host_recv = false;
    if (const char* env = std::getenv("UCCL_TCPX_HOST_RECV_DEBUG")) {
      use_host_recv = (std::string(env) != "0");
    }
    std::cout << "[PERF] Host recv debug mode: " << (use_host_recv ? "ON" : "OFF") << std::endl;

    // CUDA init/context (keep consistent with transfer)
    CUdevice cuDev; CUcontext cuCtx;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS) {
      std::cerr << "[ERROR] CUDA initialization failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    if (cudaSetDevice(gpu_id) != cudaSuccess) {
      std::cerr << "[ERROR] cudaSetDevice failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // Choose receive buffer according to mode
    CUdeviceptr d_base = 0, d_aligned = 0;
    void* h_recv_base = nullptr;
    void* recv_buf = nullptr;
    int recv_ptr_type = NCCL_PTR_CUDA;

    if (!use_host_recv) {
      if (cuMemAlloc(&d_base, kRegisteredBytes + 4096) != CUDA_SUCCESS) {
        std::cerr << "[ERROR] CUDA allocation failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      // Align to 4KB
      uintptr_t addr = static_cast<uintptr_t>(d_base);
      addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
      d_aligned = static_cast<CUdeviceptr>(addr);
      recv_buf = reinterpret_cast<void*>(d_aligned);
      recv_ptr_type = NCCL_PTR_CUDA;
    } else {
      if (cudaMallocHost(&h_recv_base, kRegisteredBytes) != cudaSuccess) {
        std::cerr << "[ERROR] cudaMallocHost failed" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      recv_buf = h_recv_base;
      recv_ptr_type = NCCL_PTR_HOST;
    }

    void* recv_mhandle = nullptr;
    std::cout << "[PERF][SERVER] Registering recv buffer: ptr=" << recv_buf
              << " size=" << kRegisteredBytes
              << " type=" << (recv_ptr_type == NCCL_PTR_CUDA ? "NCCL_PTR_CUDA" : "NCCL_PTR_HOST") << std::endl;
    if (tcpx_reg_mr(recv_comm, recv_buf, kRegisteredBytes, recv_ptr_type, &recv_mhandle) != 0) {
      std::cerr << "[ERROR] tcpx_reg_mr failed" << std::endl;
      if (h_recv_base) cudaFreeHost(h_recv_base);
      if (d_base) cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[PERF][SERVER] Recv buffer registered successfully, mhandle=" << recv_mhandle << std::endl;

    // Create stream and launcher once (outside the loop) for kernel mode
    cudaStream_t unpack_stream = nullptr;
    tcpx::device::UnpackLauncher* launcher_ptr = nullptr;
    std::vector<void*> pending_reqs;  // defer irecv_consumed for kernel mode until stream sync

    if (!use_host_recv && impl == "kernel") {
      if (cudaStreamCreate(&unpack_stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to create unpack stream" << std::endl;
        if (h_recv_base) cudaFreeHost(h_recv_base);
        if (d_base) cuMemFree(d_base);
        tcpx_dereg_mr(recv_comm, recv_mhandle);
        tcpx_close_recv(recv_comm);
        tcpx_close_listen(listen_comm);
        close(bootstrap_fd);
        return 1;
      }
      tcpx::device::UnpackLaunchConfig cfg;
      cfg.stream = unpack_stream;
      cfg.enable_profiling = false;
      cfg.use_small_kernel = true;
      launcher_ptr = new tcpx::device::UnpackLauncher(cfg);
      std::cout << "[PERF][SERVER] Created persistent stream and launcher for kernel mode" << std::endl;
    }

    double total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;

      auto start = std::chrono::high_resolution_clock::now();

      size_t offset = 0;
      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* dst_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(recv_buf) + offset);
        const size_t chunk_idx = offset / chunk_bytes;
        const int tag = kTransferTag + static_cast<int>(iter) * 10000 + static_cast<int>(chunk_idx);
        std::cout << "[PERF][SERVER] chunk_idx=" << chunk_idx << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset << std::endl;

        void* recv_data[1] = {dst_ptr};
        int recv_sizes[1] = {static_cast<int>(this_chunk)};
        int recv_tags[1] = {tag};
        void* recv_mhandles[1] = {recv_mhandle};
        void* recv_request = nullptr;

        if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request) != 0) {
          std::cerr << "[ERROR] tcpx_irecv failed (chunk)" << std::endl;
          break;
        }

        int done = 0, received_size = 0;
        for (int poll = 0; poll < 1000000 && !done; ++poll) {
          tcpx_test(recv_request, &done, &received_size);
          if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        if (!done) {
          std::cerr << "[ERROR] Receive timeout at iteration " << iter << " offset=" << offset << std::endl;
          break;
        }

        if (use_host_recv) {
          // In host-recv debug mode, data is already in host buffer; skip unpack.
          std::cout << "[PERF][SERVER] host-recv completed size=" << received_size
                    << " (skip unpack)" << std::endl;
          offset += this_chunk;
        } else {
          auto* rx_req = reinterpret_cast<tcpx::plugin::tcpxRequest*>(recv_request);
          auto* dev_handle_struct = reinterpret_cast<tcpx::plugin::NcclNetDeviceHandle*>(recv_dev_handle);
          if (!rx_req || !dev_handle_struct || !rx_req->unpack_slot.mem || !rx_req->unpack_slot.cnt) {
            std::cerr << "[ERROR] Missing TCPX metadata for unpack" << std::endl;
            break;
          }
          uint64_t frag_count = *(rx_req->unpack_slot.cnt);
          std::cout << "[PERF][SERVER] frag_count=" << frag_count << std::endl;

          if (frag_count == 0 || frag_count > MAX_UNPACK_DESCRIPTORS) {
            std::cerr << "[ERROR] Invalid fragment count: " << frag_count << std::endl;
            break;
          }

          tcpx::plugin::unpackNetDeviceHandle dev_handle{};
          if (cuMemcpyDtoH(&dev_handle, reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
                           sizeof(dev_handle)) != CUDA_SUCCESS) {
            std::cerr << "[ERROR] Failed to read device handle" << std::endl;
            break;
          }

          auto* meta_entries = static_cast<tcpx::plugin::loadMeta*>(rx_req->unpack_slot.mem);
          tcpx::rx::UnpackDescriptorBlock desc_block;
          tcpx::rx::buildDescriptorBlock(meta_entries, static_cast<uint32_t>(frag_count),
                                         dev_handle.bounce_buf, dst_ptr, desc_block);
          desc_block.ready_flag = rx_req->unpack_slot.cnt;
          desc_block.ready_threshold = frag_count;

          int lrc = 0;
          if (impl == "kernel") {
            // Use persistent launcher with async launch (no sync here!)
            lrc = launcher_ptr->launch(desc_block);
            if (lrc != 0) {
              std::cerr << "[ERROR] Unpack kernel launch failed: " << lrc << std::endl;
              break;
            }
          } else if (impl == "d2d") {
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              const auto& meta = desc_block.descriptors[i];
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) + meta.src_off);
              CUdeviceptr dst_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.dst_buffer) + meta.dst_off);
              if (cuMemcpyDtoD(dst_ptr, src_ptr, meta.len) != CUDA_SUCCESS) {
                std::cerr << "[ERROR] D2D copy failed at descriptor " << i << std::endl;
                lrc = -1;
                break;
              }
            }
            if (lrc != 0) break;
          } else { // host gather
            std::vector<unsigned char> tmp(desc_block.total_bytes);
            size_t off = 0;
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              const auto& meta = desc_block.descriptors[i];
              CUdeviceptr src_ptr = static_cast<CUdeviceptr>(
                  reinterpret_cast<uintptr_t>(desc_block.bounce_buffer) + meta.src_off);
              if (cuMemcpyDtoH(tmp.data() + off, src_ptr, meta.len) != CUDA_SUCCESS) {
                std::cerr << "[ERROR] Host gather DtoH failed at descriptor " << i << std::endl;
                lrc = -1;
                break;
              }
              off += meta.len;
            }
            if (lrc != 0) break;
            if (cuMemcpyHtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(desc_block.dst_buffer)),
                             tmp.data(), tmp.size()) != CUDA_SUCCESS) {
              std::cerr << "[ERROR] Host gather HtoD failed" << std::endl;
              break;
            }
          }

          // Defer consume for kernel (async) until stream sync; immediate for others
          if (impl == "kernel") {
            pending_reqs.push_back(recv_request);
          } else {
            tcpx_irecv_consumed(recv_comm, 1, recv_request);
          }
          offset += this_chunk;
        }
      }

      // Sync stream once per iteration (after all chunks) for kernel mode
      if (!use_host_recv && impl == "kernel") {
        cudaError_t err = cudaStreamSynchronize(unpack_stream);
        if (err != cudaSuccess) {
          std::cerr << "[ERROR] cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
          break;
        }
        // Now it is safe to release bounce buffers for all kernel chunks
        for (void* req_ptr : pending_reqs) {
          tcpx_irecv_consumed(recv_comm, 1, req_ptr);
        }
        pending_reqs.clear();
      }

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += iter_time_ms;
      std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms" << std::endl;
    }

    double avg_ms = total_time_ms / iterations;
    double bw_gbps = (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << "[PERF] Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
              << "BW: " << std::fixed << std::setprecision(2) << bw_gbps << " GB/s" << std::endl;

    // Cleanup persistent launcher and stream
    if (launcher_ptr) {
      delete launcher_ptr;
      launcher_ptr = nullptr;
    }
    if (unpack_stream) {
      cudaStreamDestroy(unpack_stream);
      unpack_stream = nullptr;
    }

    tcpx_dereg_mr(recv_comm, recv_mhandle);
    if (h_recv_base) cudaFreeHost(h_recv_base);
    if (d_base) cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);
    close(bootstrap_fd);

  } else {
    int bootstrap_fd = connect_bootstrap(server_ip.c_str());
    if (bootstrap_fd < 0) {
      std::cerr << "[ERROR] bootstrap connect failed" << std::endl;
      return 1;
    }

    // Receive handle from server (SAME as transfer - with loop)
    ncclNetHandle_v7 handle{};
    size_t total_received = 0;
    while (total_received < kHandleBytes) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received, kHandleBytes - total_received, 0);
      if (r <= 0) {
        std::cerr << "[ERROR] Failed to receive handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }

    void* send_comm = nullptr;
    alignas(16) unsigned char send_dev_handle_storage[512] = {0};
    void* send_dev_handle = send_dev_handle_storage;
    if (tcpx_connect_v5(gpu_id, &handle, &send_comm, &send_dev_handle) != 0 || !send_comm) {
      std::cerr << "[ERROR] tcpx_connect_v5 failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    std::cout << "[PERF] TCPX connection established" << std::endl;

    // Allocate GPU buffer (SAME as test_tcpx_transfer)
    CUdevice cuDev;
    CUcontext cuCtx;
    CUdeviceptr d_base, d_aligned;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS ||
        cuMemAlloc(&d_base, kRegisteredBytes + 4096) != CUDA_SUCCESS) {
      std::cerr << "[ERROR] CUDA initialization or allocation failed" << std::endl;
      close(bootstrap_fd);
      return 1;
    }

    // IMPORTANT: cudaSetDevice (SAME as transfer)
    if (cudaSetDevice(gpu_id) != cudaSuccess) {
      std::cerr << "[ERROR] cudaSetDevice failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }

    // Align to 4KB
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* send_buf = reinterpret_cast<void*>(d_aligned);

    void* send_mhandle = nullptr;
    std::cout << "[PERF][CLIENT] Registering send buffer: ptr=" << send_buf
              << " size=" << kRegisteredBytes << " type=NCCL_PTR_CUDA" << std::endl;
    if (tcpx_reg_mr(send_comm, send_buf, kRegisteredBytes, NCCL_PTR_CUDA, &send_mhandle) != 0) {
      std::cerr << "[ERROR] tcpx_reg_mr failed" << std::endl;
      cuMemFree(d_base);
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "[PERF][CLIENT] Send buffer registered successfully, mhandle=" << send_mhandle << std::endl;

    double total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
      std::cout << "[PERF] Iteration " << iter << ": total bytes=" << test_size
                << ", chunk_bytes=" << chunk_bytes << std::endl;
      auto start = std::chrono::high_resolution_clock::now();

      size_t offset = 0;
      while (offset < test_size) {
        const size_t this_chunk = std::min(chunk_bytes, test_size - offset);
        void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(send_buf) + offset);
        const size_t chunk_idx = offset / chunk_bytes;
        const int tag = kTransferTag + static_cast<int>(iter) * 10000 + static_cast<int>(chunk_idx);
        std::cout << "[PERF][CLIENT] chunk_idx=" << chunk_idx << " tag=" << tag
                  << " size=" << this_chunk << " offset=" << offset << std::endl;
        void* send_request = nullptr;
        if (tcpx_isend(send_comm, src_ptr, static_cast<int>(this_chunk), tag, send_mhandle, &send_request) != 0) {
          std::cerr << "[ERROR] tcpx_isend failed (chunk)" << std::endl;
          break;
        }
        int done = 0, sent_size = 0;
        for (int poll = 0; poll < 1000000 && !done; ++poll) {
          tcpx_test(send_request, &done, &sent_size);
          if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        if (!done) {
          std::cerr << "[ERROR] Send timeout at iteration " << iter << " offset=" << offset << std::endl;
          break;
        }
        offset += this_chunk;
      }

      auto end = std::chrono::high_resolution_clock::now();
      double iter_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      total_time_ms += iter_time_ms;
      std::cout << "[PERF] Iter " << iter << " time=" << iter_time_ms << "ms" << std::endl;
    }

    double avg_ms = total_time_ms / iterations;
    double bw_gbps = (test_size / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    std::cout << "[PERF] Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
              << "BW: " << std::fixed << std::setprecision(2) << bw_gbps << " GB/s" << std::endl;

    tcpx_dereg_mr(send_comm, send_mhandle);
    cuMemFree(d_base);
    cuDevicePrimaryCtxRelease(cuDev);
    tcpx_close_send(send_comm);
    close(bootstrap_fd);
  }

  return 0;
}

