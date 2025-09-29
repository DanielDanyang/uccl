/**
 * @file test_tcpx_transfer_clean.cc
 * @brief TCPX GPU-to-GPU end-to-end transfer validation
 *
 * This test uses nccl-plugin-gpudirecttcpx APIs to transfer data between CUDA device memory
 * on two nodes. It builds upon test_connection.cc (handshake only) by adding CUDA buffer
 * registration and data validation.
 *
 * Server steps:
 *   1. Listen for TCPX connections on device 0, publish NCCL handle via bootstrap TCP socket.
 *   2. Accept TCPX connection, register 4KB CUDA buffer, submit async receive request.
 *   3. Poll for completion, copy data back to host memory and validate payload content.
 *
 * Client steps:
 *   1. Get NCCL handle via bootstrap TCP socket.
 *   2. Connect to server via TCPX, register 4KB CUDA buffer, write test payload and send.
 *   3. Wait for completion and cleanup resources.
 */

#include "../tcpx_interface.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../device/unpack_launch.h"
#include "../rx/rx_descriptor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kHandleBytes = 128;
struct ncclNetHandle_v7 {
  char data[kHandleBytes];
};

constexpr int kBootstrapPort = 12345;
constexpr size_t kRegisteredBytes = 4096;      // 4 KB aligned allocation.

// PROBLEM BACKGROUND MARKER:
// Previously using sizeof(kTestMessage) directly as payload size would include trailing '\0',
// causing client to think 25B while logs/transport layer only handled 24B visible chars,
// creating 25 vs 24B inconsistency, further triggering small packet zero-copy (MSG_ZEROCOPY)
// kernel errqueue flakiness, server only seeing 16B control message, no payload received.
// Fix: Use strlen semantics (sizeof-1), force <4KB to copy path at runtime to avoid small packet zero-copy.

constexpr char kTestMessage[] = "Hello from TCPX client!";
// Default payload length (visible chars only). Can be overridden by env
// UCCL_TCPX_PAYLOAD_BYTES up to kRegisteredBytes.
constexpr size_t kDefaultPayloadBytes = sizeof(kTestMessage) - 1;
constexpr int kTransferTag = 42;  // Payload tag
constexpr int kAcceptMaxRetries = 120;         // ~12 s (100 ms per retry).

// Minimal replicas of plugin-internal structures we need for device unpack.
struct NcclNetDeviceHandle {
  int netDeviceType;
  int netDeviceVersion;
  void* handle;
  size_t size;
  int needsProxyProgress;
};

struct DevmemToken {
  uint32_t token_start;
  uint32_t token_count;
};

struct TcpxUnpackSlot {
  bool active;
  uint64_t idx;
  void* mem;               // Array of loadMeta entries
  uint64_t* cnt;           // Number of valid entries written by transport
  uint64_t cnt_cache;
  size_t* fds_cnt;
  size_t* pgtok_cnts;
  int* fds;
  DevmemToken* pgtoks;
};

struct TcpxRequest {
  void* comm;
  void* data;
  int op;
  int mem_type;
  int next_sock_id;
  int next_size;
  int offset;
  int size;
  int size_pending;
  int gpu_mem_fd;
  int gpu_mem_off;
  TcpxUnpackSlot unpack_slot;
};

struct UnpackNetDeviceHandle {
  void* meta;
  void* bounce_buf;
  uint64_t head;
};

struct LoadMetaEntry {
  uint32_t src_off;
  uint32_t len;
  uint64_t dst_off;
};
static_assert(sizeof(LoadMetaEntry) == 16, "loadMeta layout must match plugin");

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
  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);
  return client_fd;
}

int connect_to_bootstrap_server(const char* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(server_ip, &addr.sin_addr);

  for (int retry = 0; retry < 10; ++retry) {
    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return sock_fd;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  close(sock_fd);
  return -1;
}

void dump_hex(const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t limit = std::min<size_t>(bytes, 32);
  for (size_t i = 0; i < limit; ++i) {
    if (i && i % 16 == 0) std::cout << "\n";
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(p[i]) << ' ';
  }
  std::cout << std::dec << std::endl;
}

bool cuda_check(CUresult res, const char* what) {
  if (res == CUDA_SUCCESS) return true;

  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &desc);

  std::cout << "[DEBUG] CUDA error at " << what << ": "
            << (name ? name : "?") << " - " << (desc ? desc : "")
            << std::endl;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  // Debug/stability knobs: avoid zero-copy for tiny payloads to reduce errqueue flakiness
  setenv("UCCL_TCPX_DEBUG", "1", 0);
  setenv("NCCL_MIN_ZCOPY_SIZE", "4096", 0);  // Force <4KB to copy path
  setenv("NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE", "4096", 0);  // Plugin-specific backup
  // Optional: make receive path more deterministic during debug
  setenv("NCCL_GPUDIRECTTCPX_RECV_SYNC", "1", 0);
  // Device-unpack path is the default. Set UCCL_TCPX_HOST_RECV_DEBUG=1 to force host fallback.

  std::cout << "[DEBUG] === TCPX GPU-to-GPU transfer test ===" << std::endl;
  if (argc < 2) {
    std::cout << "[DEBUG] Usage: " << argv[0] << " <server|client> [remote_ip]" << std::endl;
    return 1;
  }
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "[DEBUG] ERROR: no TCPX devices detected" << std::endl;
    return 1;
  }
  int dev_id = 0;
  bool is_server = std::strcmp(argv[1], "server") == 0;

  // Resolve payload size (can be overridden via env UCCL_TCPX_PAYLOAD_BYTES)
  size_t payload_bytes = kDefaultPayloadBytes;
  if (const char* p = std::getenv("UCCL_TCPX_PAYLOAD_BYTES")) {
    char* endp = nullptr;
    unsigned long v = std::strtoul(p, &endp, 10);
    if (endp && *endp == '\0' && v > 0) {
      if (v > kRegisteredBytes) v = kRegisteredBytes;
      payload_bytes = static_cast<size_t>(v);
    }
  }
  std::cout << "[DEBUG] Using payload_bytes=" << payload_bytes << std::endl;

  if (is_server) {
    std::cout << "[DEBUG] Running in SERVER mode" << std::endl;

    ncclNetHandle_v7 handle{};
    void* listen_comm = nullptr;
    if (tcpx_listen(dev_id, &handle, &listen_comm) != 0) {
      std::cout << "[DEBUG] ERROR: tcpx_listen failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] Listening on device " << dev_id << std::endl;

    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "[DEBUG] ERROR: bootstrap server creation failed" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "[DEBUG] Bootstrap connection established, sending handle" << std::endl;

    size_t total_sent = 0;
    while (total_sent < kHandleBytes) {
      ssize_t sent = send(bootstrap_fd, handle.data + total_sent,
                          kHandleBytes - total_sent, 0);
      if (sent <= 0) {
        std::cout << "[DEBUG] ERROR: failed to send NCCL handle" << std::endl;
        close(bootstrap_fd);
        tcpx_close_listen(listen_comm);
        return 1;
      }
      total_sent += static_cast<size_t>(sent);
    }
    // Keep bootstrap_fd open to optionally send a 1-byte ACK after payload verification

    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    int attempts = 0;
    while (attempts < kAcceptMaxRetries) {
      int rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cout << "[DEBUG] ERROR: tcpx_accept_v5 returned error rc=" << rc << std::endl;
        tcpx_close_listen(listen_comm);
        return 1;
      }
      if (recv_comm) break;  // Successfully established
      ++attempts;
      if (attempts % 10 == 0) {
        std::cout << "[DEBUG] INFO: tcpx_accept_v5 rc=0 but recv_comm still null (attempt "
                  << attempts << "), continuing to wait for peer handshake..." << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cout << "[DEBUG] ERROR: failed to get valid recv_comm after retries" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "[DEBUG] Connection accepted; recv_comm=" << recv_comm << std::endl;

    // ===== 【问题代码区�?：服务端GPU缓冲区分配与对齐�?=====
    // 这里是TCPX传输问题的第一个关键环节：GPU内存分配�?KB对齐
    // 历史问题：如果GPU内存没有正确对齐�?KB边界，tcpx_reg_mr会失�?    // 或者即使注册成功，GPUDirect TCPX的DMA传输也可能出现数据损�?    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;      // 原始分配的GPU内存地址（可能未对齐�?    CUdeviceptr d_aligned = 0;   // 4KB对齐后的GPU内存地址（用于TCPX注册�?
    // CUDA初始化和GPU内存分配
    // 注意：这里分�?kRegisteredBytes + 4096 是为了确保有足够空间进行4KB对齐
    // 因为原始地址可能不在4KB边界上，需要向上调整到最近的4KB边界
    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      if (d_base) cuMemFree(d_base);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      return 1;
    }
    if (cudaSetDevice(dev_id) != cudaSuccess) {
      std::cout << "[DEBUG] ERROR: cudaSetDevice failed" << std::endl;
      if (use_host_recv && recv_buf) cudaFreeHost(recv_buf);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    // Align GPU buffer to the nearest 4KB boundary for GPUDirect TCPX.
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);  // 4KB aligned
              << std::dec << " (原始地址: 0x" << std::hex << d_base << std::dec << ")" << std::endl;

    // Allow host receive debug path via env to isolate GPUDirect issues
    const char* host_recv_dbg = std::getenv("UCCL_TCPX_HOST_RECV_DEBUG");
    bool use_host_recv = false;
    if (host_recv_dbg) {
      char c = host_recv_dbg[0];
      if (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y') {
        use_host_recv = true;
      }
    }
    void* recv_mhandle = nullptr;
    void* recv_buf = nullptr;
    int recv_ptr_type = NCCL_PTR_CUDA;
    void* recv_mhandle = nullptr;
    void* recv_buf = nullptr;
    int recv_ptr_type = NCCL_PTR_CUDA;
    if (use_host_recv) {
      void* host_aligned = nullptr;
      if (cudaMallocHost(&host_aligned, kRegisteredBytes) != cudaSuccess || !host_aligned) {
        std::cout << "[DEBUG] ERROR: cudaMallocHost failed for host recv" << std::endl;
        cuMemFree(d_base);
        tcpx_close_recv(recv_comm);
        tcpx_close_listen(listen_comm);
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
      }
      std::memset(host_aligned, 0, kRegisteredBytes);
      recv_buf = host_aligned;
      recv_ptr_type = NCCL_PTR_HOST;
      std::cout << "[DEBUG] Host-recv fallback enabled; host buffer=" << recv_buf << std::endl;
    } else {
      recv_buf = reinterpret_cast<void*>(d_aligned);
      recv_ptr_type = NCCL_PTR_CUDA;
    }

    // ===== �����������??��������ڴ�ע������������ύ??=====
    // �����ǵڶ����ؼ�����㣺��GPU�ڴ�ע�ᵽTCPX���ύ�첽������??    // ��ʷ����������??    // 1. ����ڴ�ע��ʧ�ܣ�ͨ������ΪGPU�ڴ�??KB�����gpumemd����δ��??    // 2. ���tcpx_irecvʧ�ܣ�������recv_comm�����Ч���ڴ���������

    // ===== 【问题代码区�?：服务端内存注册与接收请求提交�?=====
    // 这里是第二个关键问题点：将GPU内存注册到TCPX并提交异步接收请�?    // 历史问题根因分析�?    // 1. 如果内存注册失败，通常是因为GPU内存�?KB对齐或gpumemd服务未运�?    // 2. 如果tcpx_irecv失败，可能是recv_comm句柄无效或内存句柄有问题
    if (tcpx_reg_mr(recv_comm, recv_buf, kRegisteredBytes, recv_ptr_type, &recv_mhandle) != 0) {
      std::cout << "[DEBUG] 错误：服务端内存注册失败 (tcpx_reg_mr)" << std::endl;
      std::cout << "[DEBUG] 可能原因�?)GPU内存�?KB对齐 2)gpumemd服务未运�?3)DMABUF权限问题" << std::endl;
      if (use_host_recv && recv_buf) cudaFreeHost(recv_buf);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "[DEBUG] 服务端内存注册成�? recv_mhandle=" << recv_mhandle
              << ", 缓冲�?" << recv_buf << ", 大小=" << kRegisteredBytes << "字节" << std::endl;

    // 准备异步接收请求的参数数�?    // TCPX API使用数组形式支持批量操作，这里只有一个接收操�?    void* recv_data[1] = {recv_buf};                              // 接收缓冲区地址
    int recv_sizes[1] = {static_cast<int>(payload_bytes)};        // 期望接收的数据大�?    int recv_tags[1] = {kTransferTag};                            // 消息标签（用于匹配发送端�?    void* recv_mhandles[1] = {recv_mhandle};                      // 内存句柄
    void* recv_request = nullptr;                                 // 异步请求句柄（用于后续轮询）

    // 【关键步骤】提交异步接收请�?    // 这里是历史问题的核心：如果这一步成功但后续只收�?6B控制消息�?    // 说明客户端的发送有问题（通常是小包zero-copy路径异常或过早关闭连接）
    if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      std::cout << "[DEBUG] 错误：异步接收请求提交失�?(tcpx_irecv)" << std::endl;
      std::cout << "[DEBUG] 可能原因�?)recv_comm句柄无效 2)内存句柄问题 3)参数不匹�? << std::endl;
      tcpx_dereg_mr(recv_comm, recv_mhandle);
      if (use_host_recv && recv_buf) cudaFreeHost(recv_buf);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    // ===== 【问题代码区�?：服务端接收轮询与历史问题现场�?=====
    // 这里是历史问题的核心现场：服务端等待数据但只收到16B控制消息
    //
    // 【历史问题详细分析】：
    // 1. 现象：服务端tcpx_irecv提交成功，但tcpx_test轮询时只收到16B控制消息�?    //    随后客户端连接关闭，payload数据永远没有到达，最终超�?    // 2. 根本原因�?    //    a) 客户端使用sizeof(kTestMessage)=25B，但实际发�?4B，大小不一�?    //    b) 小包(<4KB)触发MSG_ZEROCOPY路径，内核errqueue处理异常
    //    c) 客户端发送完成后立即关闭连接，没有等待服务端确认
    // 3. 修复方案�?    //    a) 统一使用strlen语义(sizeof-1)，确保收发大小一�?    //    b) 强制<4KB走copy路径，避免小包zero-copy的errqueue问题
    //    c) 客户端发送后增加延迟，给服务端处理时�?
    std::cout << "[DEBUG] 开始等待客户端数据，期望大�?" << payload_bytes << "字节..." << std::endl;
    int done = 0;           // 完成标志�?=未完成，1=已完�?    int received_size = 0;  // 实际接收到的字节�?
    // 轮询接收完成状态，最多等待约2�?200000 * 10微秒)
    // 如果在这个循环中一直done=0，说明数据没有到达（历史问题现场�?    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(recv_request, &done, &received_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] 错误：tcpx_test返回错误�?" << rc_test << std::endl;
        std::cout << "[DEBUG] 这通常表示连接异常或请求句柄无�? << std::endl;
        break;
      }
      // �?0微秒检查一次，避免CPU占用过高
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));

      // �?000次迭�?�?0ms)打印一次进度，便于诊断卡住的位�?      if (i > 0 && i % 1000 == 0) {
        std::cout << "[DEBUG] 轮询进度: " << i << "/200000, done=" << done
                  << ", received_size=" << received_size << std::endl;
      }
    }

    std::vector<unsigned char> host(payload_bytes, 0);
    bool success = false;
    bool copy_ok = false;
    size_t bytes_copied = 0;

    if (!done) {
      std::cout << "[DEBUG] ERROR: receive timed out" << std::endl;
    } else if (use_host_recv) {
      std::memcpy(host.data(), recv_buf, payload_bytes);
      copy_ok = true;
      bytes_copied = payload_bytes;
    } else {
      auto* rx_req = reinterpret_cast<TcpxRequest*>(recv_request);
      auto* dev_handle_struct = reinterpret_cast<NcclNetDeviceHandle*>(recv_dev_handle);
      if (!rx_req || !rx_req->unpack_slot.mem || !rx_req->unpack_slot.cnt || !dev_handle_struct) {
        std::cout << "[DEBUG] ERROR: missing TCPX metadata for device unpack" << std::endl;
      } else {
        uint64_t frag_count = *(rx_req->unpack_slot.cnt);
        auto* meta_entries = static_cast<LoadMetaEntry*>(rx_req->unpack_slot.mem);
        if (frag_count == 0) {
          std::cout << "[DEBUG] ERROR: unpack metadata contains zero fragments" << std::endl;
        } else if (frag_count > tcpx::rx::MAX_UNPACK_DESCRIPTORS) {
          std::cout << "[DEBUG] ERROR: fragment count " << frag_count
                    << " exceeds descriptor capacity" << std::endl;
        } else {
          UnpackNetDeviceHandle dev_handle{};
          if (!cuda_check(cuMemcpyDtoH(&dev_handle,
                                       reinterpret_cast<CUdeviceptr>(dev_handle_struct->handle),
                                       sizeof(dev_handle)),
                          "cuMemcpyDtoH(device_handle)")) {
            std::cout << "[DEBUG] ERROR: failed to read device handle metadata" << std::endl;
          } else {
            tcpx::rx::UnpackDescriptorBlock desc_block;
            desc_block.count = static_cast<uint32_t>(frag_count);
            desc_block.total_bytes = 0;
            desc_block.bounce_buffer = dev_handle.bounce_buf;
            desc_block.dst_buffer = reinterpret_cast<void*>(d_aligned);
            for (uint32_t i = 0; i < desc_block.count; ++i) {
              desc_block.descriptors[i].src_off = meta_entries[i].src_off;
              desc_block.descriptors[i].len = meta_entries[i].len;
              desc_block.descriptors[i].dst_off = meta_entries[i].dst_off;
              desc_block.total_bytes += meta_entries[i].len;
            }

            tcpx::device::UnpackLaunchConfig unpack_cfg;
            tcpx::device::UnpackLauncher unpack_launcher(unpack_cfg);
            if (unpack_launcher.launchSync(desc_block) != 0) {
              std::cout << "[DEBUG] ERROR: device unpack kernel launch failed" << std::endl;
            } else if (!cuda_check(cuMemcpyDtoH(host.data(), d_aligned, payload_bytes),
                                    "cuMemcpyDtoH(dst)")) {
              std::cout << "[DEBUG] ERROR: failed to copy unpacked payload to host" << std::endl;
            } else {
              copy_ok = true;
              bytes_copied = desc_block.total_bytes;
            }
          }
        }
      }
    }

    if (copy_ok) {
      dump_hex(host.data(), std::min<size_t>(payload_bytes, 32));
      size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);
      bool prefix_ok = std::memcmp(host.data(), kTestMessage, prefix) == 0;
      bool tail_ok = (payload_bytes <= prefix) || host[payload_bytes - 1] == 0xAB;
      success = prefix_ok && tail_ok;
      std::cout << "[DEBUG] Receive completed, bytes=" << bytes_copied << std::endl;
      if (success) {
        std::cout << "[DEBUG] SUCCESS: payload matches expected string" << std::endl;
        if (bootstrap_fd >= 0) {
          char ack = 1;
          (void)send(bootstrap_fd, &ack, 1, 0);
        }
      } else {
        std::cout << "[DEBUG] ERROR: payload mismatch" << std::endl;
      }
    } else if (done) {
      std::cout << "[DEBUG] ERROR: device unpack failed" << std::endl;
    }

    if (recv_request && done) {
      tcpx_irecv_consumed(recv_comm, 1, recv_request);
    }

    if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
    if (use_host_recv && recv_buf) cudaFreeHost(recv_buf);
    cuMemFree(d_base);
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);
    if (bootstrap_fd >= 0) close(bootstrap_fd);
    cuDevicePrimaryCtxRelease(cuDev);
    return success ? 0 : 1;


  } else {

    size_t total_received = 0;
    while (total_received < kHandleBytes) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       kHandleBytes - total_received, 0);
      if (r <= 0) {
        std::cout << "[DEBUG] ERROR: failed to receive NCCL handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }
    // Keep bootstrap_fd open to wait for server's 1-byte ACK after send

    void* send_comm = nullptr;
    // Pre-allocate device handle storage (some implementations require caller to provide buffer)
    alignas(16) unsigned char send_dev_handle_storage[512] = {0};
    void* send_dev_handle = send_dev_handle_storage;
    if (tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle) != 0 ||
        !send_comm) {
      std::cout << "[DEBUG] ERROR: tcpx_connect_v5 connection failed" << std::endl;
      return 1;
    }
    std::cout << "[DEBUG] TCPX connection established; send_comm=" << send_comm
              << ", send_dev_handle=" << send_dev_handle << std::endl;

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;

    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      if (d_base) cuMemFree(d_base);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      tcpx_close_send(send_comm);
      return 1;
    }

    // Prepare payload in host memory according to payload_bytes

    // ===== 【问题代码区�?：客户端GPU缓冲区准备与内存注册�?=====
    // 这里是客户端侧的关键问题区域：GPU内存对齐、数据准备和内存注册

    // 【关键步�?】GPU内存4KB对齐（与服务端相同的要求�?    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);  // 4KB对齐
    d_aligned = static_cast<CUdeviceptr>(addr);

    // 【关键步�?】准备要发送的数据
    // 历史问题：之前直接使用sizeof(kTestMessage)=25B，包含了'\0'终止�?    // 但传输层实际只处�?4B可见字符，造成大小不一致，触发各种异常
    std::vector<unsigned char> host_payload(payload_bytes, 0);
    size_t prefix = std::min(payload_bytes, sizeof(kTestMessage) - 1);  // 只复制可见字�?    if (prefix) std::memcpy(host_payload.data(), kTestMessage, prefix);
    if (payload_bytes > prefix) host_payload[payload_bytes - 1] = 0xAB;  // 添加哨兵字节便于验证

    // 将数据从主机内存复制到GPU内存
    cuda_check(cuMemcpyHtoD(d_aligned, host_payload.data(), payload_bytes), "cuMemcpyHtoD");

    // 【重要】确保GPU内存写入完成，避免zero-copy发送时读取到未完成的数�?    cuCtxSynchronize();

    // 调试：发送前验证GPU缓冲区内容，确保数据正确
    {
      size_t dump = std::min<size_t>(payload_bytes, 32);
      std::vector<unsigned char> verify(dump, 0);
      if (cuda_check(cuMemcpyDtoH(verify.data(), d_aligned, dump), "pre-send cuMemcpyDtoH")) {
        std::cout << "[DEBUG] 客户端GPU缓冲区内容验�?(�? << dump << "字节):" << std::endl;
        dump_hex(verify.data(), dump);
      }
    }

    // ===== 【问题代码区�?：客户端内存注册�?=====
    // 这里是客户端内存注册，如果失败通常是GPU内存对齐或gpumemd问题
    void* send_mhandle = nullptr;
    if (tcpx_reg_mr(send_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &send_mhandle) != 0) {
      std::cout << "[DEBUG] 错误：客户端内存注册失败 (tcpx_reg_mr)" << std::endl;
      std::cout << "[DEBUG] 可能原因�?)GPU内存�?KB对齐 2)gpumemd服务未运�?3)send_comm句柄无效" << std::endl;
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "[DEBUG] 客户端内存注册成�? send_mhandle=" << send_mhandle << std::endl;

    // ===== 【问题代码区�?：客户端异步发送与历史问题根源�?=====
    // 这里是历史问题的根源：客户端发送逻辑和过早关闭连�?
    void* send_request = nullptr;
    // 【关键步骤】提交异步发送请�?    // 历史问题：这里成功提交，但由于小包zero-copy路径异常�?    // 实际只发送了16B控制消息，payload数据没有正确传输
    if (tcpx_isend(send_comm, reinterpret_cast<void*>(d_aligned),
                   static_cast<int>(payload_bytes), kTransferTag, send_mhandle,
                   &send_request) != 0) {
      std::cout << "[DEBUG] 错误：异步发送请求提交失�?(tcpx_isend)" << std::endl;
      std::cout << "[DEBUG] 可能原因�?)send_comm句柄无效 2)内存句柄问题 3)参数错误" << std::endl;
      tcpx_dereg_mr(send_comm, send_mhandle);
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "[DEBUG] 异步发送请求已提交，开始轮询完成状�?.." << std::endl;

    // 【关键步骤】轮询发送完成状�?    // 历史问题分析�?    // 1. 客户端这里可能返回done=1，但实际上只是控制消息发送完�?    // 2. 由于小包走了MSG_ZEROCOPY路径，errqueue处理异常，payload数据丢失
    // 3. 客户端误以为发送成功，立即关闭连接，导致服务端只收�?6B控制消息
    int done = 0;       // 发送完成标�?    int sent_size = 0;  // 实际发送的字节�?    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(send_request, &done, &sent_size);
      if (rc_test != 0) {
        std::cout << "[DEBUG] 错误：tcpx_test返回错误�?" << rc_test << std::endl;
        break;
      }
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));

      // �?000次迭代打印进�?      if (i > 0 && i % 1000 == 0) {
        std::cout << "[DEBUG] 发送轮询进�? " << i << "/200000, done=" << done
                  << ", sent_size=" << sent_size << std::endl;
      }
    }

    if (done) {
      std::cout << "[DEBUG] 发送完成，实际发送字节数=" << sent_size
                << " (期望=" << payload_bytes << ")" << std::endl;
      // 检查发送字节数是否与期望一�?      if (sent_size != static_cast<int>(payload_bytes)) {
        std::cout << "[DEBUG] 警告：发送字节数不匹配！这可能表示部分数据丢�? << std::endl;
      }
    } else {
      std::cout << "[DEBUG] 警告：发送在超时前未完成，可能存在网络或传输问题" << std::endl;
    }

    // ===== 【问题代码区�?：客户端等待服务端确认，避免过早关闭�?=====
    // 这里是修复历史问题的关键：通过bootstrap TCP连接等待服务端ACK
    // 历史问题：客户端发送完成后立即关闭TCPX连接，服务端来不及处理payload
    // 修复方案：复用bootstrap TCP连接，等待服务端发�?字节ACK确认收到数据
    std::cout << "[DEBUG] 等待服务端通过bootstrap连接发送ACK确认..." << std::endl;
    if (bootstrap_fd >= 0) {
      // 设置2秒接收超时，避免无限等待
      timeval tv{}; tv.tv_sec = 2; tv.tv_usec = 0;
      setsockopt(bootstrap_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      char ack = 0;
      ssize_t r = recv(bootstrap_fd, &ack, 1, 0);
      if (r == 1 && ack == 1) {
        std::cout << "[DEBUG] 已收到服务端ACK确认，数据传输成�? << std::endl;
      } else {
        std::cout << "[DEBUG] 警告：未收到服务端ACK，可能传输有问题 (recv返回=" << r << ", ack=" << (int)ack << ")" << std::endl;
      }
    } else {
      std::cout << "[DEBUG] 警告：bootstrap连接无效，无法等待服务端ACK" << std::endl;
    }

    tcpx_dereg_mr(send_comm, send_mhandle);
    cuMemFree(d_base);
    tcpx_close_send(send_comm);
    if (bootstrap_fd >= 0) close(bootstrap_fd);
    cuDevicePrimaryCtxRelease(cuDev);
    return done ? 0 : 1;
  }
}

