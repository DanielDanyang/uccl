#include "../tcpx_interface.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kBootstrapPort = 12345;
constexpr size_t kRegisteredBytes = 4096;  // align with GPUDirectTCPX expectations
constexpr char kTestMessage[] = "Hello from TCPX client!";
constexpr int kTransferTag = 42;

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
  std::cout << "CUDA error at " << what << ": "
            << (name ? name : "?") << " - " << (desc ? desc : "")
            << std::endl;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << "=== TCPX Device-to-Device Transfer Test ===" << std::endl;
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <server|client> [remote_ip]" << std::endl;
    return 1;
  }

  setenv("UCCL_TCPX_DEBUG", "1", 1);

  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "�?FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  int dev_id = 0;
  bool is_server = std::strcmp(argv[1], "server") == 0;

  if (is_server) {
    std::cout << "Starting in SERVER mode" << std::endl;
    ncclNetHandle_v7 handle{};
    void* listen_comm = nullptr;
    if (tcpx_listen(dev_id, &handle, &listen_comm) != 0) {
      std::cout << "�?FAILED: tcpx_listen" << std::endl;
      return 1;
    }

    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: bootstrap server" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    size_t total_sent = 0;
    while (total_sent < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t s = send(bootstrap_fd, handle.data + total_sent,
                       NCCL_NET_HANDLE_MAXSIZE - total_sent, 0);
      if (s <= 0) {
        std::cout << "�?FAILED: sending handle" << std::endl;
        close(bootstrap_fd);
        tcpx_close_listen(listen_comm);
        return 1;
      }
      total_sent += static_cast<size_t>(s);
    }
    close(bootstrap_fd);

    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512]{};
    void* recv_dev_handle = recv_dev_handle_storage;
    if (tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle) != 0 || !recv_comm) {
      std::cout << "�?FAILED: tcpx_accept_v5" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);

    void* recv_mhandle = nullptr;
    if (tcpx_reg_mr(recv_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &recv_mhandle) != 0) {
      std::cout << "�?FAILED: tcpx_reg_mr (recv)" << std::endl;
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    void* recv_data[1] = {reinterpret_cast<void*>(d_aligned)};
    int recv_sizes[1] = {static_cast<int>(kRegisteredBytes)};
    int recv_tags[1] = {kTransferTag};
    void* recv_mhandles[1] = {recv_mhandle};
    void* recv_request = nullptr;
    if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      std::cout << "�?FAILED: tcpx_irecv" << std::endl;
      tcpx_dereg_mr(recv_comm, recv_mhandle);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    int done = 0, received_size = 0;
    for (int i = 0; i < 200000 && !done; ++i) {
      if (tcpx_test(recv_request, &done, &received_size) != 0) break;
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    cuCtxSynchronize();

    std::vector<unsigned char> host(kRegisteredBytes, 0);
    cuda_check(cuMemcpyDtoH(host.data(), d_aligned, kRegisteredBytes), "cuMemcpyDtoH");

    std::cout << "Received payload (hex preview):" << std::endl;
    dump_hex(host.data(), kRegisteredBytes);

    bool match = std::memcmp(host.data(), kTestMessage, sizeof(kTestMessage)) == 0;
    if (match) {
      std::cout << "�?SUCCESS: Payload matches expected string" << std::endl;
    } else {
      std::cout << "�?FAILED: Payload mismatch" << std::endl;
    }

    if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
    cuMemFree(d_base);
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);
    cuDevicePrimaryCtxRelease(cuDev);
    return match ? 0 : 1;

  } else {
    if (argc < 3) {
      std::cout << "�?ERROR: client mode requires remote IP" << std::endl;
      return 1;
    }
    std::cout << "Starting in CLIENT mode, connecting to " << argv[2] << std::endl;

    int bootstrap_fd = connect_to_bootstrap_server(argv[2]);
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: bootstrap connect" << std::endl;
      return 1;
    }
    ncclNetHandle_v7 handle{};
    size_t total_received = 0;
    while (total_received < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       NCCL_NET_HANDLE_MAXSIZE - total_received, 0);
      if (r <= 0) {
        std::cout << "�?FAILED: receiving handle" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }
    close(bootstrap_fd);

    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;
    if (tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle) != 0 || !send_comm) {
      std::cout << "�?FAILED: tcpx_connect_v5" << std::endl;
      return 1;
    }

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;
    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      tcpx_close_send(send_comm);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    cuda_check(cuMemsetD8(d_aligned, 0, kRegisteredBytes), "cuMemsetD8");
    cuda_check(cuMemcpyHtoD(d_aligned, kTestMessage, sizeof(kTestMessage)), "cuMemcpyHtoD");

    void* send_mhandle = nullptr;
    if (tcpx_reg_mr(send_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &send_mhandle) != 0) {
      std::cout << "�?FAILED: tcpx_reg_mr (send)" << std::endl;
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    void* send_request = nullptr;
    if (tcpx_isend(send_comm, reinterpret_cast<void*>(d_aligned),
                   sizeof(kTestMessage), kTransferTag, send_mhandle,
                   &send_request) != 0) {
      std::cout << "�?FAILED: tcpx_isend" << std::endl;
      tcpx_dereg_mr(send_comm, send_mhandle);
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    int done = 0, sent_size = 0;
    for (int i = 0; i < 200000 && !done; ++i) {
      if (tcpx_test(send_request, &done, &sent_size) != 0) break;
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    tcpx_dereg_mr(send_comm, send_mhandle);
    cuMemFree(d_base);
    tcpx_close_send(send_comm);
    cuDevicePrimaryCtxRelease(cuDev);
    return 0;
  }
}
