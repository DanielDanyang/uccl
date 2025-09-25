/**
 * @file test_tcpx_transfer.cc
 * @brief TCPX GPU到GPU数据传输测试
 *
 * 这个测试验证TCPX插件在GPU设备内存之间进行直接数据传输的能力�? * 与test_connection.cc不同，这个测试专门测试GPU内存的注册和传输�? * 是验证TCPX GPUDirect功能的核心测试�? *
 * 测试流程�? * 1. 服务器端：监听连接，分配GPU内存，注册内存，等待接收数据
 * 2. 客户端：连接服务器，分配GPU内存，准备测试数据，发送数�? * 3. 验证：检查接收到的数据是否与发送的数据一�? *
 * 关键特性：
 * - 使用CUDA设备内存 (NCCL_PTR_CUDA)
 * - 4KB对齐的内存分�?(GPUDirect TCPX要求)
 * - 完整的错误处理和资源清理
 * - 详细的调试输出和十六进制数据转储
 */

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
/**
 * @brief 测试配置常量
 *
 * 这些常量定义了测试的关键参数�? * - NCCL_NET_HANDLE_MAXSIZE: NCCL网络句柄的最大大�?(128字节)
 * - kBootstrapPort: 用于句柄交换的TCP端口
 * - kRegisteredBytes: 注册的GPU内存大小�?KB对齐以满足GPUDirect TCPX要求
 * - kTestMessage: 测试传输的消息内�? * - kTransferTag: TCPX传输标签，用于匹配发送和接收操作
 */
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

constexpr int kBootstrapPort = 12345;
constexpr size_t kRegisteredBytes = 4096;  // 4KB对齐，满足GPUDirect TCPX要求
constexpr char kTestMessage[] = "Hello from TCPX client!";  // 测试消息内容
constexpr size_t kPayloadBytes = sizeof(kTestMessage);      // 实际传输字节�?constexpr int kTransferTag = 42;                            // 传输标签，用于匹配发送和接收操作

/**
 * @brief 创建引导服务器用于句柄交�? *
 * 在TCPX连接建立之前，需要通过普通TCP连接交换NCCL网络句柄�? * 这个函数创建一个TCP服务器来接收客户端连接并交换句柄�? *
 * @return 成功返回客户端socket文件描述符，失败返回-1
 */
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

/**
 * @brief 连接到引导服务器获取句柄
 *
 * 客户端使用这个函数连接到服务器的引导socket�? * 用于接收NCCL网络句柄，然后才能建立TCPX连接�? *
 * @param server_ip 服务器IP地址
 * @return 成功返回socket文件描述符，失败返回-1
 */
int connect_to_bootstrap_server(const char* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kBootstrapPort);
  inet_aton(server_ip, &addr.sin_addr);

  // 重试连接，因为服务器可能还没准备�?  for (int retry = 0; retry < 10; ++retry) {
    if (connect(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return sock_fd;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));  // 等待1秒后重试
  }
  close(sock_fd);
  return -1;
}

/**
 * @brief 以十六进制格式转储数据内�? *
 * 用于调试目的，显示内存中数据的十六进制表示�? * 限制显示�?2字节以避免输出过长�? *
 * @param data 要转储的数据指针
 * @param bytes 数据字节�? */
void dump_hex(const void* data, size_t bytes) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t limit = std::min<size_t>(bytes, 32);  // 限制显示�?2字节
  for (size_t i = 0; i < limit; ++i) {
    if (i && i % 16 == 0) std::cout << "\n";  // �?6字节换行
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(p[i]) << ' ';
  }
  std::cout << std::dec << std::endl;
}

/**
 * @brief CUDA错误检查辅助函�? *
 * 检查CUDA API调用的返回值，如果出错则打印详细的错误信息�? * 这对于调试CUDA相关问题非常有用�? *
 * @param res CUDA API的返回�? * @param what 描述正在执行的操�? * @return 成功返回true，失败返回false
 */
bool cuda_check(CUresult res, const char* what) {
  if (res == CUDA_SUCCESS) return true;

  // 获取CUDA错误名称和详细描�?  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &desc);

  std::cout << "CUDA error at " << what << ": "
            << (name ? name : "?") << " - " << (desc ? desc : "")
            << std::endl;
  return false;
}

}  // namespace

/**
 * @brief 主函�?- TCPX GPU到GPU传输测试
 *
 * 这个测试程序验证TCPX插件在GPU设备内存之间进行直接数据传输的能力�? * 程序可以运行在服务器模式或客户端模式�? *
 * 服务器模式流程：
 * 1. 初始化TCPX设备
 * 2. 创建监听连接
 * 3. 通过引导连接发送句柄给客户�? * 4. 接受TCPX连接
 * 5. 分配和注册GPU内存
 * 6. 等待接收数据
 * 7. 验证接收到的数据
 *
 * 客户端模式流程：
 * 1. 初始化TCPX设备
 * 2. 通过引导连接获取服务器句�? * 3. 建立TCPX连接
 * 4. 分配和注册GPU内存
 * 5. 准备测试数据
 * 6. 发送数据到服务�? */
int main(int argc, char** argv) {
  std::cout << "=== TCPX 设备到设备传输测�?===" << std::endl;
  if (argc < 2) {
    std::cout << "用法: " << argv[0] << " <server|client> [remote_ip]" << std::endl;
    return 1;
  }

  // 启用TCPX调试输出
  setenv("UCCL_TCPX_DEBUG", "1", 1);

  // 初始化TCPX并检查设备数�?  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "�?失败: 未找到TCPX设备" << std::endl;
    return 1;
  }
  int dev_id = 0;  // 使用第一个TCPX设备
  bool is_server = std::strcmp(argv[1], "server") == 0;

  if (is_server) {
    std::cout << "启动服务器模�? << std::endl;

    // === 步骤1: 创建TCPX监听连接 ===
    // 生成NCCL网络句柄，用于客户端连接
    ncclNetHandle_v7 handle{};
    void* listen_comm = nullptr;
    if (tcpx_listen(dev_id, &handle, &listen_comm) != 0) {
      std::cout << "�?失败: tcpx_listen" << std::endl;
      return 1;
    }
    std::cout << "�?TCPX在设�?" << dev_id << " 上监�? << std::endl;

    // === 步骤2: 通过引导连接发送句柄给客户�?===
    // 创建普通TCP服务器用于句柄交�?    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "�?失败: 引导服务器创建失�? << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }
    std::cout << "�?引导服务器已创建，正在发送句柄给客户�?.." << std::endl;

    // 发送完整的NCCL句柄数据
    size_t total_sent = 0;
    while (total_sent < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t s = send(bootstrap_fd, handle.data + total_sent,
                       NCCL_NET_HANDLE_MAXSIZE - total_sent, 0);
      if (s <= 0) {
        std::cout << "�?失败: 发送句柄失�? << std::endl;
        close(bootstrap_fd);
        tcpx_close_listen(listen_comm);
        return 1;
      }
      total_sent += static_cast<size_t>(s);
    }
    close(bootstrap_fd);
    std::cout << "�?句柄已发送给客户�?(" << total_sent << " 字节)" << std::endl;

    void* recv_comm = nullptr;
    alignas(16) unsigned char recv_dev_handle_storage[512]{};
    void* recv_dev_handle = recv_dev_handle_storage;
    constexpr int kAcceptMaxRetries = 120;  // ~12s total with 100ms sleep
    int accept_attempts = 0;
    while (accept_attempts < kAcceptMaxRetries) {
      if (tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle) != 0) {
        std::cout << "�?FAILED: tcpx_accept_v5" << std::endl;
        tcpx_close_listen(listen_comm);
        return 1;
      }
      if (recv_comm) break;
      ++accept_attempts;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!recv_comm) {
      std::cout << "�?FAILED: tcpx_accept_v5 returned null after retries" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    // === 步骤4: 初始化CUDA并分配GPU内存 ===
    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;

    std::cout << "初始化CUDA并分配GPU内存..." << std::endl;
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

    // 确保GPU内存4KB对齐 (GPUDirect TCPX要求)
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);  // 4KB对齐
    d_aligned = static_cast<CUdeviceptr>(addr);
    std::cout << "�?GPU内存已分配并对齐: " << std::hex << d_aligned << std::dec << std::endl;

    // === 步骤5: 注册GPU内存到TCPX ===
    void* recv_mhandle = nullptr;
    std::cout << "向TCPX注册GPU内存 (NCCL_PTR_CUDA)..." << std::endl;
    if (tcpx_reg_mr(recv_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &recv_mhandle) != 0) {
      std::cout << "�?失败: tcpx_reg_mr (recv) - GPU内存注册失败" << std::endl;
      std::cout << "   这通常意味着gpumemd服务未运行或GPU DMA-BUF不受支持" << std::endl;
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "�?GPU内存注册成功，句�? " << recv_mhandle << std::endl;

    // === 步骤6: 发起异步接收请求 ===
    void* recv_data[1] = {reinterpret_cast<void*>(d_aligned)};
    int recv_sizes[1] = {static_cast<int>(kPayloadBytes)};
    int recv_tags[1] = {kTransferTag};  // 必须与客户端发送标签匹�?    void* recv_mhandles[1] = {recv_mhandle};
    void* recv_request = nullptr;

    std::cout << "发起接收请求 (标签=" << kTransferTag << ", 大小=" << kRegisteredBytes << ")..." << std::endl;
    if (tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                   recv_mhandles, &recv_request) != 0) {
      std::cout << "�?失败: tcpx_irecv" << std::endl;
      tcpx_dereg_mr(recv_comm, recv_mhandle);
      cuMemFree(d_base);
      tcpx_close_recv(recv_comm);
      tcpx_close_listen(listen_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "�?接收请求已发起，等待数据..." << std::endl;

    int done = 0, received_size = 0;
    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(recv_request, &done, &received_size);
      if (rc_test != 0) {
        std::cout << "�?FAILED: tcpx_test returned " << rc_test << std::endl;
        tcpx_dereg_mr(recv_comm, recv_mhandle);
        cuMemFree(d_base);
        tcpx_close_recv(recv_comm);
        tcpx_close_listen(listen_comm);
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
      }
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    cuCtxSynchronize();

    // === 步骤8: 验证接收到的数据 ===
    std::cout << "传输完成！验证接收到的数�?.." << std::endl;

    // 将GPU内存数据复制到主机内存进行验�?    std::vector<unsigned char> host(kRegisteredBytes, 0);
    cuda_check(cuMemcpyDtoH(host.data(), d_aligned, kPayloadBytes), "cuMemcpyDtoH");

    std::cout << "接收到的数据 (十六进制预览):" << std::endl;
    dump_hex(host.data(), kPayloadBytes);

    // 验证数据完整�?    bool match = std::memcmp(host.data(), kTestMessage, kPayloadBytes) == 0;
    if (match) {
      std::cout << "🎉 成功: 数据匹配预期字符�?" << std::endl;
      std::cout << "   预期: \"" << kTestMessage << "\"" << std::endl;
      std::cout << "   接收: \"" << reinterpret_cast<const char*>(host.data()) << "\"" << std::endl;
    } else {
      std::cout << "�?失败: 数据不匹�? << std::endl;
      std::cout << "   预期: \"" << kTestMessage << "\"" << std::endl;
      std::cout << "   接收: \"" << reinterpret_cast<const char*>(host.data()) << "\"" << std::endl;
    }

    // 告诉 TCPX 数据已消费，释放内部资源
    tcpx_irecv_consumed(recv_comm, 1, recv_request);

    if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
    cuMemFree(d_base);
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);
    cuDevicePrimaryCtxRelease(cuDev);
    return match ? 0 : 1;

  } else {
    // === 客户端模�?===
    if (argc < 3) {
      std::cout << "�?错误: 客户端模式需要远程IP地址" << std::endl;
      return 1;
    }
    std::cout << "启动客户端模式，连接�?" << argv[2] << std::endl;

    // === 步骤1: 连接到引导服务器获取句柄 ===
    int bootstrap_fd = connect_to_bootstrap_server(argv[2]);
    if (bootstrap_fd < 0) {
      std::cout << "�?失败: 引导连接失败" << std::endl;
      return 1;
    }
    std::cout << "�?已连接到引导服务�? << std::endl;
    // 接收完整的NCCL句柄数据
    ncclNetHandle_v7 handle{};
    size_t total_received = 0;
    while (total_received < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t r = recv(bootstrap_fd, handle.data + total_received,
                       NCCL_NET_HANDLE_MAXSIZE - total_received, 0);
      if (r <= 0) {
        std::cout << "�?失败: 接收句柄失败" << std::endl;
        close(bootstrap_fd);
        return 1;
      }
      total_received += static_cast<size_t>(r);
    }
    close(bootstrap_fd);
    std::cout << "�?已接收句�?(" << total_received << " 字节)" << std::endl;

    // === 步骤2: 建立TCPX连接 ===
    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;
    std::cout << "建立TCPX连接..." << std::endl;
    if (tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle) != 0 || !send_comm) {
      std::cout << "�?失败: tcpx_connect_v5" << std::endl;
      return 1;
    }
    std::cout << "�?TCPX连接已建�? << std::endl;

    // === 步骤3: 初始化CUDA并准备发送数�?===
    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUdeviceptr d_base = 0;
    CUdeviceptr d_aligned = 0;

    std::cout << "初始化CUDA并准备发送数�?.." << std::endl;
    if (!cuda_check(cuInit(0), "cuInit") ||
        !cuda_check(cuDeviceGet(&cuDev, dev_id), "cuDeviceGet") ||
        !cuda_check(cuDevicePrimaryCtxRetain(&cuCtx, cuDev), "cuDevicePrimaryCtxRetain") ||
        !cuda_check(cuCtxSetCurrent(cuCtx), "cuCtxSetCurrent") ||
        !cuda_check(cuMemAlloc(&d_base, kRegisteredBytes + 4096), "cuMemAlloc")) {
      tcpx_close_send(send_comm);
      if (cuCtx) cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }

    // 确保GPU内存4KB对齐并准备测试数�?    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);  // 4KB对齐
    d_aligned = static_cast<CUdeviceptr>(addr);
    cuda_check(cuMemsetD8(d_aligned, 0, kRegisteredBytes), "cuMemsetD8");  // 清零
    cuda_check(cuMemcpyHtoD(d_aligned, kTestMessage, sizeof(kTestMessage)), "cuMemcpyHtoD");  // 复制测试数据
    std::cout << "�?GPU内存已准备，测试数据已复�? << std::endl;

    // === 步骤4: 注册GPU内存到TCPX ===
    void* send_mhandle = nullptr;
    std::cout << "向TCPX注册GPU内存用于发�?.." << std::endl;
    if (tcpx_reg_mr(send_comm, reinterpret_cast<void*>(d_aligned),
                    kRegisteredBytes, NCCL_PTR_CUDA, &send_mhandle) != 0) {
      std::cout << "�?失败: tcpx_reg_mr (send)" << std::endl;
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "�?GPU内存注册成功用于发�? << std::endl;

    // === 步骤5: 发送数据到服务�?===
    void* send_request = nullptr;
    std::cout << "发送数据到服务�?(大小=" << sizeof(kTestMessage) << ", 标签=" << kTransferTag << ")..." << std::endl;
    if (tcpx_isend(send_comm, reinterpret_cast<void*>(d_aligned),
                   static_cast<int>(kPayloadBytes), kTransferTag, send_mhandle,
                   &send_request) != 0) {
      std::cout << "�?失败: tcpx_isend" << std::endl;
      tcpx_dereg_mr(send_comm, send_mhandle);
      cuMemFree(d_base);
      tcpx_close_send(send_comm);
      cuDevicePrimaryCtxRelease(cuDev);
      return 1;
    }
    std::cout << "�?发送请求已发起" << std::endl;
    // === 步骤6: 等待发送完�?===
    int done = 0, sent_size = 0;
    std::cout << "等待发送完�?.." << std::endl;
    for (int i = 0; i < 200000 && !done; ++i) {
      int rc_test = tcpx_test(send_request, &done, &sent_size);
      if (rc_test != 0) {
        std::cout << "�?失败: tcpx_test 返回 " << rc_test << std::endl;
        break;
      }
      if (!done) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    if (done) {
      std::cout << "�?发送完成，已发�?" << sent_size << " 字节" << std::endl;
    } else {
      std::cout << "⚠️ 发送超�? << std::endl;
    }

    // 给服务器时间处理数据
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // === 步骤7: 清理资源 ===
    std::cout << "清理客户端资�?.." << std::endl;
    tcpx_dereg_mr(send_comm, send_mhandle);
    cuMemFree(d_base);
    tcpx_close_send(send_comm);
    cuDevicePrimaryCtxRelease(cuDev);
    std::cout << "�?客户端测试完�? << std::endl;
    return 0;
  }
}
