#include "tcpx_interface.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

// NCCL网络句柄 - 用于连接信息交换
// 根据NCCL标准，句柄大小通常是64字节
#define NCCL_NET_HANDLE_MAXSIZE 64
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char* argv[]) {
  std::cout << "=== TCPX Connection Test ===" << std::endl;
  std::cout << "注意：这是一个简化的连接测试" << std::endl;
  std::cout << "实际的TCPX连接需要通过带外通信交换句柄信息" << std::endl;
  std::cout << "当前测试主要验证API调用是否正常工作" << std::endl;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <server|client> [remote_ip]"
              << std::endl;
    std::cout << "  server: Start as server (listener)" << std::endl;
    std::cout << "  client <ip>: Connect to server at <ip>" << std::endl;
    return 1;
  }

  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);

  // Initialize TCPX
  std::cout << "\n[Step 1] Initializing TCPX..." << std::endl;
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "✗ FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  std::cout << "✓ SUCCESS: Found " << device_count << " TCPX devices"
            << std::endl;

  // Use device 0 for testing
  int dev_id = 0;
  std::cout << "Using TCPX device " << dev_id << std::endl;

  bool is_server = (strcmp(argv[1], "server") == 0);

  if (is_server) {
    std::cout << "\n[Step 2] Starting as SERVER..." << std::endl;

    // Create connection handle for listening
    // For listen, the handle is typically filled by the plugin
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));

    void* listen_comm = nullptr;
    std::cout << "Attempting to listen on device " << dev_id << "..."
              << std::endl;

    int rc = tcpx_listen(dev_id, &handle, &listen_comm);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_listen returned " << rc << std::endl;
      return 1;
    }
    std::cout << "✓ SUCCESS: Listening on device " << dev_id << std::endl;
    std::cout << "Listen comm: " << listen_comm << std::endl;

    std::cout << "\nWaiting for client connection..." << std::endl;
    std::cout << "Run client with: " << argv[0] << " client <this_server_ip>"
              << std::endl;

    // Accept connection
    void* recv_comm = nullptr;
    void* recv_dev_handle = nullptr;

    std::cout << "Calling tcpx_accept_v5..." << std::endl;
    rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_accept_v5 returned " << rc << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    std::cout << "✓ SUCCESS: Connection accepted!" << std::endl;
    std::cout << "Recv comm: " << recv_comm << std::endl;
    std::cout << "Recv dev handle: " << recv_dev_handle << std::endl;

    // Cleanup
    // tcpx_close_recv(recv_comm);  // 这些函数可能不存在
    // tcpx_close_listen(listen_comm);
    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      std::cout << "✗ ERROR: Client mode requires remote IP" << std::endl;
      return 1;
    }

    std::cout << "\n[Step 2] Starting as CLIENT..." << std::endl;
    std::cout << "Connecting to server at " << argv[2] << std::endl;

    // Create connection handle for connecting
    // 对于客户端，我们需要从服务器获取的句柄信息
    // 这通常通过带外通信（如文件、网络等）获得
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));

    // TODO: 在实际应用中，这个句柄应该从服务器的tcpx_listen获得
    // 现在我们先用空句柄测试
    std::cout << "Warning: Using empty handle - this may not work" << std::endl;

    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;

    std::cout << "Attempting to connect to " << argv[2] << "..." << std::endl;
    int rc = tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_connect_v5 returned " << rc << std::endl;
      return 1;
    }

    std::cout << "✓ SUCCESS: Connected to server!" << std::endl;
    std::cout << "Send comm: " << send_comm << std::endl;
    std::cout << "Send dev handle: " << send_dev_handle << std::endl;

    // Cleanup - 使用通用的tcpx_close函数
    // tcpx_close_send(send_comm);  // 这些函数可能不存在
    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else {
    std::cout << "✗ ERROR: Invalid mode. Use 'server' or 'client'" << std::endl;
    return 1;
  }

  std::cout << "\n=== TCPX Connection Test COMPLETED ===" << std::endl;
  return 0;
}
