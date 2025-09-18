#include "tcpx_interface.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>

// NCCL网络句柄 - 用于连接信息交换
// 根据NCCL标准，句柄大小通常是128字节（增加缓冲区以避免溢出）
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

// 句柄文件路径 - 用于节点间交换连接信息
char const* HANDLE_FILE = "/tmp/tcpx_handle.dat";

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

    // Save handle to file for client to use
    std::cout << "\n[Step 3] Saving connection handle to file..." << std::endl;
    std::ofstream handle_file(HANDLE_FILE, std::ios::binary);
    if (!handle_file) {
      std::cout << "✗ FAILED: Cannot create handle file " << HANDLE_FILE
                << std::endl;
      return 1;
    }

    handle_file.write(handle.data, NCCL_NET_HANDLE_MAXSIZE);
    handle_file.close();
    std::cout << "✓ SUCCESS: Handle saved to " << HANDLE_FILE << std::endl;

    std::cout << "\nWaiting for client connection..." << std::endl;
    std::cout << "Run client with: " << argv[0] << " client <this_server_ip>"
              << std::endl;
    std::cout << "Press Enter when client is ready..." << std::endl;
    std::cin.get();  // Wait for user input

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

    // Cleanup handle file
    unlink(HANDLE_FILE);
    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      std::cout << "✗ ERROR: Client mode requires remote IP" << std::endl;
      return 1;
    }

    std::cout << "\n[Step 2] Starting as CLIENT..." << std::endl;
    std::cout << "Connecting to server at " << argv[2] << std::endl;

    // Load connection handle from file
    std::cout << "\n[Step 3] Loading connection handle from file..."
              << std::endl;
    std::ifstream handle_file(HANDLE_FILE, std::ios::binary);
    if (!handle_file) {
      std::cout << "✗ FAILED: Cannot open handle file " << HANDLE_FILE
                << std::endl;
      std::cout << "Make sure server has created the handle file first!"
                << std::endl;
      std::cout << "If nodes don't share filesystem, copy the file manually:"
                << std::endl;
      std::cout << "  scp " << HANDLE_FILE << " user@" << argv[2] << ":"
                << HANDLE_FILE << std::endl;
      return 1;
    }

    ncclNetHandle_v7 handle;
    handle_file.read(handle.data, NCCL_NET_HANDLE_MAXSIZE);
    handle_file.close();
    std::cout << "✓ SUCCESS: Handle loaded from " << HANDLE_FILE << std::endl;

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
