/**
 * @file test_endpoint_tcpx.cc
 * @brief 测试TCPX Endpoint集成的简化版本
 * 
 * 这个测试展示了如何将TCPX连接功能集成到Endpoint类中。
 * 由于完整的Endpoint类有很多依赖，这里提供一个简化的测试框架。
 */

#include "tcpx_interface.h"
#include <iostream>
#include <string>
#include <atomic>
#include <unordered_map>
#include <memory>

// 简化的连接结构体
struct TcpxConnection {
  uint64_t conn_id;
  std::string remote_ip;
  int remote_gpu_idx;
  void* send_comm;
  void* recv_comm;
  void* send_dev_handle;
  void* recv_dev_handle;
  
  TcpxConnection(uint64_t id, const std::string& ip, int gpu_idx)
    : conn_id(id), remote_ip(ip), remote_gpu_idx(gpu_idx),
      send_comm(nullptr), recv_comm(nullptr),
      send_dev_handle(nullptr), recv_dev_handle(nullptr) {}
};

// 简化的TCPX Endpoint类
class TcpxEndpoint {
private:
  int local_gpu_idx_;
  std::atomic<uint64_t> next_conn_id_{1};
  std::unordered_map<uint64_t, std::unique_ptr<TcpxConnection>> connections_;
  void* listen_comm_;
  
public:
  explicit TcpxEndpoint(int local_gpu_idx) 
    : local_gpu_idx_(local_gpu_idx), listen_comm_(nullptr) {
    
    std::cout << "Initializing TCPX Endpoint for GPU " << local_gpu_idx << std::endl;
    
    // Initialize TCPX
    int ndev = tcpx_get_device_count();
    if (ndev <= 0) {
      throw std::runtime_error("No TCPX devices found");
    }
    
    std::cout << "Found " << ndev << " TCPX devices" << std::endl;
  }
  
  ~TcpxEndpoint() {
    std::cout << "Destroying TCPX Endpoint" << std::endl;
    // TODO: Cleanup connections and listen_comm_
  }
  
  // 开始监听连接
  bool start_listening() {
    std::cout << "Starting TCPX listener..." << std::endl;
    
    // Create handle for listening
    struct ncclNetHandle_v7 {
      char data[128];
    };
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));
    
    int rc = tcpx_listen(0, &handle, &listen_comm_);
    if (rc != 0) {
      std::cout << "Failed to start TCPX listener: " << rc << std::endl;
      return false;
    }
    
    std::cout << "TCPX listener started successfully" << std::endl;
    std::cout << "Listen comm: " << listen_comm_ << std::endl;
    
    // TODO: Save handle to file or share via OOB mechanism
    std::cout << "TODO: Share connection handle with remote clients" << std::endl;
    
    return true;
  }
  
  // 接受连接
  uint64_t accept_connection() {
    if (!listen_comm_) {
      std::cout << "No active listener" << std::endl;
      return 0;
    }
    
    std::cout << "Accepting TCPX connection..." << std::endl;
    
    uint64_t conn_id = next_conn_id_.fetch_add(1);
    auto conn = std::make_unique<TcpxConnection>(conn_id, "unknown", 0);
    
    int rc = tcpx_accept_v5(listen_comm_, &conn->recv_comm, &conn->recv_dev_handle);
    if (rc != 0) {
      std::cout << "Failed to accept TCPX connection: " << rc << std::endl;
      return 0;
    }
    
    std::cout << "TCPX connection accepted successfully" << std::endl;
    std::cout << "Connection ID: " << conn_id << std::endl;
    std::cout << "Recv comm: " << conn->recv_comm << std::endl;
    std::cout << "Recv dev handle: " << conn->recv_dev_handle << std::endl;
    
    connections_[conn_id] = std::move(conn);
    return conn_id;
  }
  
  // 连接到远程端点
  uint64_t connect_to_remote(const std::string& remote_ip, int remote_gpu_idx) {
    std::cout << "Connecting to " << remote_ip << ":" << remote_gpu_idx << std::endl;
    
    uint64_t conn_id = next_conn_id_.fetch_add(1);
    auto conn = std::make_unique<TcpxConnection>(conn_id, remote_ip, remote_gpu_idx);
    
    // TODO: Load connection handle from remote server
    struct ncclNetHandle_v7 {
      char data[128];
    };
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));
    
    std::cout << "TODO: Load connection handle from remote server" << std::endl;
    
    int rc = tcpx_connect_v5(0, &handle, &conn->send_comm, &conn->send_dev_handle);
    if (rc != 0) {
      std::cout << "Failed to connect to remote: " << rc << std::endl;
      return 0;
    }
    
    std::cout << "TCPX connection established successfully" << std::endl;
    std::cout << "Connection ID: " << conn_id << std::endl;
    std::cout << "Send comm: " << conn->send_comm << std::endl;
    std::cout << "Send dev handle: " << conn->send_dev_handle << std::endl;
    
    connections_[conn_id] = std::move(conn);
    return conn_id;
  }
  
  // 获取连接信息
  TcpxConnection* get_connection(uint64_t conn_id) {
    auto it = connections_.find(conn_id);
    return (it != connections_.end()) ? it->second.get() : nullptr;
  }
  
  // 显示所有连接
  void show_connections() {
    std::cout << "\n=== TCPX Connections ===" << std::endl;
    if (connections_.empty()) {
      std::cout << "No active connections" << std::endl;
    } else {
      for (const auto& [id, conn] : connections_) {
        std::cout << "Connection " << id << ": " << conn->remote_ip 
                  << ":" << conn->remote_gpu_idx << std::endl;
        std::cout << "  Send comm: " << conn->send_comm << std::endl;
        std::cout << "  Recv comm: " << conn->recv_comm << std::endl;
      }
    }
    std::cout << "========================" << std::endl;
  }
};

int main(int argc, char* argv[]) {
  std::cout << "=== TCPX Endpoint Integration Test ===" << std::endl;
  
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <server|client> [remote_ip]" << std::endl;
    return 1;
  }
  
  try {
    TcpxEndpoint endpoint(0);  // Use GPU 0
    
    if (strcmp(argv[1], "server") == 0) {
      std::cout << "\n=== Server Mode ===" << std::endl;
      
      if (!endpoint.start_listening()) {
        std::cout << "Failed to start server" << std::endl;
        return 1;
      }
      
      std::cout << "Press Enter to accept connection..." << std::endl;
      std::cin.get();
      
      uint64_t conn_id = endpoint.accept_connection();
      if (conn_id > 0) {
        endpoint.show_connections();
      }
      
    } else if (strcmp(argv[1], "client") == 0) {
      if (argc < 3) {
        std::cout << "Client mode requires remote IP" << std::endl;
        return 1;
      }
      
      std::cout << "\n=== Client Mode ===" << std::endl;
      
      uint64_t conn_id = endpoint.connect_to_remote(argv[2], 0);
      if (conn_id > 0) {
        endpoint.show_connections();
      }
      
    } else {
      std::cout << "Invalid mode. Use 'server' or 'client'" << std::endl;
      return 1;
    }
    
  } catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  std::cout << "\n=== TCPX Endpoint Test Completed ===" << std::endl;
  return 0;
}
