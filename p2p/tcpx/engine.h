#pragma once

#include "tcpx_interface.h"  // TCPX plugin interface
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct MR {
  uint64_t mr_id_;
  void* tcpx_mhandle_;  // TCPX memory handle
};

struct Conn {
  uint64_t conn_id_;
  tcpx::ConnID tcpx_conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;
  int remote_port_;
};

// 简化的 IP 获取函数
static inline std::string get_oob_ip() {
  return "127.0.0.1";  // 简化：返回本地地址
}

class Endpoint {
 public:
  /*
   * Create engine for TCPX transport.
   */
  Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus);

  ~Endpoint();

  /*
   * Connect to a remote server via TCPX.
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  /*
   * Accept an incoming connection via TCPX.
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  std::vector<uint8_t> get_metadata();

 private:
  int local_gpu_idx_;
  uint32_t num_cpus_;
  int numa_node_;

  tcpx::TcpxEndpoint* ep_;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;

  // 简化的连接和内存注册映射
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;
};
