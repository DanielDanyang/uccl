#include "engine.h"
#include <cstdio>
#include <cstring>

int const kMaxNumGPUs = 8;
uint8_t gpu_to_dev[kMaxNumGPUs] = {0};

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  printf("[TCPX] Creating Engine with GPU index: %u, CPUs: %u\n", local_gpu_idx,
         num_cpus);

  // 初始化 TCPX endpoint
  ep_ = new tcpx::TcpxEndpoint(num_cpus_);

  // 简化的设备映射
  for (int i = 0; i < kMaxNumGPUs; i++) {
    gpu_to_dev[i] = ep_->get_best_dev_idx(i);
  }

  numa_node_ =
      tcpx::TcpxFactory::get_factory_dev(gpu_to_dev[local_gpu_idx_])->numa_node;

  // 初始化引擎
  printf("[TCPX] Initializing engine for GPU %u\n", local_gpu_idx_);
  ep_->initialize_engine_by_dev(gpu_to_dev[local_gpu_idx_], true);
  printf("[TCPX] Engine initialized for GPU %u\n", local_gpu_idx_);

  printf("[TCPX] Engine construction completed\n");
}

Endpoint::~Endpoint() {
  printf("[TCPX] Destroying Engine...\n");

  // 清理连接
  for (auto& [conn_id, conn] : conn_id_to_conn_) {
    delete conn;
  }

  // 清理内存注册
  for (auto& [mr_id, mr] : mr_id_to_mr_) {
    delete mr;
  }

  delete ep_;

  printf("[TCPX] Engine destroyed\n");
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  printf("[TCPX] Attempting to connect to %s:%d (GPU %d)\n", ip_addr.c_str(),
         remote_port, remote_gpu_idx);

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  // 简化的连接调用
  tcpx::ConnID tcpx_conn_id = ep_->tcpx_connect(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, gpu_to_dev[remote_gpu_idx],
      remote_gpu_idx, ip_addr, remote_port);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, tcpx_conn_id, ip_addr, remote_gpu_idx, remote_port};

  printf("[TCPX] Connected successfully with conn_id: %lu\n", conn_id);
  return true;
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  printf("[TCPX] Waiting to accept incoming connection...\n");

  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  // 简化的接受连接调用
  tcpx::ConnID tcpx_conn_id = ep_->tcpx_accept(
      gpu_to_dev[local_gpu_idx_], local_gpu_idx_, ip_addr, remote_gpu_idx);

  // Store the connection ID.
  conn_id_to_conn_[conn_id] =
      new Conn{conn_id, tcpx_conn_id, ip_addr, remote_gpu_idx, -1};

  printf("[TCPX] Accepted connection from %s (GPU %d) with conn_id: %lu\n",
         ip_addr.c_str(), remote_gpu_idx, conn_id);
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  printf("[TCPX] Getting metadata for GPU %u\n", local_gpu_idx_);

  // 简化：返回固定的元数据
  std::vector<uint8_t> metadata(10);  // 简化的元数据大小

  // 填充基本信息
  metadata[0] = local_gpu_idx_;
  metadata[1] = numa_node_;

  printf("[TCPX] Metadata generated\n");
  return metadata;
}
