// Simplified TCPX engine implementation using TcpxEndpoint

#include "../uccl_engine.h"
#include "tcpx_endpoint.h"
#include <cstdio>
#include <cstring>
#include <string>

struct uccl_engine {
  tcpx::TcpxEndpoint* endpoint;
};

struct uccl_conn {
  uint64_t conn_id;
  uccl_engine* engine;
};

struct uccl_mr {
  uint64_t mr_id;
  uccl_engine* engine;
};

// Engine management
uccl_engine_t* uccl_engine_create(int local_gpu_idx, int num_cpus) {
  try {
    auto* eng = new uccl_engine;
    eng->endpoint = new tcpx::TcpxEndpoint(local_gpu_idx, num_cpus);
    return eng;
  } catch (...) {
    fprintf(stderr, "[TCPX] Engine creation failed\n");
    return nullptr;
  }
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (engine) {
    delete engine->endpoint;
    delete engine;
  }
}

// Metadata
int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;

  auto meta_vec = engine->endpoint->get_metadata();
  char* buf = new char[meta_vec.size() + 1];
  memcpy(buf, meta_vec.data(), meta_vec.size());
  buf[meta_vec.size()] = '\0';
  *metadata = buf;
  return 0;
}

void uccl_engine_free_endpoint_metadata(uint8_t* metadata) {
  delete[] reinterpret_cast<char*>(metadata);
}

// Connection management
uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 int remote_gpu_idx, int remote_port) {
  if (!engine || !ip_addr) return nullptr;

  auto* conn = new uccl_conn;
  conn->engine = engine;

  bool ok = engine->endpoint->connect(std::string(ip_addr), remote_gpu_idx,
                                      remote_port, conn->conn_id);
  if (!ok) {
    delete conn;
    return nullptr;
  }

  return conn;
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, void* ip_buf,
                                size_t ip_buf_size, int* remote_gpu_idx) {
  if (!engine || !remote_gpu_idx) return nullptr;

  auto* conn = new uccl_conn;
  conn->engine = engine;

  std::string ip_addr;
  int gpu_idx;
  bool ok = engine->endpoint->accept(ip_addr, gpu_idx, conn->conn_id);
  if (!ok) {
    delete conn;
    return nullptr;
  }

  *remote_gpu_idx = gpu_idx;
  if (ip_buf && ip_buf_size > 0) {
    strncpy(static_cast<char*>(ip_buf), ip_addr.c_str(), ip_buf_size - 1);
    static_cast<char*>(ip_buf)[ip_buf_size - 1] = '\0';
  }

  return conn;
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (conn) {
    delete conn;
  }
}

// Memory registration
uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, uintptr_t data_ptr,
                           size_t size) {
  if (!engine) return nullptr;

  auto* mr = new uccl_mr;
  mr->engine = engine;

  bool ok =
      engine->endpoint->reg(reinterpret_cast<void*>(data_ptr), size, mr->mr_id);
  if (!ok) {
    delete mr;
    return nullptr;
  }

  return mr;
}

void uccl_engine_mr_destroy(uccl_mr_t* mr) {
  if (mr) {
    mr->engine->endpoint->dereg(mr->mr_id);
    delete mr;
  }
}

// Data transfer
int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                      size_t size, uint64_t* transfer_id) {
  if (!conn || !mr || !data || !transfer_id) return -1;

  // For now, use a dummy remote address info
  char dummy_addr[64] = "dummy_remote_addr";

  bool ok = conn->engine->endpoint->write_async(conn->conn_id, mr->mr_id, data,
                                                size, dummy_addr, transfer_id);
  return ok ? 0 : -1;
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
                     size_t size) {
  if (!conn || !mr || !data) return -1;

  bool ok = conn->engine->endpoint->recv(conn->conn_id, mr->mr_id, data, size);
  return ok ? 0 : -1;
}

bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id) {
  if (!conn) return false;

  return conn->engine->endpoint->test_transfer(conn->conn_id, transfer_id);
}

// Advertise (for RDMA-style operations)
int uccl_engine_advertise(uccl_conn_t* conn, uccl_mr_t* mr, void* addr,
                          size_t len, char* out_buf) {
  if (!conn || !mr || !addr || !out_buf) return -1;

  bool ok = conn->engine->endpoint->advertise(conn->conn_id, mr->mr_id, addr,
                                              len, out_buf);
  return ok ? 0 : -1;
}

// Placeholder implementations for compatibility
int uccl_engine_get_fifo_item(uccl_conn_t*, void*) { return -1; }

int uccl_engine_get_sock_fd(uccl_conn_t* conn) {
  return conn ? -1 : -1;  // Not applicable for TCPX
}
