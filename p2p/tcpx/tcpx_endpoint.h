#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// NCCL plugin interface (simplified)
typedef struct {
  char const* name;
  int (*init)(int (*logFunction)(int, char const*, char const*, int,
                                 char const*, ...));
  int (*devices)(int* ndev);
  int (*getProperties)(int dev, void* props_v7);
  int (*listen)(int dev, void* handle, void** listenComm);
  int (*connect)(int dev, void* handle, void** sendComm, void** sendDevComm);
  int (*accept)(void* listenComm, void** recvComm, void** recvDevComm);
  int (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  int (*deregMr)(void* comm, void* mhandle);
  int (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle,
               void** request);
  int (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request);
  int (*test)(void* request, int* done, int* sizes);
  int (*closeSend)(void* sendComm);
  int (*closeRecv)(void* recvComm);
  int (*closeListen)(void* listenComm);
} ncclNet_v7_t;

namespace tcpx {

class TcpxEndpoint {
 public:
  /*
   * Create TCPX endpoint using NCCL TCPX plugin
   *
   * input:
   *   local_gpu_idx: the GPU index to use for the endpoint
   *   num_cpus: the number of CPUs to use for the endpoint
   */
  TcpxEndpoint(uint32_t local_gpu_idx, uint32_t num_cpus);

  ~TcpxEndpoint();

  /*
   * Connect to a remote server via TCPX
   *
   * input:
   *   ip_addr: the IP address of the remote server
   *   remote_gpu_idx: the GPU index of the remote server
   *   remote_port: the port of the remote server
   * output:
   *   conn_id: the ID of the connection
   */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  /*
   * Get endpoint metadata for connection establishment
   */
  std::vector<uint8_t> get_metadata();

  /*
   * Accept an incoming connection via TCPX
   *
   * output:
   *   ip_addr: the IP address of the remote client
   *   remote_gpu_idx: the GPU index of the remote client
   *   conn_id: the ID of the connection
   */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /*
   * Register memory region for TCPX operations
   */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  /*
   * Deregister memory region
   */
  bool dereg(uint64_t mr_id);

  /*
   * Write data to remote server (RDMA-style write)
   */
  bool write(uint64_t conn_id, uint64_t mr_id, void const* src, size_t size,
             void const* remote_addr_info, bool inside_python = true);

  /*
   * Write data asynchronously
   */
  bool write_async(uint64_t conn_id, uint64_t mr_id, void const* src,
                   size_t size, void const* remote_addr_info,
                   uint64_t* transfer_id);

  /*
   * Receive data from remote server
   */
  bool recv(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
            bool inside_python = true);

  /*
   * Advertise memory region for remote write access
   */
  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /*
   * Check if async transfer is complete
   */
  bool test_transfer(uint64_t conn_id, uint64_t transfer_id);

  /*
   * Wait for async transfer to complete
   */
  bool wait_transfer(uint64_t conn_id, uint64_t transfer_id);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tcpx
