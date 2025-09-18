#pragma once

// 最简化的 TCPX 接口
// 目标：替换 RDMA 传输层，让现有 p2p/engine.cc 能使用 TCPX

extern "C" {
// 基础函数
int tcpx_get_device_count();
int tcpx_load_plugin(char const* plugin_path);
}
tcpxResult_t tcpxIsend_v5(void* sendComm, void* data, int size, int tag,
                          void* mhandle, void** request);
tcpxResult_t tcpxIrecv_v5(void* recvComm, int n, void** data, int* sizes,
                          int* tags, void** mhandles, void** request);
tcpxResult_t tcpxTest(void* request, int* done, int* sizes);
tcpxResult_t tcpxClose(void* ocomm);
tcpxResult_t tcpxCloseListen(void* listenComm);

#ifdef __cplusplus
}
#endif

// C++ wrapper class for TCPX operations
#ifdef __cplusplus

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tcpx {

// TCPX connection ID structure
struct ConnID {
  void* sendComm;
  void* recvComm;
  void* sendDevComm;
  void* recvDevComm;
  int sock_fd;
};

// TCPX memory handle structure
struct Mhandle {
  void* tcpx_mhandle;
  void* data;
  int size;
  int type;
};

// TCPX Endpoint class - equivalent to uccl::RDMAEndpoint
class TcpxEndpoint {
 public:
  TcpxEndpoint(uint32_t num_cpus);
  ~TcpxEndpoint();

  // Device management
  int get_best_dev_idx(int gpu_idx);
  void initialize_engine_by_dev(int dev_idx, bool lazy_init);

  // Connection management
  ConnID tcpx_connect(int local_dev, int local_gpu_idx, int remote_dev,
                      int remote_gpu_idx, std::string const& ip_addr,
                      int remote_port);

  ConnID tcpx_accept(int local_dev, int local_gpu_idx, std::string& ip_addr,
                     int& remote_gpu_idx);

  // Memory registration
  std::unique_ptr<Mhandle> reg_mr(void* data, size_t size, int type);
  void dereg_mr(std::unique_ptr<Mhandle> mhandle);

  // Data transfer operations
  bool send_async(ConnID const& conn_id, void* data, size_t size,
                  Mhandle const& mhandle, uint64_t* transfer_id);
  bool recv_async(ConnID const& conn_id, void* data, size_t size,
                  Mhandle const& mhandle, uint64_t* transfer_id);
  bool test_transfer(uint64_t transfer_id, bool* done);

  // Utility functions
  int get_p2p_listen_port();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Factory class for TCPX devices
class TcpxFactory {
 public:
  struct DeviceInfo {
    int numa_node;
    std::string name;
    std::string pci_path;
  };

  static DeviceInfo* get_factory_dev(int dev_idx);
  static void initialize();
  static int get_num_devices();

 private:
  static std::vector<DeviceInfo> devices_;
  static bool initialized_;
};

}  // namespace tcpx

#endif  // __cplusplus
