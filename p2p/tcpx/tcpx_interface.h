#pragma once

#include <stdint.h>

// TCPX interface definitions
// Based on nccl-plugin-gpudirecttcpx/src/net_tcpx.h

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for TCPX types
typedef int tcpxResult_t;
typedef void* tcpxDebugLogger_t;
typedef void* devNetDeviceHandle;

// TCPX result codes (matching nccl-plugin-gpudirecttcpx)
#define tcpxSuccess 0
#define tcpxInternalError 1
#define tcpxSystemError 2
#define tcpxInvalidArgument 3

// TCPX pointer types (matching nccl-plugin-gpudirecttcpx)
#define TCPX_PTR_HOST 1
#define TCPX_PTR_CUDA 2

// TCPX network properties (simplified)
typedef struct {
  char* name;
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int speed;
  int maxComms;
  float latency;
  int maxRecvs;
} tcpxNetProperties_t;

// TCPX plugin function declarations
tcpxResult_t tcpxInit(tcpxDebugLogger_t logFunction);
tcpxResult_t tcpxDevices(int* ndev);
tcpxResult_t tcpxGetProperties(int dev, tcpxNetProperties_t* props);
tcpxResult_t tcpxListen(int dev, void* oHandle, void** listenComm);
tcpxResult_t tcpxConnect_v5(int dev, void* oHandle, void** sendComm,
                            devNetDeviceHandle** sendDevHandle);
tcpxResult_t tcpxAccept_v5(void* listenComm, void** recvComm,
                           devNetDeviceHandle** recvDevHandle);
tcpxResult_t tcpxRegMr(void* ocomm, void* data, int size, int type,
                       void** mhandle);
tcpxResult_t tcpxDeregMr(void* comm, void* mhandle);
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

 private:
  static std::vector<DeviceInfo> devices_;
  static bool initialized_;
};

}  // namespace tcpx

#endif  // __cplusplus
