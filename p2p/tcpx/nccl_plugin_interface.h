#pragma once

#include <stdint.h>

// NCCL Plugin Interface for TCPX
// Based on nccl-plugin-gpudirecttcpx/src/nccl_plugin.cc

#ifdef __cplusplus
extern "C" {
#endif

// NCCL result types
typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclInternalError 1
#define ncclInvalidArgument 2
#define ncclSystemError 3

// NCCL pointer types
#define NCCL_PTR_HOST 1
#define NCCL_PTR_CUDA 2

// Forward declarations
typedef void* ncclNetHandle_t;
typedef void* ncclNetDeviceHandle_v7_t;

// NCCL debug logger type
typedef int (*ncclDebugLogger_t)(int level, const char* file, const char* func, 
                                 int line, const char* fmt, ...);

// NCCL network properties
typedef struct {
    char* name;
    char* pciPath;
    uint64_t guid;
    int ptrSupport;
    int speed;
    int maxComms;
    float latency;
    int maxRecvs;
} ncclNetProperties_t;

// NCCL Net v7 function table structure
typedef struct {
    // Plugin name
    const char* name;
    
    // Core functions
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
    
    // Connection management
    ncclResult_t (*listen)(int dev, void* oHandle, void** listenComm);
    ncclResult_t (*connect)(int dev, void* oHandle, void** sendComm, ncclNetDeviceHandle_v7_t** sendDevHandle);
    ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v7_t** recvDevHandle);
    
    // Memory management
    ncclResult_t (*regMr)(void* ocomm, void* data, int size, int type, void** mhandle);
    ncclResult_t (*regMrDmaBuf)(void* ocomm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
    ncclResult_t (*deregMr)(void* ocomm, void* mhandle);
    
    // Data transfer
    ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
    ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
    ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
    ncclResult_t (*test)(void* request, int* done, int* sizes);
    
    // Cleanup
    ncclResult_t (*closeSend)(void* sendComm);
    ncclResult_t (*closeRecv)(void* recvComm);
    ncclResult_t (*closeListen)(void* listenComm);
    
    // Device operations
    ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
    ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
    
} ncclNet_v7_t;

// External symbol from TCPX plugin
extern volatile ncclNet_v7_t ncclNetPlugin_v7;

#ifdef __cplusplus
}
#endif

// C++ wrapper for NCCL TCPX plugin
#ifdef __cplusplus

#include <memory>
#include <string>

namespace tcpx {

// TCPX connection ID structure
struct ConnID {
    void* sendComm;
    void* recvComm;
    ncclNetDeviceHandle_v7_t* sendDevHandle;
    ncclNetDeviceHandle_v7_t* recvDevHandle;
    int sock_fd;
};

// TCPX memory handle structure  
struct Mhandle {
    void* nccl_mhandle;
    void* data;
    size_t size;
    int type;
};

// TCPX Plugin Manager - manages NCCL plugin interface
class TcpxPluginManager {
public:
    static TcpxPluginManager& getInstance();
    
    // Plugin management
    bool loadPlugin(const char* plugin_path);
    bool isLoaded() const { return plugin_loaded_; }
    
    // Access to NCCL plugin functions
    const ncclNet_v7_t* getPlugin() const { return plugin_; }
    
    // Convenience wrappers
    ncclResult_t init(ncclDebugLogger_t logger);
    ncclResult_t devices(int* ndev);
    ncclResult_t getProperties(int dev, ncclNetProperties_t* props);
    
private:
    TcpxPluginManager() = default;
    ~TcpxPluginManager();
    
    void* dl_handle_ = nullptr;
    const ncclNet_v7_t* plugin_ = nullptr;
    bool plugin_loaded_ = false;
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
                        int remote_gpu_idx, std::string const& ip_addr, int remote_port);
    
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

} // namespace tcpx

#endif // __cplusplus
