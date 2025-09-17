#include "tcpx_interface.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tcpx {

// 简化的 TcpxFactory 实现
std::vector<TcpxFactory::DeviceInfo> TcpxFactory::devices_;
bool TcpxFactory::initialized_ = false;

void TcpxFactory::initialize() {
    if (initialized_) return;
    
    fprintf(stderr, "[TCPX] Initializing TcpxFactory (simplified mode)\n");
    
    // 创建一个虚拟设备
    devices_.resize(1);
    devices_[0].numa_node = 0;
    devices_[0].name = "tcpx0";
    devices_[0].pci_path = "0000:00:00.0";
    
    fprintf(stderr, "[TCPX] Created 1 virtual TCPX device\n");
    initialized_ = true;
}

TcpxFactory::DeviceInfo* TcpxFactory::get_factory_dev(int dev_idx) {
    initialize();
    if (dev_idx < 0 || dev_idx >= (int)devices_.size()) {
        return &devices_[0];  // Fallback to first device
    }
    return &devices_[dev_idx];
}

// 简化的 TcpxEndpoint 实现
struct TcpxEndpoint::Impl {
    uint32_t num_cpus;
    int listen_port;
    
    Impl(uint32_t cpus) : num_cpus(cpus), listen_port(12345) {}
};

TcpxEndpoint::TcpxEndpoint(uint32_t num_cpus) {
    fprintf(stderr, "[TCPX] Creating TcpxEndpoint with %u CPUs (simplified)\n", num_cpus);
    impl_ = new Impl(num_cpus);
    TcpxFactory::initialize();
}

TcpxEndpoint::~TcpxEndpoint() {
    fprintf(stderr, "[TCPX] Destroying TcpxEndpoint\n");
    delete impl_;
}

int TcpxEndpoint::get_best_dev_idx(int gpu_idx) {
    // Simple mapping: GPU index to device index
    TcpxFactory::initialize();
    int ndev = TcpxFactory::devices_.size();
    return gpu_idx % ndev;
}

void TcpxEndpoint::initialize_engine_by_dev(int dev_idx, bool lazy_init) {
    fprintf(stderr, "[TCPX] Initializing engine for device %d (lazy=%d, simplified)\n", 
            dev_idx, lazy_init);
    
    impl_->listen_port = 12345 + dev_idx;
    fprintf(stderr, "[TCPX] Engine initialized for device %d, port %d\n", 
            dev_idx, impl_->listen_port);
}

ConnID TcpxEndpoint::tcpx_connect(int local_dev, int local_gpu_idx, int remote_dev, 
                                  int remote_gpu_idx, std::string const& ip_addr, int remote_port) {
    fprintf(stderr, "[TCPX] Connecting (simplified): local_dev=%d local_gpu=%d -> remote_dev=%d remote_gpu=%d %s:%d\n",
            local_dev, local_gpu_idx, remote_dev, remote_gpu_idx, ip_addr.c_str(), remote_port);
    
    ConnID conn_id;
    conn_id.sendComm = nullptr;
    conn_id.recvComm = nullptr;
    conn_id.sendDevComm = nullptr;
    conn_id.recvDevComm = nullptr;
    conn_id.sock_fd = -1;
    
    fprintf(stderr, "[TCPX] Connection established (simplified)\n");
    return conn_id;
}

std::unique_ptr<Mhandle> TcpxEndpoint::reg_mr(void* data, size_t size, int type) {
    fprintf(stderr, "[TCPX] Registering memory: data=%p size=%zu type=%d (simplified)\n", 
            data, size, type);
    
    auto mhandle = std::make_unique<Mhandle>();
    mhandle->tcpx_mhandle = nullptr;
    mhandle->data = data;
    mhandle->size = size;
    mhandle->type = type;
    
    return mhandle;
}

void TcpxEndpoint::dereg_mr(std::unique_ptr<Mhandle> mhandle) {
    fprintf(stderr, "[TCPX] Deregistering memory: data=%p (simplified)\n", 
            mhandle->data);
}

bool TcpxEndpoint::send_async(ConnID const& conn_id, void* data, size_t size, 
                             Mhandle const& mhandle, uint64_t* transfer_id) {
    fprintf(stderr, "[TCPX] Async send: data=%p size=%zu (simplified)\n", data, size);
    *transfer_id = 1;  // Dummy transfer ID
    return true;
}

bool TcpxEndpoint::recv_async(ConnID const& conn_id, void* data, size_t size,
                             Mhandle const& mhandle, uint64_t* transfer_id) {
    fprintf(stderr, "[TCPX] Async recv: data=%p size=%zu (simplified)\n", data, size);
    *transfer_id = 2;  // Dummy transfer ID
    return true;
}

bool TcpxEndpoint::test_transfer(uint64_t transfer_id, bool* done) {
    fprintf(stderr, "[TCPX] Test transfer: id=%lu (simplified)\n", transfer_id);
    *done = true;  // Always completed in simplified mode
    return true;
}

int TcpxEndpoint::get_p2p_listen_port() {
    return impl_->listen_port;
}

} // namespace tcpx
