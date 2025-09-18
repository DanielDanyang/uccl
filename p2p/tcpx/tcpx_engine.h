#pragma once

#include "tcpx_interface.h"
#include <string>
#include <vector>

// 简化的 TCPX Endpoint 类 - 对应 p2p/engine.h 中的 Endpoint
class TcpxEndpoint {
public:
    TcpxEndpoint(uint32_t local_gpu_idx, uint32_t num_cpus);
    ~TcpxEndpoint();
    
    // 获取元数据 - 对应 RDMA 版本的接口
    std::vector<uint8_t> get_metadata();
    
    // 获取 OOB IP
    std::string get_oob_ip();
    
    // 获取设备数量
    int get_device_count();

private:
    uint32_t local_gpu_idx_;
    uint32_t num_cpus_;
};

// 全局函数 - 对应 RDMA 版本
std::string get_oob_ip();
