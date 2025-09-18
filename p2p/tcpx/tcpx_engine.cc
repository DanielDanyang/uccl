#include "tcpx_engine.h"
#include <stdio.h>

// 全局函数实现
std::string get_oob_ip() {
    return "127.0.0.1";  // 简化实现
}

// TcpxEndpoint 实现
TcpxEndpoint::TcpxEndpoint(uint32_t local_gpu_idx, uint32_t num_cpus) 
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
    printf("[TCPX] Creating TcpxEndpoint with GPU %u, CPUs %u\n", local_gpu_idx, num_cpus);
}

TcpxEndpoint::~TcpxEndpoint() {
    printf("[TCPX] Destroying TcpxEndpoint\n");
}

std::vector<uint8_t> TcpxEndpoint::get_metadata() {
    printf("[TCPX] Generating metadata\n");
    
    // 简化的元数据：只返回 10 字节
    std::vector<uint8_t> metadata(10, 0x42);
    
    printf("[TCPX] Metadata generated: %zu bytes\n", metadata.size());
    return metadata;
}

std::string TcpxEndpoint::get_oob_ip() {
    return ::get_oob_ip();
}

int TcpxEndpoint::get_device_count() {
    printf("[TCPX] Getting device count\n");
    int count = tcpx_get_device_count();
    printf("[TCPX] Device count: %d\n", count);
    return count;
}
