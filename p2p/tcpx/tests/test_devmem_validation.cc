// Devmem Validation Test
// Purpose: Verify single-process can use multiple channels on same NIC
// This is where original multi-process conflicts occurred

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include "../include/tcpx_interface.h"
#include "../include/bootstrap.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

const size_t TRANSFER_SIZE = 64 * 1024 * 1024;  // 64 MB
const int NUM_CHANNELS = 4;  // Test 4 channels on same NIC

void run_channel_test(int gpu_id, int ch_id, const char* nic, 
                      const char* role, const char* peer_ip) {
    printf("[GPU %d CH %d] Starting %s on %s\n", gpu_id, ch_id, role, nic);
    
    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(gpu_id));
    
    // Allocate GPU buffer
    void* gpu_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_buf, TRANSFER_SIZE));
    
    // Initialize TCPX
    TcpxInterface tcpx;
    if (tcpx.init(nic) != 0) {
        fprintf(stderr, "[GPU %d CH %d] Failed to init TCPX on %s\n", 
                gpu_id, ch_id, nic);
        exit(1);
    }
    
    // Bootstrap
    int port = 20000 + gpu_id * 8 + ch_id;
    int sock_fd = -1;
    
    if (strcmp(role, "server") == 0) {
        sock_fd = bootstrap_server(port);
    } else {
        sock_fd = bootstrap_client(peer_ip, port);
    }
    
    if (sock_fd < 0) {
        fprintf(stderr, "[GPU %d CH %d] Bootstrap failed\n", gpu_id, ch_id);
        exit(1);
    }
    
    // Create TCPX channel
    TcpxChannel* channel = tcpx.create_channel(sock_fd, gpu_buf, TRANSFER_SIZE);
    if (!channel) {
        fprintf(stderr, "[GPU %d CH %d] Failed to create channel\n", gpu_id, ch_id);
        exit(1);
    }
    
    printf("[GPU %d CH %d] Channel created successfully on %s\n", 
           gpu_id, ch_id, nic);
    
    // Simple transfer test
    if (strcmp(role, "server") == 0) {
        // Receive
        if (tcpx.recv(channel, TRANSFER_SIZE) != 0) {
            fprintf(stderr, "[GPU %d CH %d] Recv failed\n", gpu_id, ch_id);
            exit(1);
        }
        printf("[GPU %d CH %d] Recv OK\n", gpu_id, ch_id);
    } else {
        // Send
        if (tcpx.send(channel, TRANSFER_SIZE) != 0) {
            fprintf(stderr, "[GPU %d CH %d] Send failed\n", gpu_id, ch_id);
            exit(1);
        }
        printf("[GPU %d CH %d] Send OK\n", gpu_id, ch_id);
    }
    
    // Cleanup
    tcpx.destroy_channel(channel);
    CHECK_CUDA(cudaFree(gpu_buf));
    
    printf("[GPU %d CH %d] Test PASSED\n", gpu_id, ch_id);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server|client> <peer_ip> [gpu_id] [nic]\n", argv[0]);
        fprintf(stderr, "Example (server): %s server 0.0.0.0 0 eth1\n", argv[0]);
        fprintf(stderr, "Example (client): %s client 10.0.0.1 0 eth1\n", argv[0]);
        return 1;
    }
    
    const char* role = argv[1];
    const char* peer_ip = argv[2];
    int gpu_id = (argc > 3) ? atoi(argv[3]) : 0;
    const char* nic = (argc > 4) ? argv[4] : "eth1";
    
    printf("=== Devmem Validation Test ===\n");
    printf("Role: %s\n", role);
    printf("GPU: %d\n", gpu_id);
    printf("NIC: %s\n", nic);
    printf("Channels: %d\n", NUM_CHANNELS);
    printf("Test: Multiple channels on SAME NIC in SINGLE process\n");
    printf("==============================\n\n");
    
    // Launch all channels concurrently
    std::vector<std::thread> threads;
    
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        threads.emplace_back([=]() {
            run_channel_test(gpu_id, ch, nic, role, peer_ip);
        });
    }
    
    // Wait for all channels
    for (auto& t : threads) {
        t.join();
    }
    
    printf("\n=== ALL CHANNELS PASSED ===\n");
    printf("Result: Single-process CAN use multiple channels on same NIC\n");
    printf("Devmem conflicts: RESOLVED\n");
    printf("Proceed to Step 3: Full refactor\n");
    
    return 0;
}

