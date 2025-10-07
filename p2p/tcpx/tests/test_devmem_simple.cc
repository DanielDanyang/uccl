// Simplified Devmem Validation Test
// Purpose: Verify single-process can use multiple channels on same NIC
// Approach: Sequential execution (like ChannelManager) to avoid race conditions

#include "../include/tcpx_interface.h"
#include "../include/tcpx_handles.h"
#include "../include/bootstrap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

const size_t TRANSFER_SIZE = 16 * 1024 * 1024;  // 16 MB
const int NUM_CHANNELS = 4;  // Test 4 channels on same NIC

struct ChannelState {
    void* listen_comm;
    void* send_comm;
    void* recv_comm;
    void* send_dev_handle;
    void* recv_dev_handle;
    void* mhandle;
    void* gpu_buf;
    CUdeviceptr d_base;
    ncclNetHandle_v7 handle;
    
    ChannelState() : listen_comm(nullptr), send_comm(nullptr), recv_comm(nullptr),
                     send_dev_handle(nullptr), recv_dev_handle(nullptr),
                     mhandle(nullptr), gpu_buf(nullptr), d_base(0) {
        memset(&handle, 0, sizeof(handle));
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server|client> <peer_ip> [gpu_id] [dev_id]\n", argv[0]);
        return 1;
    }
    
    const char* role = argv[1];
    const char* peer_ip = argv[2];
    int gpu_id = (argc > 3) ? atoi(argv[3]) : 0;
    int dev_id = (argc > 4) ? atoi(argv[4]) : 0;
    bool is_server = (strcmp(role, "server") == 0);
    
    // Load TCPX plugin
    const char* plugin_path = getenv("NCCL_GPUDIRECTTCPX_PLUGIN_PATH");
    if (!plugin_path) {
        if (access("/usr/local/tcpx/lib64/libnccl-net.so", F_OK) == 0) {
            plugin_path = "/usr/local/tcpx/lib64/libnccl-net.so";
        } else {
            plugin_path = "/var/lib/tcpx/lib64/libnccl-net.so";
        }
    }
    
    printf("=== Devmem Validation Test (Sequential) ===\n");
    printf("Role: %s\n", role);
    printf("GPU: %d, Dev: %d\n", gpu_id, dev_id);
    printf("Channels: %d (all on same NIC)\n", NUM_CHANNELS);
    printf("Plugin: %s\n", plugin_path);
    printf("==========================================\n\n");
    
    if (tcpx_load_plugin(plugin_path) != 0) {
        fprintf(stderr, "Failed to load TCPX plugin\n");
        return 1;
    }
    
    int dev_count = tcpx_get_device_count();
    printf("TCPX devices: %d\n\n", dev_count);
    
    // Initialize CUDA
    CUdevice cuDev;
    CUcontext cuCtx;
    
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialization failed\n");
        return 1;
    }
    
    CHECK_CUDA(cudaSetDevice(gpu_id));
    
    std::vector<ChannelState> channels(NUM_CHANNELS);
    
    // Allocate GPU buffers for all channels
    printf("Step 1: Allocating GPU buffers...\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        size_t alloc_size = TRANSFER_SIZE + 4096;
        if (cuMemAlloc(&channels[ch].d_base, alloc_size) != CUDA_SUCCESS) {
            fprintf(stderr, "CH %d: cuMemAlloc failed\n", ch);
            return 1;
        }
        
        // 4KB alignment
        uintptr_t addr = (uintptr_t)channels[ch].d_base;
        addr = (addr + 4095) & ~4095;
        channels[ch].gpu_buf = (void*)addr;
        
        CHECK_CUDA(cudaMemset(channels[ch].gpu_buf, ch, TRANSFER_SIZE));
        printf("  CH %d: buffer=%p\n", ch, channels[ch].gpu_buf);
    }
    
    if (is_server) {
        // ===== SERVER SIDE =====
        
        // Step 2: Listen on all channels
        printf("\nStep 2: Listening on all channels...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (tcpx_listen(dev_id, &channels[ch].handle, &channels[ch].listen_comm) != 0) {
                fprintf(stderr, "CH %d: tcpx_listen failed\n", ch);
                return 1;
            }
            printf("  CH %d: listen_comm=%p\n", ch, channels[ch].listen_comm);
        }
        
        // Step 3: Bootstrap - send handles to client
        printf("\nStep 3: Bootstrap handshake...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int port = 20000 + gpu_id * 8 + ch;
            int client_fd = -1;
            
            if (bootstrap_server_create(port, &client_fd) != 0) {
                fprintf(stderr, "CH %d: bootstrap_server_create failed\n", ch);
                return 1;
            }
            
            std::vector<ncclNetHandle_v7> handles = {channels[ch].handle};
            if (bootstrap_server_send_handles(client_fd, handles) != 0) {
                fprintf(stderr, "CH %d: bootstrap_server_send_handles failed\n", ch);
                close(client_fd);
                return 1;
            }
            close(client_fd);
            printf("  CH %d: sent handle\n", ch);
        }
        
        // Step 4: Accept all connections
        printf("\nStep 4: Accepting connections...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int retries = 0;
            while (retries < 100) {
                if (tcpx_accept_v5(channels[ch].listen_comm, &channels[ch].recv_comm, 
                                   &channels[ch].recv_dev_handle) == 0 
                    && channels[ch].recv_comm != nullptr) {
                    break;
                }
                usleep(10000);
                retries++;
            }
            
            if (!channels[ch].recv_comm) {
                fprintf(stderr, "CH %d: tcpx_accept_v5 failed after %d retries\n", ch, retries);
                return 1;
            }
            printf("  CH %d: recv_comm=%p\n", ch, channels[ch].recv_comm);
        }
        
        // Step 5: Register memory on all channels
        printf("\nStep 5: Registering memory...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (tcpx_reg_mr(channels[ch].recv_comm, channels[ch].gpu_buf, 
                           TRANSFER_SIZE, NCCL_PTR_CUDA, &channels[ch].mhandle) != 0) {
                fprintf(stderr, "CH %d: tcpx_reg_mr failed (ptr=%p, size=%zu)\n", 
                        ch, channels[ch].gpu_buf, TRANSFER_SIZE);
                return 1;
            }
            printf("  CH %d: mhandle=%p\n", ch, channels[ch].mhandle);
        }
        
        // Step 6: Receive on all channels
        printf("\nStep 6: Receiving data...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            void* request = nullptr;
            void* data_ptrs[1] = {channels[ch].gpu_buf};
            int sizes[1] = {(int)TRANSFER_SIZE};
            int tags[1] = {99 + ch};
            void* mhandles[1] = {channels[ch].mhandle};
            
            if (tcpx_irecv(channels[ch].recv_comm, 1, data_ptrs, sizes, tags, mhandles, &request) != 0) {
                fprintf(stderr, "CH %d: tcpx_irecv failed\n", ch);
                return 1;
            }
            
            int done = 0, recv_size = 0;
            while (!done) {
                tcpx_test(request, &done, &recv_size);
                usleep(100);
            }
            printf("  CH %d: received %d bytes\n", ch, recv_size);
        }
        
        printf("\n=== ALL CHANNELS PASSED (SERVER) ===\n");
        
    } else {
        // ===== CLIENT SIDE =====
        
        // Step 2: Bootstrap - receive handles from server
        printf("\nStep 2: Bootstrap handshake...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int port = 20000 + gpu_id * 8 + ch;
            int server_fd = -1;
            
            if (bootstrap_client_connect(peer_ip, port, &server_fd) != 0) {
                fprintf(stderr, "CH %d: bootstrap_client_connect failed\n", ch);
                return 1;
            }
            
            std::vector<ncclNetHandle_v7> handles;
            if (bootstrap_client_recv_handles(server_fd, handles) != 0) {
                fprintf(stderr, "CH %d: bootstrap_client_recv_handles failed\n", ch);
                close(server_fd);
                return 1;
            }
            close(server_fd);
            
            if (handles.empty()) {
                fprintf(stderr, "CH %d: no handles received\n", ch);
                return 1;
            }
            channels[ch].handle = handles[0];
            printf("  CH %d: received handle\n", ch);
        }
        
        // Step 3: Connect all channels
        printf("\nStep 3: Connecting to server...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (tcpx_connect_v5(dev_id, &channels[ch].handle, &channels[ch].send_comm, 
                               &channels[ch].send_dev_handle) != 0) {
                fprintf(stderr, "CH %d: tcpx_connect_v5 failed\n", ch);
                return 1;
            }
            printf("  CH %d: send_comm=%p\n", ch, channels[ch].send_comm);
        }
        
        // Step 4: Register memory on all channels
        printf("\nStep 4: Registering memory...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (tcpx_reg_mr(channels[ch].send_comm, channels[ch].gpu_buf, 
                           TRANSFER_SIZE, NCCL_PTR_CUDA, &channels[ch].mhandle) != 0) {
                fprintf(stderr, "CH %d: tcpx_reg_mr failed (ptr=%p, size=%zu)\n", 
                        ch, channels[ch].gpu_buf, TRANSFER_SIZE);
                return 1;
            }
            printf("  CH %d: mhandle=%p\n", ch, channels[ch].mhandle);
        }
        
        // Step 5: Send on all channels
        printf("\nStep 5: Sending data...\n");
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            void* request = nullptr;
            if (tcpx_isend(channels[ch].send_comm, channels[ch].gpu_buf, 
                          TRANSFER_SIZE, 99 + ch, channels[ch].mhandle, &request) != 0) {
                fprintf(stderr, "CH %d: tcpx_isend failed\n", ch);
                return 1;
            }
            
            int done = 0, send_size = 0;
            while (!done) {
                tcpx_test(request, &done, &send_size);
                usleep(100);
            }
            printf("  CH %d: sent %d bytes\n", ch, send_size);
        }
        
        printf("\n=== ALL CHANNELS PASSED (CLIENT) ===\n");
    }
    
    printf("\nResult: Single-process CAN use %d channels on same NIC\n", NUM_CHANNELS);
    printf("Devmem conflicts: RESOLVED\n");
    printf("Proceed to Step 2: Full refactor\n");
    
    return 0;
}

