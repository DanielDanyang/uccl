// Devmem Validation Test
// Purpose: Verify single-process can use multiple channels on same NIC
// This is where original multi-process conflicts occurred

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
#include <thread>
#include <atomic>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

const size_t TRANSFER_SIZE = 16 * 1024 * 1024;  // 16 MB (smaller for quick test)
const int NUM_CHANNELS = 4;  // Test 4 channels on same NIC

std::atomic<int> g_success_count{0};
std::atomic<int> g_fail_count{0};

// Shared state for all channels
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

std::vector<ChannelState> g_channels(NUM_CHANNELS);

void run_channel_test(int gpu_id, int ch_id, int dev_id,
                      const char* role, const char* peer_ip) {
    printf("[GPU %d CH %d] Starting %s on dev %d\n", gpu_id, ch_id, role, dev_id);

    // Initialize CUDA Driver API
    CUdevice cuDev;
    CUcontext cuCtx;

    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "[GPU %d CH %d] cuInit failed\n", gpu_id, ch_id);
        g_fail_count++;
        return;
    }

    if (cuDeviceGet(&cuDev, gpu_id) != CUDA_SUCCESS) {
        fprintf(stderr, "[GPU %d CH %d] cuDeviceGet failed\n", gpu_id, ch_id);
        g_fail_count++;
        return;
    }

    if (cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS) {
        fprintf(stderr, "[GPU %d CH %d] cuDevicePrimaryCtxRetain failed\n", gpu_id, ch_id);
        g_fail_count++;
        return;
    }

    if (cuCtxSetCurrent(cuCtx) != CUDA_SUCCESS) {
        fprintf(stderr, "[GPU %d CH %d] cuCtxSetCurrent failed\n", gpu_id, ch_id);
        g_fail_count++;
        return;
    }

    CHECK_CUDA(cudaSetDevice(gpu_id));

    // Allocate GPU buffer with 4KB alignment (required by devmem-tcp)
    CUdeviceptr d_base = 0, d_aligned = 0;
    size_t alloc_size = TRANSFER_SIZE + 4096;

    if (cuMemAlloc(&d_base, alloc_size) != CUDA_SUCCESS) {
        fprintf(stderr, "[GPU %d CH %d] cuMemAlloc failed\n", gpu_id, ch_id);
        g_fail_count++;
        return;
    }

    // Align to 4KB boundary
    uintptr_t addr = static_cast<uintptr_t>(d_base);
    addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
    d_aligned = static_cast<CUdeviceptr>(addr);
    void* gpu_buf = reinterpret_cast<void*>(d_aligned);

    CHECK_CUDA(cudaMemset(gpu_buf, ch_id, TRANSFER_SIZE));  // Fill with channel ID

    // Bootstrap port
    int port = 20000 + gpu_id * 8 + ch_id;

    void* listen_comm = nullptr;
    void* send_comm = nullptr;
    void* recv_comm = nullptr;
    void* send_dev_handle = nullptr;
    void* recv_dev_handle = nullptr;
    void* mhandle = nullptr;

    try {
        if (strcmp(role, "server") == 0) {
            // Server: Listen and accept
            ncclNetHandle_v7 handle;
            memset(&handle, 0, sizeof(handle));

            if (tcpx_listen(dev_id, &handle, &listen_comm) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_listen failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            // Send handle to client via bootstrap
            int client_fd = -1;
            if (bootstrap_server_create(port, &client_fd) != 0) {
                fprintf(stderr, "[GPU %d CH %d] bootstrap_server_create failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            std::vector<ncclNetHandle_v7> handles = {handle};
            if (bootstrap_server_send_handles(client_fd, handles) != 0) {
                fprintf(stderr, "[GPU %d CH %d] bootstrap_server_send_handles failed\n", gpu_id, ch_id);
                close(client_fd);
                g_fail_count++;
                return;
            }
            close(client_fd);

            // Accept connection (may need retries)
            int accept_retries = 0;
            while (accept_retries < 100) {
                if (tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle) == 0 && recv_comm != nullptr) {
                    break;
                }
                usleep(10000);  // 10ms
                accept_retries++;
            }

            if (!recv_comm) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_accept_v5 failed after %d retries\n",
                        gpu_id, ch_id, accept_retries);
                g_fail_count++;
                return;
            }

            printf("[GPU %d CH %d] Channel accepted successfully (server)\n", gpu_id, ch_id);

            // Register memory (CRITICAL: must be after accept)
            if (tcpx_reg_mr(recv_comm, gpu_buf, TRANSFER_SIZE, NCCL_PTR_CUDA, &mhandle) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_reg_mr failed (ptr=%p, size=%zu)\n",
                        gpu_id, ch_id, gpu_buf, TRANSFER_SIZE);
                g_fail_count++;
                return;
            }

            printf("[GPU %d CH %d] Memory registered successfully (mhandle=%p)\n",
                   gpu_id, ch_id, mhandle);

            // Simple receive test
            void* request = nullptr;
            void* data_ptrs[1] = {gpu_buf};
            int sizes[1] = {(int)TRANSFER_SIZE};
            int tags[1] = {99};
            void* mhandles[1] = {mhandle};

            if (tcpx_irecv(recv_comm, 1, data_ptrs, sizes, tags, mhandles, &request) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_irecv failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            // Poll for completion
            int done = 0;
            int recv_size = 0;
            while (!done) {
                tcpx_test(request, &done, &recv_size);
                usleep(100);
            }

            printf("[GPU %d CH %d] Recv OK (%d bytes)\n", gpu_id, ch_id, recv_size);

            // Cleanup
            if (mhandle) tcpx_dereg_mr(recv_comm, mhandle);
            if (recv_comm) tcpx_close_recv(recv_comm);
            if (listen_comm) tcpx_close_listen(listen_comm);

        } else {
            // Client: Connect
            int server_fd = -1;
            if (bootstrap_client_connect(peer_ip, port, &server_fd) != 0) {
                fprintf(stderr, "[GPU %d CH %d] bootstrap_client_connect failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            std::vector<ncclNetHandle_v7> handles;
            if (bootstrap_client_recv_handles(server_fd, handles) != 0) {
                fprintf(stderr, "[GPU %d CH %d] bootstrap_client_recv_handles failed\n", gpu_id, ch_id);
                close(server_fd);
                g_fail_count++;
                return;
            }
            close(server_fd);

            if (handles.empty()) {
                fprintf(stderr, "[GPU %d CH %d] No handles received\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            // Connect
            if (tcpx_connect_v5(dev_id, &handles[0], &send_comm, &send_dev_handle) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_connect_v5 failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            printf("[GPU %d CH %d] Channel connected successfully (client)\n", gpu_id, ch_id);

            // Register memory (CRITICAL: must be after connect)
            if (tcpx_reg_mr(send_comm, gpu_buf, TRANSFER_SIZE, NCCL_PTR_CUDA, &mhandle) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_reg_mr failed (ptr=%p, size=%zu)\n",
                        gpu_id, ch_id, gpu_buf, TRANSFER_SIZE);
                g_fail_count++;
                return;
            }

            printf("[GPU %d CH %d] Memory registered successfully (mhandle=%p)\n",
                   gpu_id, ch_id, mhandle);

            // Simple send test
            void* request = nullptr;
            if (tcpx_isend(send_comm, gpu_buf, TRANSFER_SIZE, 99, mhandle, &request) != 0) {
                fprintf(stderr, "[GPU %d CH %d] tcpx_isend failed\n", gpu_id, ch_id);
                g_fail_count++;
                return;
            }

            // Poll for completion
            int done = 0;
            int send_size = 0;
            while (!done) {
                tcpx_test(request, &done, &send_size);
                usleep(100);
            }

            printf("[GPU %d CH %d] Send OK (%d bytes)\n", gpu_id, ch_id, send_size);

            // Cleanup
            if (mhandle) tcpx_dereg_mr(send_comm, mhandle);
            if (send_comm) tcpx_close_send(send_comm);
        }

        if (d_base) cuMemFree(d_base);
        printf("[GPU %d CH %d] Test PASSED\n", gpu_id, ch_id);
        g_success_count++;

    } catch (...) {
        fprintf(stderr, "[GPU %d CH %d] Exception caught\n", gpu_id, ch_id);
        if (d_base) cuMemFree(d_base);
        g_fail_count++;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server|client> <peer_ip> [gpu_id] [dev_id]\n", argv[0]);
        fprintf(stderr, "Example (server): %s server 0.0.0.0 0 0\n", argv[0]);
        fprintf(stderr, "Example (client): %s client 10.0.0.1 0 0\n", argv[0]);
        fprintf(stderr, "\ndev_id: TCPX device ID (0=eth1, 1=eth2, 2=eth3, 3=eth4)\n");
        return 1;
    }

    const char* role = argv[1];
    const char* peer_ip = argv[2];
    int gpu_id = (argc > 3) ? atoi(argv[3]) : 0;
    int dev_id = (argc > 4) ? atoi(argv[4]) : 0;

    // Load TCPX plugin - try multiple paths
    const char* plugin_path = getenv("NCCL_GPUDIRECTTCPX_PLUGIN_PATH");
    if (!plugin_path) {
        // Try /usr/local first (where you copied it)
        if (access("/usr/local/tcpx/lib64/libnccl-net.so", F_OK) == 0) {
            plugin_path = "/usr/local/tcpx/lib64/libnccl-net.so";
        } else {
            plugin_path = "/var/lib/tcpx/lib64/libnccl-net.so";
        }
    }

    printf("=== Devmem Validation Test ===\n");
    printf("Role: %s\n", role);
    printf("GPU: %d\n", gpu_id);
    printf("Dev: %d\n", dev_id);
    printf("Channels: %d\n", NUM_CHANNELS);
    printf("Plugin: %s\n", plugin_path);
    printf("Test: Multiple channels on SAME NIC in SINGLE process\n");
    printf("==============================\n\n");

    // Load plugin
    if (tcpx_load_plugin(plugin_path) != 0) {
        fprintf(stderr, "Failed to load TCPX plugin from %s\n", plugin_path);
        return 1;
    }

    int dev_count = tcpx_get_device_count();
    printf("TCPX devices: %d\n", dev_count);

    if (dev_id >= dev_count) {
        fprintf(stderr, "Invalid dev_id %d (max %d)\n", dev_id, dev_count - 1);
        return 1;
    }

    // Get device properties
    struct tcpx_net_properties props;
    if (tcpx_get_properties(dev_id, &props) == 0) {
        printf("Device %d: %s (speed=%d Mbps)\n\n", dev_id, props.name, props.speed);
    }

    // Launch all channels concurrently
    std::vector<std::thread> threads;

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        threads.emplace_back([=]() {
            run_channel_test(gpu_id, ch, dev_id, role, peer_ip);
        });
    }

    // Wait for all channels
    for (auto& t : threads) {
        t.join();
    }

    printf("\n=== TEST RESULTS ===\n");
    printf("Success: %d / %d\n", g_success_count.load(), NUM_CHANNELS);
    printf("Failed:  %d / %d\n", g_fail_count.load(), NUM_CHANNELS);

    if (g_success_count.load() == NUM_CHANNELS) {
        printf("\n=== ALL CHANNELS PASSED ===\n");
        printf("Result: Single-process CAN use multiple channels on same NIC\n");
        printf("Devmem conflicts: RESOLVED\n");
        printf("Proceed to Step 3: Full refactor\n");
        return 0;
    } else {
        printf("\n=== TEST FAILED ===\n");
        printf("Result: Single-process CANNOT use multiple channels on same NIC\n");
        printf("Devmem conflicts: STILL PRESENT\n");
        printf("Action: Contact Google, reconsider approach\n");
        return 1;
    }
}

