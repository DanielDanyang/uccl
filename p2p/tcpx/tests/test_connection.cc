#include "../tcpx_interface.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda.h>

// NCCL network handle - used to exchange connection details
// According to the NCCL spec, the handle is typically 128 bytes (extra space
// avoids overflow)
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

// TCP port for handle exchange (similar to RDMA's bootstrap)
#define TCPX_BOOTSTRAP_PORT 12345

// Server: create bootstrap socket and wait for client to connect
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "Failed to create bootstrap socket" << std::endl;
    return -1;
  }

  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);

  if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to bind bootstrap socket" << std::endl;
    close(listen_fd);
    return -1;
  }

  if (listen(listen_fd, 1) < 0) {
    std::cerr << "Failed to listen on bootstrap socket" << std::endl;
    close(listen_fd);
    return -1;
  }

  std::cout << "Bootstrap server listening on port " << TCPX_BOOTSTRAP_PORT
            << std::endl;

  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);

  if (client_fd < 0) {
    std::cerr << "Failed to accept bootstrap connection" << std::endl;
    return -1;
  }

  return client_fd;
}

// Client: connect to server's bootstrap socket
int connect_to_bootstrap_server(char const* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    std::cerr << "Failed to create bootstrap socket" << std::endl;
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);
  inet_aton(server_ip, &addr.sin_addr);

  // Retry connection with backoff
  int retry = 0;
  while (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    if (++retry > 10) {
      std::cerr << "Failed to connect to bootstrap server after " << retry
                << " retries" << std::endl;
      close(sock_fd);
      return -1;
    }
    std::cout << "Retrying bootstrap connection... (" << retry << "/10)"
              << std::endl;
    sleep(1);
  }

  std::cout << "Connected to bootstrap server at " << server_ip << std::endl;
  return sock_fd;
}

int main(int argc, char* argv[]) {
  std::cout << "=== TCPX Connection Test ===" << std::endl;
  std::cout << "Note: this is a simplified connectivity test" << std::endl;
  std::cout << "A real TCPX setup must exchange handles via out-of-band "
               "communication"
            << std::endl;
  std::cout << "This run only verifies that the API calls behave as expected"
            << std::endl;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <server|client> [remote_ip]"
              << std::endl;
    std::cout << "  server: Start as server (listener)" << std::endl;
    std::cout << "  client <ip>: Connect to server at <ip>" << std::endl;
    return 1;
  }

  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);
  // Respect external env for RX memory import; do not override here.

  // Initialize TCPX
  std::cout << "\n[Step 1] Initializing TCPX..." << std::endl;
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "�?FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  std::cout << "�?SUCCESS: Found " << device_count << " TCPX devices"
            << std::endl;

  // Use device 0 for testing
  int dev_id = 0;
  std::cout << "Using TCPX device " << dev_id << std::endl;

  bool is_server = (strcmp(argv[1], "server") == 0);

  if (is_server) {
    std::cout << "\n[Step 2] Starting as SERVER..." << std::endl;

    // Create connection handle for listening
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));

    void* listen_comm = nullptr;
    std::cout << "Attempting to listen on device " << dev_id << "..."
              << std::endl;

    int rc = tcpx_listen(dev_id, &handle, &listen_comm);
    if (rc != 0) {
      std::cout << "�?FAILED: tcpx_listen returned " << rc << std::endl;
      return 1;
    }
    std::cout << "�?SUCCESS: Listening on device " << dev_id << std::endl;
    std::cout << "Listen comm: " << listen_comm << std::endl;

    // Keep TCPX handle opaque - don't parse or modify it
    std::cout << "\n[Step 3] TCPX handle ready for transmission..."
              << std::endl;

    // Print handle data for debugging only
    std::cout << "TCPX handle data (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Don't modify the handle - pass it as-is to the client

    // Create bootstrap server to send handle to client
    std::cout << "Creating bootstrap server for handle exchange..."
              << std::endl;
    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: Cannot create bootstrap server" << std::endl;
      return 1;
    }

    // Send handle to client via bootstrap connection
    std::cout << "Sending TCPX handle to client..." << std::endl;

    // Small delay to ensure TCPX is fully ready
    usleep(100000);  // 100ms

    ssize_t sent = send(bootstrap_fd, handle.data, NCCL_NET_HANDLE_MAXSIZE, 0);
    if (sent != NCCL_NET_HANDLE_MAXSIZE) {
      std::cout << "�?FAILED: Cannot send handle to client (sent " << sent
                << " bytes)" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "�?SUCCESS: Handle sent to client (" << sent << " bytes)"
              << std::endl;

    close(bootstrap_fd);

    // Wait for client to process handle and initiate connection
    std::cout << "Waiting for client to connect..." << std::endl;
    sleep(2);  // Give client time to process handle and connect

    // Accept connection with retry loop
    void* recv_comm = nullptr;
    // TCPX plugin expects caller-provided storage for devNetDeviceHandle
    // (see tcpxAccept_v5 -> tcpxGetDeviceHandle). Pre-allocate a buffer.
    alignas(16) unsigned char recv_dev_handle_storage[512] = {0};
    void* recv_dev_handle = recv_dev_handle_storage;

    std::cout << "Calling tcpx_accept_v5..." << std::endl;

    // Retry accept until we get a valid connection
    int accept_retries = 0;
    int const max_accept_retries = 10;

    while (accept_retries < max_accept_retries) {
      rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
      if (rc != 0) {
        std::cout << "�?FAILED: tcpx_accept_v5 returned " << rc << std::endl;
        tcpx_close_listen(listen_comm);
        return 1;
      }

      if (recv_comm != nullptr) {
        std::cout << "�?SUCCESS: Connection accepted!" << std::endl;
        std::cout << "Recv comm: " << recv_comm << std::endl;
        std::cout << "Recv dev handle: " << recv_dev_handle << std::endl;
        break;
      }

      accept_retries++;
      std::cout << "Accept returned null comm, retrying... (" << accept_retries
                << "/" << max_accept_retries << ")" << std::endl;
      sleep(1);
    }

    if (recv_comm == nullptr) {
      std::cout << "�?FAILED: No valid connection after " << max_accept_retries
                << " retries" << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    // Test data transfer - receive data from client
    std::cout << "\n[Step 4] Testing data transfer (receive)..." << std::endl;

    // Print current environment variables for debugging
    std::cout << "Environment check:" << std::endl;
    const char* socket_if = getenv("NCCL_SOCKET_IFNAME");
    const char* tcpx_if = getenv("NCCL_GPUDIRECTTCPX_SOCKET_IFNAME");
    const char* rxmem_import = getenv("NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT");
    std::cout << "  NCCL_SOCKET_IFNAME=" << (socket_if ? socket_if : "not set") << std::endl;
    std::cout << "  NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=" << (tcpx_if ? tcpx_if : "not set") << std::endl;
    std::cout << "  NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=" << (rxmem_import ? rxmem_import : "not set") << std::endl;

    // Try GPU receive first (NCCL_PTR_CUDA). Fallback to host if CUDA fails.
    const char* forceHost = std::getenv("UCCL_TCPX_FORCE_HOST_RECV");
    bool want_cuda = !(forceHost && *forceHost == '1');
    bool did_cuda = false;
    CUdevice cuDev; CUcontext cuCtx;
    CUdeviceptr d_base = 0, d_aligned = 0;
    void* recv_mhandle = nullptr;
    void* recv_request = nullptr;
    int done = 0, received_size = 0;

    if (want_cuda) {
      CUresult curet = cuInit(0);
      if (curet == CUDA_SUCCESS) curet = cuDeviceGet(&cuDev, dev_id);
      if (curet == CUDA_SUCCESS) curet = cuDevicePrimaryCtxRetain(&cuCtx, cuDev);
      if (curet == CUDA_SUCCESS) curet = cuCtxSetCurrent(cuCtx);
      if (curet == CUDA_SUCCESS) curet = cuMemAlloc(&d_base, 1024 + 4096);
      if (curet == CUDA_SUCCESS) {
        // 4KB-align the device pointer as required by plugin's DMABUF path.
        uintptr_t addr = static_cast<uintptr_t>(d_base);
        addr = (addr + 4095) & ~static_cast<uintptr_t>(4095);
        d_aligned = static_cast<CUdeviceptr>(addr);

        int rc_reg = tcpx_reg_mr(recv_comm, reinterpret_cast<void*>(d_aligned), 1024,
                                  NCCL_PTR_CUDA, &recv_mhandle);
        if (rc_reg == 0) {
          void* recv_data[1] = { reinterpret_cast<void*>(d_aligned) };
          int   recv_sizes[1] = { 1024 };
          int   recv_tags[1]  = { 42 };
          void* recv_mhandles[1] = { recv_mhandle };
          std::cout << "Using CUDA recv buffer ptr="
                    << reinterpret_cast<void*>(d_aligned)
                    << ", size=1024" << std::endl;

          int rc_recv = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                                   recv_mhandles, &recv_request);
          if (rc_recv == 0) {
            for (int i = 0; i < 1000 && !done; i++) {
              int rc_test = tcpx_test(recv_request, &done, &received_size);
              if (rc_test != 0) { std::cout << "tcpx_test rc=" << rc_test << std::endl; break; }
              if (!done) usleep(1000);
            }
            if (done) {
              std::vector<char> host_out(1024, 0);
              cuMemcpyDtoH(host_out.data(), d_aligned, 1024);
              std::cout << "SUCCESS: Received " << received_size << " bytes" << std::endl;
              std::cout << "Data: '" << host_out.data() << "'" << std::endl;
              did_cuda = true;
            }
          }
        }
      }
      // Cleanup CUDA resources if used/allocated
      if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle), recv_mhandle = nullptr;
      if (d_base) cuMemFree(d_base), d_base = 0, d_aligned = 0;
      if (curet == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(cuDev);
    }

    if (!did_cuda) {
// ===== Allocate receive buffer with better alignment and error handling =====
size_t const buffer_size = 1024;
size_t const page_size = 4096;  // Force 4KB alignment for TCPX
void*   recv_mmap   = MAP_FAILED;
void*   recv_aligned = nullptr;
char*   recv_buffer = nullptr;
bool    used_mmap   = false;
bool    used_posix  = false;

// 1) Try mmap with explicit alignment
size_t aligned_size = (buffer_size + page_size - 1) & ~(page_size - 1);
recv_mmap = mmap(nullptr, aligned_size,
                 PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS
#ifdef MAP_POPULATE
                 | MAP_POPULATE
#endif
                 , -1, 0);
if (recv_mmap != MAP_FAILED) {
  used_mmap   = true;
  recv_buffer = static_cast<char*>(recv_mmap);
  memset(recv_buffer, 0, buffer_size);
  printf("Using mmap buffer: addr=%p, size=%zu (aligned to %zu)\n",
         recv_buffer, buffer_size, aligned_size);
  if (mlock(recv_buffer, buffer_size) != 0) {
    perror("mlock failed, continuing without lock");
  }
} else if (posix_memalign(&recv_aligned, page_size, aligned_size) == 0) {
  used_posix  = true;
  recv_buffer = static_cast<char*>(recv_aligned);
  memset(recv_buffer, 0, buffer_size);
  printf("Using posix_memalign buffer: addr=%p, size=%zu (aligned to %zu)\n",
         recv_buffer, buffer_size, page_size);
  if (mlock(recv_buffer, buffer_size) != 0) {
    perror("mlock failed, continuing without lock");
  }
} else {
  printf("WARNING: Using unaligned buffer allocation\n");
  recv_buffer = new char[buffer_size];
  memset(recv_buffer, 0, buffer_size);
}

printf("Recv buffer addr=%p, size=%zu\n", recv_buffer, buffer_size);

// ===== Register MR =====
recv_mhandle = nullptr;
int rc_reg = tcpx_reg_mr(recv_comm, recv_buffer, buffer_size, NCCL_PTR_HOST, &recv_mhandle);
if (rc_reg != 0) {
  printf("WARNING: tcpx_reg_mr failed rc=%d, continue unregistered\n", rc_reg);
  recv_mhandle = nullptr;
} else {
  printf("Memory registered for receive, mhandle=%p\n", recv_mhandle);
}

// ===== Post receive =====
void* recv_data[1]    = { recv_buffer };
int   recv_sizes[1]   = { buffer_size };
int   recv_tags[1]    = { 42 };
void* recv_mhandles[1]= { recv_mhandle };
recv_request    = nullptr;

printf("  recv_data[0]=%p, mhandle=%p\n", recv_buffer, recv_mhandle);

int rc_recv = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request);
if (rc_recv != 0) {
  printf("FAILED: tcpx_irecv rc=%d\n", rc_recv);
} else {
  printf("Receive request posted, request=%p\n", recv_request);
  // ===== Poll =====
  done = 0; received_size = 0;
  for (int i = 0; i < 1000 && !done; i++) {
    int rc_test = tcpx_test(recv_request, &done, &received_size);
    if (rc_test != 0) { printf("tcpx_test rc=%d\n", rc_test); break; }
    if (!done) usleep(1000);
  }
  if (done) {
    printf("SUCCESS: Received %d bytes\n", received_size);
    printf("Data: '%s'\n", recv_buffer);
  } else {
    printf("TIMEOUT: no data\n");
  }
}

if (recv_mhandle) {
  tcpx_dereg_mr(recv_comm, recv_mhandle);
}
    } // end host fallback
if (used_mmap) {
  munlock(recv_buffer, buffer_size);
  munmap(recv_buffer, buffer_size);
} else if (used_posix) {
  munlock(recv_buffer, buffer_size);
  free(recv_aligned);
} else {
  delete[] recv_buffer;
}


    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      std::cout << "�?ERROR: Client mode requires remote IP" << std::endl;
      return 1;
    }

    std::cout << "\n[Step 2] Starting as CLIENT..." << std::endl;
    std::cout << "Connecting to server at " << argv[2] << std::endl;

    // Connect to server's bootstrap socket to receive handle
    std::cout << "\n[Step 3] Connecting to server for handle exchange..."
              << std::endl;
    int bootstrap_fd = connect_to_bootstrap_server(argv[2]);
    if (bootstrap_fd < 0) {
      std::cout << "�?FAILED: Cannot connect to bootstrap server" << std::endl;
      return 1;
    }

    // Receive handle from server via bootstrap connection
    std::cout << "Receiving TCPX handle from server..." << std::endl;
    ncclNetHandle_v7 handle;
    ssize_t received =
        recv(bootstrap_fd, handle.data, NCCL_NET_HANDLE_MAXSIZE, 0);
    if (received != NCCL_NET_HANDLE_MAXSIZE) {
      std::cout << "�?FAILED: Cannot receive handle from server (received "
                << received << " bytes)" << std::endl;
      close(bootstrap_fd);
      return 1;
    }
    std::cout << "�?SUCCESS: Handle received from server (" << received
              << " bytes)" << std::endl;

    // Debug: Print received handle data
    std::cout << "Received TCPX handle data (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    close(bootstrap_fd);

    // Use TCPX handle as-is (opaque data)
    std::cout << "Using TCPX handle for connection (opaque data)..."
              << std::endl;

    // Print handle for debugging
    std::cout << "Received handle (first 32 bytes):" << std::endl;
    for (int i = 0; i < 32; i++) {
      printf("%02x ", (unsigned char)handle.data[i]);
      if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Small delay to ensure server is ready for accept
    std::cout << "Preparing to connect..." << std::endl;
    sleep(1);

    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;

    std::cout << "Attempting TCPX connection..." << std::endl;
    int rc = tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle);
    if (rc != 0) {
      std::cout << "�?FAILED: tcpx_connect_v5 returned " << rc << std::endl;
      return 1;
    }

    std::cout << "�?SUCCESS: Connected to server!" << std::endl;
    std::cout << "Send comm: " << send_comm << std::endl;
    std::cout << "Send dev handle: " << send_dev_handle << std::endl;

    // Wait a bit to ensure connection is fully established
    std::cout << "Waiting for connection to stabilize..." << std::endl;
    sleep(2);

    // Test data transfer - send data to server
    std::cout << "\n[Step 4] Testing data transfer (send)..." << std::endl;

    // Prepare test data
    char const* test_message = "Hello from TCPX client!";
    int const message_len =
        strlen(test_message) + 1;  // Include null terminator
    char* send_buffer = new char[message_len];
    strcpy(send_buffer, test_message);

    std::cout << "Sending message: '" << test_message << "' (" << message_len
              << " bytes)" << std::endl;

    // Register memory for TCPX
    void* send_mhandle = nullptr;
    int rc_reg = tcpx_reg_mr(send_comm, send_buffer, message_len, NCCL_PTR_HOST,
                             &send_mhandle);
    if (rc_reg != 0) {
      std::cout << "�?WARNING: tcpx_reg_mr failed with rc=" << rc_reg
                << std::endl;
      std::cout << "Continuing without memory registration..." << std::endl;
      send_mhandle = nullptr;
    } else {
      std::cout << "�?Memory registered for send, mhandle=" << send_mhandle
                << std::endl;
    }

    // Send data
    void* send_request = nullptr;
    int send_tag = 42;  // Match server's receive tag

    std::cout << "Posting send request..." << std::endl;
    int rc_send = tcpx_isend(send_comm, send_buffer, message_len, send_tag,
                             send_mhandle, &send_request);
    if (rc_send != 0) {
      std::cout << "�?FAILED: tcpx_isend returned " << rc_send << std::endl;
    } else {
      std::cout << "�?Send request posted, request=" << send_request
                << std::endl;

      // Wait for completion
      std::cout << "Waiting for send completion..." << std::endl;
      int done = 0, sent_size = 0;
      int max_polls = 1000;
      for (int i = 0; i < max_polls && !done; i++) {
        int rc_test = tcpx_test(send_request, &done, &sent_size);
        if (rc_test != 0) {
          std::cout << "�?tcpx_test failed with rc=" << rc_test << std::endl;
          break;
        }
        if (!done) {
          usleep(1000);  // 1ms delay
        }
      }

      if (done) {
        std::cout << "�?SUCCESS: Sent " << sent_size << " bytes" << std::endl;
      } else {
        std::cout << "�?TIMEOUT: Send not completed after " << max_polls
                  << " polls" << std::endl;
      }
    }

    // Cleanup
    if (send_mhandle) {
      tcpx_dereg_mr(send_comm, send_mhandle);
    }
    delete[] send_buffer;

    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else {
    std::cout << "�?ERROR: Invalid mode. Use 'server' or 'client'" << std::endl;
    return 1;
  }

  std::cout << "\n=== TCPX Connection Test COMPLETED ===" << std::endl;
  return 0;
}
