#include "../tcpx_interface.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

// TCPX Performance Test - measures actual send/recv latency and throughput
// Usage: test_performance server | test_performance client <server_ip>
//
// Improvements based on feedback:
// - Proper handling of partial send/recv for bootstrap connection
// - Complete polling until done=1 or hard timeout failure
// - Error checking for tcpx_test return codes
// - Prevents use-after-free by ensuring completion before cleanup

// NCCL network handle for connection establishment
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

// TCP port for handle exchange
#define TCPX_BOOTSTRAP_PORT 12347

// Test parameters
size_t const test_sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
int const num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
int const iterations = 50;
int const warmup_iterations = 5;

// Bootstrap connection helpers (from test_connection.cc)
int create_bootstrap_server() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) return -1;

  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);

  if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    close(listen_fd);
    return -1;
  }

  if (listen(listen_fd, 1) < 0) {
    close(listen_fd);
    return -1;
  }

  int client_fd = accept(listen_fd, nullptr, nullptr);
  close(listen_fd);
  return client_fd;
}

int connect_to_bootstrap_server(char const* server_ip) {
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) return -1;

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCPX_BOOTSTRAP_PORT);
  inet_aton(server_ip, &addr.sin_addr);

  int retry = 0;
  while (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    if (++retry > 10) {
      close(sock_fd);
      return -1;
    }
    sleep(1);
  }
  return sock_fd;
}

// Format size for display (ASCII only, no Unicode)
char const* format_size(size_t bytes) {
  static char buffer[32];
  if (bytes >= 1024 * 1024) {
    snprintf(buffer, sizeof(buffer), "%zuMB", bytes / (1024 * 1024));
  } else if (bytes >= 1024) {
    snprintf(buffer, sizeof(buffer), "%zuKB", bytes / 1024);
  } else {
    snprintf(buffer, sizeof(buffer), "%zuB", bytes);
  }
  return buffer;
}

// Calculate bandwidth in GB/s
double calculate_bandwidth(size_t bytes, double time_seconds) {
  return (double)bytes / time_seconds / (1024 * 1024 * 1024);
}

int main(int argc, char* argv[]) {
  printf("=== TCPX Performance Test ===\n");

  if (argc < 2) {
    printf("Usage: %s <server|client> [remote_ip]\n", argv[0]);
    printf("This test measures actual TCPX send/recv performance\n");
    return 1;
  }

  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);

  // Initialize TCPX
  printf("\n[Init] Initializing TCPX...\n");
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    printf("FAILED: No TCPX devices found\n");
    return 1;
  }
  printf("Found %d TCPX devices\n", device_count);

  int dev_id = 0;
  bool is_server = (strcmp(argv[1], "server") == 0);

  if (is_server) {
    printf("\n[Server] Starting performance test server...\n");

    // Create TCPX handle
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));
    void* listen_comm = nullptr;

    int rc = tcpx_listen(dev_id, &handle, &listen_comm);
    if (rc != 0) {
      printf("FAILED: tcpx_listen returned %d\n", rc);
      return 1;
    }
    printf("Listening on device %d\n", dev_id);

    // Send handle to client via bootstrap
    int bootstrap_fd = create_bootstrap_server();
    if (bootstrap_fd < 0) {
      printf("FAILED: Cannot create bootstrap server\n");
      return 1;
    }

    // Send handle with proper loop to handle partial sends
    size_t total_sent = 0;
    while (total_sent < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t sent = send(bootstrap_fd, handle.data + total_sent,
                          NCCL_NET_HANDLE_MAXSIZE - total_sent, 0);
      if (sent <= 0) {
        printf("FAILED: Cannot send handle to client (sent %zd bytes)\n",
               total_sent);
        close(bootstrap_fd);
        return 1;
      }
      total_sent += sent;
    }
    close(bootstrap_fd);

    // Accept connection
    void* recv_comm = nullptr;
    void* recv_dev_handle = nullptr;
    rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
    if (rc != 0) {
      printf("FAILED: tcpx_accept_v5 returned %d\n", rc);
      return 1;
    }
    printf("Connection accepted\n");

    // Performance test - receive data
    printf("\n[Performance] Starting receive tests...\n");
    printf("%-8s %-12s %-12s\n", "Size", "Bandwidth", "Latency");
    printf("%-8s %-12s %-12s\n", "----", "---------", "-------");

    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
      size_t msg_size = test_sizes[size_idx];

      // Allocate receive buffer
      char* recv_buffer = new char[msg_size];
      memset(recv_buffer, 0, msg_size);

      // Register memory (fall back to unregistered if needed)
      void* recv_mhandle = nullptr;
      int rc_reg = tcpx_reg_mr(recv_comm, recv_buffer, msg_size, NCCL_PTR_HOST,
                               &recv_mhandle);
      if (rc_reg != 0) {
        recv_mhandle = nullptr;  // Fall back to unregistered
      }

      double total_time = 0.0;
      int successful_transfers = 0;

      // Warmup + actual test iterations
      for (int i = 0; i < warmup_iterations + iterations; i++) {
        // Setup receive request
        void* recv_data[1] = {recv_buffer};
        int recv_sizes[1] = {(int)msg_size};
        int recv_tags[1] = {42};
        void* recv_mhandles[1] = {recv_mhandle};
        void* recv_request = nullptr;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Post receive
        int rc_recv = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                                 recv_mhandles, &recv_request);
        if (rc_recv != 0) continue;

        // Wait for completion - keep polling until done or error
        int done = 0, received_size = 0;
        int poll_count = 0;
        int const max_polls = 100000;  // Increase timeout
        while (!done && poll_count < max_polls) {
          int rc_test = tcpx_test(recv_request, &done, &received_size);
          if (rc_test != 0) {
            printf("FAILED: tcpx_test returned error %d\n", rc_test);
            break;
          }
          if (!done) {
            usleep(10);
            poll_count++;
          }
        }

        // Treat timeout as hard failure
        if (!done) {
          printf("FAILED: Receive timeout after %d polls for size %s\n",
                 max_polls, format_size(msg_size));
          // Don't continue with this size - abort the test
          if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
          delete[] recv_buffer;
          tcpx_close_recv(recv_comm);
          tcpx_close_listen(listen_comm);
          return 1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        if (done && i >= warmup_iterations) {
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          total_time += duration.count() / 1000000.0;  // Convert to seconds
          successful_transfers++;
        }
      }

      if (successful_transfers > 0) {
        double avg_time = total_time / successful_transfers;
        double bandwidth = calculate_bandwidth(msg_size, avg_time);
        printf("%-8s %-12.2f %-12.6f\n", format_size(msg_size), bandwidth,
               avg_time);
      } else {
        printf("%-8s %-12s %-12s\n", format_size(msg_size), "FAILED", "FAILED");
      }

      // Cleanup
      if (recv_mhandle) tcpx_dereg_mr(recv_comm, recv_mhandle);
      delete[] recv_buffer;
    }

    // Cleanup connections
    tcpx_close_recv(recv_comm);
    tcpx_close_listen(listen_comm);

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      printf("ERROR: Client mode requires remote IP\n");
      return 1;
    }

    printf("\n[Client] Connecting to server at %s\n", argv[2]);

    // Get handle from server via bootstrap
    int bootstrap_fd = connect_to_bootstrap_server(argv[2]);
    if (bootstrap_fd < 0) {
      printf("FAILED: Cannot connect to bootstrap server\n");
      return 1;
    }

    // Receive handle with proper loop to handle partial receives
    ncclNetHandle_v7 handle;
    size_t total_received = 0;
    while (total_received < NCCL_NET_HANDLE_MAXSIZE) {
      ssize_t received = recv(bootstrap_fd, handle.data + total_received,
                              NCCL_NET_HANDLE_MAXSIZE - total_received, 0);
      if (received <= 0) {
        printf(
            "FAILED: Cannot receive handle from server (received %zd bytes)\n",
            total_received);
        close(bootstrap_fd);
        return 1;
      }
      total_received += received;
    }
    close(bootstrap_fd);

    // Connect to server
    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;
    int rc = tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle);
    if (rc != 0) {
      printf("FAILED: tcpx_connect_v5 returned %d\n", rc);
      return 1;
    }
    printf("Connected to server\n");

    // Performance test - send data
    printf("\n[Performance] Starting send tests...\n");

    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
      size_t msg_size = test_sizes[size_idx];

      // Allocate send buffer
      char* send_buffer = new char[msg_size];
      memset(send_buffer, 0xAB, msg_size);  // Fill with test pattern

      // Register memory (fall back to unregistered if needed)
      void* send_mhandle = nullptr;
      int rc_reg = tcpx_reg_mr(send_comm, send_buffer, msg_size, NCCL_PTR_HOST,
                               &send_mhandle);
      if (rc_reg != 0) {
        send_mhandle = nullptr;  // Fall back to unregistered
      }

      // Warmup + actual test iterations
      for (int i = 0; i < warmup_iterations + iterations; i++) {
        void* send_request = nullptr;
        int send_tag = 42;

        // Post send
        int rc_send = tcpx_isend(send_comm, send_buffer, msg_size, send_tag,
                                 send_mhandle, &send_request);
        if (rc_send != 0) continue;

        // Wait for completion - keep polling until done or error
        int done = 0, sent_size = 0;
        int poll_count = 0;
        int const max_polls = 100000;  // Increase timeout
        while (!done && poll_count < max_polls) {
          int rc_test = tcpx_test(send_request, &done, &sent_size);
          if (rc_test != 0) {
            printf("FAILED: tcpx_test returned error %d\n", rc_test);
            break;
          }
          if (!done) {
            usleep(10);
            poll_count++;
          }
        }

        // Treat timeout as hard failure
        if (!done) {
          printf("FAILED: Send timeout after %d polls for size %s\n", max_polls,
                 format_size(msg_size));
          // Don't continue with this size - abort the test
          if (send_mhandle) tcpx_dereg_mr(send_comm, send_mhandle);
          delete[] send_buffer;
          tcpx_close_send(send_comm);
          return 1;
        }
      }

      // Cleanup
      if (send_mhandle) tcpx_dereg_mr(send_comm, send_mhandle);
      delete[] send_buffer;
    }

    printf("Send tests completed\n");

    // Cleanup connections
    tcpx_close_send(send_comm);
  }

  printf("\n=== TCPX Performance Test COMPLETED ===\n");
  return 0;
}
