#include "../tcpx_interface.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>

// NCCL network handle - used to exchange connection details
// According to the NCCL spec, the handle is typically 128 bytes (extra space
// avoids overflow)
#define NCCL_NET_HANDLE_MAXSIZE 128
struct ncclNetHandle_v7 {
  char data[NCCL_NET_HANDLE_MAXSIZE];
};

// Handle file path - used to share connection info between nodes
char const* HANDLE_FILE = "/tmp/tcpx_handle.dat";

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

  // Initialize TCPX
  std::cout << "\n[Step 1] Initializing TCPX..." << std::endl;
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "✗ FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  std::cout << "✓ SUCCESS: Found " << device_count << " TCPX devices"
            << std::endl;

  // Use device 0 for testing
  int dev_id = 0;
  std::cout << "Using TCPX device " << dev_id << std::endl;

  bool is_server = (strcmp(argv[1], "server") == 0);

  if (is_server) {
    std::cout << "\n[Step 2] Starting as SERVER..." << std::endl;

    // Create connection handle for listening
    // For listen, the handle is typically filled by the plugin
    ncclNetHandle_v7 handle;
    memset(&handle, 0, sizeof(handle));

    void* listen_comm = nullptr;
    std::cout << "Attempting to listen on device " << dev_id << "..."
              << std::endl;

    int rc = tcpx_listen(dev_id, &handle, &listen_comm);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_listen returned " << rc << std::endl;
      return 1;
    }
    std::cout << "✓ SUCCESS: Listening on device " << dev_id << std::endl;
    std::cout << "Listen comm: " << listen_comm << std::endl;

    // Save handle to file for client to use
    std::cout << "\n[Step 3] Saving connection handle to file..." << std::endl;
    std::ofstream handle_file(HANDLE_FILE, std::ios::binary);
    if (!handle_file) {
      std::cout << "✗ FAILED: Cannot create handle file " << HANDLE_FILE
                << std::endl;
      return 1;
    }

    handle_file.write(handle.data, NCCL_NET_HANDLE_MAXSIZE);
    handle_file.close();
    std::cout << "✓ SUCCESS: Handle saved to " << HANDLE_FILE << std::endl;

    std::cout << "\nWaiting for client connection..." << std::endl;
    std::cout << "Run client with: " << argv[0] << " client <this_server_ip>"
              << std::endl;
    std::cout << "Press Enter when client is ready..." << std::endl;
    std::cin.get();  // Wait for user input

    // Accept connection
    void* recv_comm = nullptr;
    void* recv_dev_handle = nullptr;

    std::cout << "Calling tcpx_accept_v5..." << std::endl;
    rc = tcpx_accept_v5(listen_comm, &recv_comm, &recv_dev_handle);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_accept_v5 returned " << rc << std::endl;
      tcpx_close_listen(listen_comm);
      return 1;
    }

    std::cout << "✓ SUCCESS: Connection accepted!" << std::endl;
    std::cout << "Recv comm: " << recv_comm << std::endl;
    std::cout << "Recv dev handle: " << recv_dev_handle << std::endl;

    // Test data transfer - receive data from client
    std::cout << "\n[Step 4] Testing data transfer (receive)..." << std::endl;

    // Allocate receive buffer
    int const buffer_size = 1024;
    char* recv_buffer = new char[buffer_size];
    memset(recv_buffer, 0, buffer_size);

    // Register memory for TCPX
    void* recv_mhandle = nullptr;
    int rc_reg =
        tcpx_reg_mr(recv_comm, recv_buffer, buffer_size, 0, &recv_mhandle);
    if (rc_reg != 0) {
      std::cout << "✗ WARNING: tcpx_reg_mr failed with rc=" << rc_reg
                << std::endl;
      std::cout << "Continuing without memory registration..." << std::endl;
      recv_mhandle = nullptr;
    } else {
      std::cout << "✓ Memory registered for receive, mhandle=" << recv_mhandle
                << std::endl;
    }

    // Setup receive request
    void* recv_data[1] = {recv_buffer};
    int recv_sizes[1] = {buffer_size};
    int recv_tags[1] = {42};  // Match client's send tag
    void* recv_mhandles[1] = {recv_mhandle};
    void* recv_request = nullptr;

    std::cout << "Posting receive request..." << std::endl;
    int rc_recv = tcpx_irecv(recv_comm, 1, recv_data, recv_sizes, recv_tags,
                             recv_mhandles, &recv_request);
    if (rc_recv != 0) {
      std::cout << "✗ FAILED: tcpx_irecv returned " << rc_recv << std::endl;
    } else {
      std::cout << "✓ Receive request posted, request=" << recv_request
                << std::endl;

      // Wait for completion
      std::cout << "Waiting for data from client..." << std::endl;
      int done = 0, received_size = 0;
      int max_polls = 1000;
      for (int i = 0; i < max_polls && !done; i++) {
        int rc_test = tcpx_test(recv_request, &done, &received_size);
        if (rc_test != 0) {
          std::cout << "✗ tcpx_test failed with rc=" << rc_test << std::endl;
          break;
        }
        if (!done) {
          usleep(1000);  // 1ms delay
        }
      }

      if (done) {
        std::cout << "✓ SUCCESS: Received " << received_size << " bytes"
                  << std::endl;
        std::cout << "Data: '" << recv_buffer << "'" << std::endl;
      } else {
        std::cout << "✗ TIMEOUT: No data received after " << max_polls
                  << " polls" << std::endl;
      }
    }

    // Cleanup
    if (recv_mhandle) {
      tcpx_dereg_mr(recv_comm, recv_mhandle);
    }
    delete[] recv_buffer;
    unlink(HANDLE_FILE);

    std::cout << "TODO: Implement proper cleanup for TCPX connections"
              << std::endl;

  } else if (strcmp(argv[1], "client") == 0) {
    if (argc < 3) {
      std::cout << "✗ ERROR: Client mode requires remote IP" << std::endl;
      return 1;
    }

    std::cout << "\n[Step 2] Starting as CLIENT..." << std::endl;
    std::cout << "Connecting to server at " << argv[2] << std::endl;

    // Load connection handle from file
    std::cout << "\n[Step 3] Loading connection handle from file..."
              << std::endl;
    std::ifstream handle_file(HANDLE_FILE, std::ios::binary);
    if (!handle_file) {
      std::cout << "✗ FAILED: Cannot open handle file " << HANDLE_FILE
                << std::endl;
      std::cout << "Make sure server has created the handle file first!"
                << std::endl;
      std::cout << "If nodes don't share filesystem, copy the file manually:"
                << std::endl;
      std::cout << "  scp " << HANDLE_FILE << " user@" << argv[2] << ":"
                << HANDLE_FILE << std::endl;
      return 1;
    }

    ncclNetHandle_v7 handle;
    handle_file.read(handle.data, NCCL_NET_HANDLE_MAXSIZE);
    handle_file.close();
    std::cout << "✓ SUCCESS: Handle loaded from " << HANDLE_FILE << std::endl;

    void* send_comm = nullptr;
    void* send_dev_handle = nullptr;

    std::cout << "Attempting to connect to " << argv[2] << "..." << std::endl;
    int rc = tcpx_connect_v5(dev_id, &handle, &send_comm, &send_dev_handle);
    if (rc != 0) {
      std::cout << "✗ FAILED: tcpx_connect_v5 returned " << rc << std::endl;
      return 1;
    }

    std::cout << "✓ SUCCESS: Connected to server!" << std::endl;
    std::cout << "Send comm: " << send_comm << std::endl;
    std::cout << "Send dev handle: " << send_dev_handle << std::endl;

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
    int rc_reg =
        tcpx_reg_mr(send_comm, send_buffer, message_len, 0, &send_mhandle);
    if (rc_reg != 0) {
      std::cout << "✗ WARNING: tcpx_reg_mr failed with rc=" << rc_reg
                << std::endl;
      std::cout << "Continuing without memory registration..." << std::endl;
      send_mhandle = nullptr;
    } else {
      std::cout << "✓ Memory registered for send, mhandle=" << send_mhandle
                << std::endl;
    }

    // Send data
    void* send_request = nullptr;
    int send_tag = 42;  // Match server's receive tag

    std::cout << "Posting send request..." << std::endl;
    int rc_send = tcpx_isend(send_comm, send_buffer, message_len, send_tag,
                             send_mhandle, &send_request);
    if (rc_send != 0) {
      std::cout << "✗ FAILED: tcpx_isend returned " << rc_send << std::endl;
    } else {
      std::cout << "✓ Send request posted, request=" << send_request
                << std::endl;

      // Wait for completion
      std::cout << "Waiting for send completion..." << std::endl;
      int done = 0, sent_size = 0;
      int max_polls = 1000;
      for (int i = 0; i < max_polls && !done; i++) {
        int rc_test = tcpx_test(send_request, &done, &sent_size);
        if (rc_test != 0) {
          std::cout << "✗ tcpx_test failed with rc=" << rc_test << std::endl;
          break;
        }
        if (!done) {
          usleep(1000);  // 1ms delay
        }
      }

      if (done) {
        std::cout << "✓ SUCCESS: Sent " << sent_size << " bytes" << std::endl;
      } else {
        std::cout << "✗ TIMEOUT: Send not completed after " << max_polls
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
    std::cout << "✗ ERROR: Invalid mode. Use 'server' or 'client'" << std::endl;
    return 1;
  }

  std::cout << "\n=== TCPX Connection Test COMPLETED ===" << std::endl;
  return 0;
}
