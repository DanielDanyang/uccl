#include "../tcpx_interface.h"
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// Simple test to verify we can extract connection info from TCPX listen_comm
int main() {
  std::cout << "=== TCPX Handle Extraction Test ===" << std::endl;
  
  // Enable debug output
  setenv("UCCL_TCPX_DEBUG", "1", 1);
  
  // Initialize TCPX
  std::cout << "\n[Step 1] Initializing TCPX..." << std::endl;
  int device_count = tcpx_get_device_count();
  if (device_count <= 0) {
    std::cout << "✗ FAILED: No TCPX devices found" << std::endl;
    return 1;
  }
  std::cout << "✓ SUCCESS: Found " << device_count << " TCPX devices" << std::endl;
  
  // Create TCPX listening socket
  std::cout << "\n[Step 2] Creating TCPX listen socket..." << std::endl;
  int dev_id = 0;
  char handle_data[128];
  memset(handle_data, 0, sizeof(handle_data));
  
  void* listen_comm = nullptr;
  int rc = tcpx_listen(dev_id, handle_data, &listen_comm);
  if (rc != 0) {
    std::cout << "✗ FAILED: tcpx_listen returned " << rc << std::endl;
    return 1;
  }
  std::cout << "✓ SUCCESS: TCPX listen created" << std::endl;
  std::cout << "Listen comm: " << listen_comm << std::endl;
  
  // Try to extract connection information
  std::cout << "\n[Step 3] Analyzing handle data..." << std::endl;
  std::cout << "Handle data (first 64 bytes as hex):" << std::endl;
  for (int i = 0; i < 64; i++) {
    printf("%02x ", (unsigned char)handle_data[i]);
    if ((i + 1) % 16 == 0) printf("\n");
  }
  printf("\n");
  
  // Check if handle contains any meaningful data
  bool has_data = false;
  for (int i = 0; i < 128; i++) {
    if (handle_data[i] != 0) {
      has_data = true;
      break;
    }
  }
  
  if (has_data) {
    std::cout << "✓ Handle contains data - TCPX may have filled it" << std::endl;
  } else {
    std::cout << "✗ Handle is empty - TCPX didn't fill connection info" << std::endl;
  }
  
  // Try to get local network interface info
  std::cout << "\n[Step 4] Getting local network info..." << std::endl;
  
  // Create a dummy socket to get local IP
  int dummy_sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (dummy_sock >= 0) {
    struct sockaddr_in dummy_addr;
    memset(&dummy_addr, 0, sizeof(dummy_addr));
    dummy_addr.sin_family = AF_INET;
    dummy_addr.sin_addr.s_addr = inet_addr("8.8.8.8");  // Google DNS
    dummy_addr.sin_port = htons(53);
    
    if (connect(dummy_sock, (struct sockaddr*)&dummy_addr, sizeof(dummy_addr)) == 0) {
      struct sockaddr_in local_addr;
      socklen_t addr_len = sizeof(local_addr);
      if (getsockname(dummy_sock, (struct sockaddr*)&local_addr, &addr_len) == 0) {
        char* local_ip = inet_ntoa(local_addr.sin_addr);
        std::cout << "✓ Local IP detected: " << local_ip << std::endl;
      }
    }
    close(dummy_sock);
  }
  
  // Check if we can get port info from TCPX logs
  std::cout << "\n[Step 5] Checking TCPX logs for port info..." << std::endl;
  std::cout << "Look for 'listen port' in the TCPX debug output above" << std::endl;
  
  std::cout << "\n=== Analysis Complete ===" << std::endl;
  std::cout << "Next steps:" << std::endl;
  std::cout << "1. If handle is empty, we need to extract info from TCPX internals" << std::endl;
  std::cout << "2. If handle has data, we need to decode its format" << std::endl;
  std::cout << "3. Look for port number in TCPX debug logs" << std::endl;
  
  return 0;
}
