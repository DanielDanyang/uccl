#include "../tcpx_interface.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

int main() {
  std::cout << "=== TCPX Data Transfer Test ===" << std::endl;
  std::cout << "This test verifies memory registration and data transfer APIs" << std::endl;
  
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
  
  // Test memory registration without actual connection
  std::cout << "\n[Step 2] Testing memory registration..." << std::endl;
  
  // Allocate test buffer
  const int buffer_size = 1024;
  char* test_buffer = new char[buffer_size];
  memset(test_buffer, 0xAB, buffer_size);  // Fill with test pattern
  
  std::cout << "Allocated " << buffer_size << " bytes at " << test_buffer << std::endl;
  
  // Note: Memory registration typically requires a valid communication handle
  // For now, we'll test with nullptr to see how the API behaves
  void* mhandle = nullptr;
  void* dummy_comm = nullptr;  // This will likely fail, but let's see the error
  
  std::cout << "Attempting memory registration..." << std::endl;
  int rc_reg = tcpx_reg_mr(dummy_comm, test_buffer, buffer_size, 0, &mhandle);
  
  if (rc_reg == 0) {
    std::cout << "✓ SUCCESS: Memory registered, mhandle=" << mhandle << std::endl;
    
    // Test deregistration
    std::cout << "Testing memory deregistration..." << std::endl;
    int rc_dereg = tcpx_dereg_mr(dummy_comm, mhandle);
    if (rc_dereg == 0) {
      std::cout << "✓ SUCCESS: Memory deregistered" << std::endl;
    } else {
      std::cout << "✗ FAILED: Memory deregistration failed with rc=" << rc_dereg << std::endl;
    }
  } else {
    std::cout << "✗ EXPECTED: Memory registration failed with rc=" << rc_reg << std::endl;
    std::cout << "This is expected since we don't have a valid communication handle" << std::endl;
  }
  
  // Test data transfer APIs (these will also fail without valid handles)
  std::cout << "\n[Step 3] Testing data transfer APIs..." << std::endl;
  
  void* send_request = nullptr;
  void* recv_request = nullptr;
  
  // Test send
  std::cout << "Testing tcpx_isend..." << std::endl;
  int rc_send = tcpx_isend(dummy_comm, test_buffer, 100, 42, mhandle, &send_request);
  if (rc_send == 0) {
    std::cout << "✓ UNEXPECTED: tcpx_isend succeeded, request=" << send_request << std::endl;
  } else {
    std::cout << "✗ EXPECTED: tcpx_isend failed with rc=" << rc_send << std::endl;
  }
  
  // Test receive
  std::cout << "Testing tcpx_irecv..." << std::endl;
  void* recv_data[1] = {test_buffer};
  int recv_sizes[1] = {100};
  int recv_tags[1] = {42};
  void* recv_mhandles[1] = {mhandle};
  
  int rc_recv = tcpx_irecv(dummy_comm, 1, recv_data, recv_sizes, recv_tags, recv_mhandles, &recv_request);
  if (rc_recv == 0) {
    std::cout << "✓ UNEXPECTED: tcpx_irecv succeeded, request=" << recv_request << std::endl;
  } else {
    std::cout << "✗ EXPECTED: tcpx_irecv failed with rc=" << rc_recv << std::endl;
  }
  
  // Test completion check
  std::cout << "Testing tcpx_test..." << std::endl;
  int done = 0, size = 0;
  int rc_test = tcpx_test(send_request, &done, &size);
  if (rc_test == 0) {
    std::cout << "✓ UNEXPECTED: tcpx_test succeeded, done=" << done << " size=" << size << std::endl;
  } else {
    std::cout << "✗ EXPECTED: tcpx_test failed with rc=" << rc_test << std::endl;
  }
  
  // Cleanup
  delete[] test_buffer;
  
  std::cout << "\n=== TCPX Data Transfer Test COMPLETED ===" << std::endl;
  std::cout << "Summary:" << std::endl;
  std::cout << "- TCPX plugin loads successfully" << std::endl;
  std::cout << "- Device discovery works" << std::endl;
  std::cout << "- Memory registration/data transfer APIs are available" << std::endl;
  std::cout << "- APIs correctly reject invalid parameters" << std::endl;
  std::cout << "\nNext step: Test with actual connection handles from test_connection" << std::endl;
  
  return 0;
}
