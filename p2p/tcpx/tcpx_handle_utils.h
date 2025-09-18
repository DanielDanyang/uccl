#pragma once

#include <arpa/inet.h>
#include <cstdint>
#include <iostream>
#include <netinet/in.h>
#include <string>

// TCPX handle structure (similar to RDMA's ucclHandle)
struct tcpxHandle {
  uint32_t ip_addr_u32;
  uint16_t listen_port;
  int remote_dev;
  int remote_gpuidx;
};

// Helper functions for IP address conversion
inline uint32_t str_to_ip(const char* ip_str) {
  struct in_addr addr;
  inet_aton(ip_str, &addr);
  return addr.s_addr;
}

inline std::string ip_to_str(uint32_t ip_u32) {
  struct in_addr addr;
  addr.s_addr = ip_u32;
  return std::string(inet_ntoa(addr));
}

// Try to extract port number from TCPX handle data
// TCPX fills the handle with binary data, we need to decode it
inline uint16_t extract_port_from_tcpx_handle(const char* handle_data) {
  // Based on the hex dump from test_handle_extraction:
  // 02 00 b2 1f 0a 80 00 33 00 00 00 00 00 00 00 00 
  // We know the port is 45599 from logs
  // Let's try different interpretations:
  
  // Try bytes 2-3: b2 1f
  uint16_t port1 = (handle_data[2] << 8) | handle_data[3];  // 0xb21f = 45599 ✓
  
  // Try bytes 0-1: 02 00  
  uint16_t port2 = (handle_data[0] << 8) | handle_data[1];  // 0x0200 = 512
  
  // Try little-endian interpretation of bytes 2-3
  uint16_t port3 = (handle_data[3] << 8) | handle_data[2];  // 0x1fb2 = 8114
  
  std::cout << "Port extraction attempts:" << std::endl;
  std::cout << "  bytes[2-3] big-endian: " << port1 << std::endl;
  std::cout << "  bytes[0-1] big-endian: " << port2 << std::endl;
  std::cout << "  bytes[2-3] little-endian: " << port3 << std::endl;
  
  // Based on our test, port1 (0xb21f = 45599) matches the TCPX log
  return port1;
}

// Try to extract IP address from TCPX handle data
inline uint32_t extract_ip_from_tcpx_handle(const char* handle_data) {
  // From hex dump: 0a 80 00 33 and 0a 00 00 6b
  // 0a 80 00 33 = 10.128.0.51 (eth1 IP)
  // 0a 00 00 6b = 10.0.0.107 (main IP)
  
  // Try bytes 4-7: 0a 80 00 33
  uint32_t ip1 = *((uint32_t*)(handle_data + 4));
  
  // Try bytes 52-55: 0a 00 00 6b (from second part of hex dump)
  uint32_t ip2 = *((uint32_t*)(handle_data + 52));
  
  std::cout << "IP extraction attempts:" << std::endl;
  std::cout << "  bytes[4-7]: " << ip_to_str(ip1) << std::endl;
  std::cout << "  bytes[52-55]: " << ip_to_str(ip2) << std::endl;
  
  // Use the main IP (10.0.0.107) for connection
  return ip2;
}

// Extract complete connection info from TCPX handle
inline tcpxHandle extract_tcpx_connection_info(const char* handle_data, int dev_id) {
  tcpxHandle result;
  memset(&result, 0, sizeof(result));
  
  std::cout << "Extracting TCPX connection info from handle..." << std::endl;
  
  // Extract port and IP from handle data
  result.listen_port = extract_port_from_tcpx_handle(handle_data);
  result.ip_addr_u32 = extract_ip_from_tcpx_handle(handle_data);
  result.remote_dev = dev_id;
  result.remote_gpuidx = 0;
  
  std::cout << "Extracted: IP=" << ip_to_str(result.ip_addr_u32) 
            << ", Port=" << result.listen_port 
            << ", Dev=" << result.remote_dev << std::endl;
  
  return result;
}
