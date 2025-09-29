/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TCPX_RX_CMSG_PARSER_H_
#define TCPX_RX_CMSG_PARSER_H_

#include <stdint.h>
#include <sys/socket.h>
#include <vector>
#include <memory>

namespace tcpx {
namespace rx {

// DevMem-TCP control message types (from Linux kernel)
#define SCM_DEVMEM_DMABUF 0x42
#define SCM_DEVMEM_LINEAR 0x43

// DevMem fragment descriptor from kernel cmsg
struct DevMemFragment {
  uint32_t frag_offset;   // Offset within dmabuf
  uint32_t frag_size;     // Fragment size in bytes
  uint32_t frag_token;    // Token for later freeing
  uint32_t dmabuf_id;     // DMA buffer ID
};

// Normalized scatter-gather entry for host processing
struct ScatterEntry {
  void* src_ptr;          // Source pointer (bounce buffer or dmabuf)
  uint32_t src_offset;    // Offset within source
  uint32_t dst_offset;    // Offset within destination buffer
  uint32_t length;        // Data length
  bool is_devmem;         // True if from dmabuf, false if from linear buffer
  uint32_t token;         // Token for devmem fragments (for freeing)
};

// Parsed scatter-gather list
struct ScatterList {
  std::vector<ScatterEntry> entries;
  size_t total_bytes;
  size_t devmem_bytes;
  size_t linear_bytes;
  
  ScatterList() : total_bytes(0), devmem_bytes(0), linear_bytes(0) {}
  
  void clear() {
    entries.clear();
    total_bytes = devmem_bytes = linear_bytes = 0;
  }
};

// CMSG parser configuration
struct ParserConfig {
  void* bounce_buffer;      // Linear bounce buffer base
  size_t bounce_size;       // Bounce buffer size
  void* dmabuf_base;        // DMA buffer base address
  size_t dmabuf_size;       // DMA buffer size
  uint32_t expected_dmabuf_id; // Expected dmabuf ID
  
  ParserConfig() 
    : bounce_buffer(nullptr), bounce_size(0)
    , dmabuf_base(nullptr), dmabuf_size(0)
    , expected_dmabuf_id(0) {}
};

// CMSG parser statistics
struct ParserStats {
  uint64_t total_messages;
  uint64_t devmem_fragments;
  uint64_t linear_fragments;
  uint64_t parse_errors;
  uint64_t validation_errors;
  
  ParserStats() { reset(); }
  
  void reset() {
    total_messages = devmem_fragments = linear_fragments = 0;
    parse_errors = validation_errors = 0;
  }
};

// Main CMSG parser class
class CmsgParser {
public:
  explicit CmsgParser(const ParserConfig& config);
  ~CmsgParser() = default;

  // Parse control messages from recvmsg into normalized scatter list
  // Returns 0 on success, negative error code on failure
  int parse(const struct msghdr* msg, ScatterList& scatter_list);
  
  // Validate parsed scatter list for consistency
  bool validate(const ScatterList& scatter_list, size_t expected_total_bytes) const;
  
  // Get parser statistics
  const ParserStats& getStats() const { return stats_; }
  void resetStats() { stats_.reset(); }
  
  // Update configuration
  void updateConfig(const ParserConfig& config) { config_ = config; }

private:
  // Parse individual control message
  int parseCmsg(const struct cmsghdr* cmsg, ScatterList& scatter_list, 
                uint32_t& current_dst_offset);
  
  // Validate fragment bounds
  bool validateFragment(const DevMemFragment& frag) const;
  bool validateLinearFragment(uint32_t size, uint32_t offset) const;
  
  ParserConfig config_;
  ParserStats stats_;
};

// Utility functions
namespace utils {

// Extract DevMem fragment from cmsg data
DevMemFragment extractDevMemFragment(const struct cmsghdr* cmsg);

// Calculate total data size from scatter list
size_t calculateTotalSize(const ScatterList& scatter_list);

// Validate scatter list for overlaps and gaps
bool validateScatterList(const ScatterList& scatter_list, size_t expected_size);

// Debug: dump scatter list to string
std::string dumpScatterList(const ScatterList& scatter_list);

} // namespace utils
} // namespace rx
} // namespace tcpx

#endif // TCPX_RX_CMSG_PARSER_H_
