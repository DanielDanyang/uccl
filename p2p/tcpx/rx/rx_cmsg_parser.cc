/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "rx_cmsg_parser.h"
#include <cstring>
#include <sstream>
#include <iostream>

namespace tcpx {
namespace rx {

CmsgParser::CmsgParser(const ParserConfig& config) 
  : config_(config) {
}

int CmsgParser::parse(const struct msghdr* msg, ScatterList& scatter_list) {
  if (!msg) return -1;
  
  scatter_list.clear();
  stats_.total_messages++;
  
  uint32_t current_dst_offset = 0;
  
  // Iterate through all control messages
  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(msg); 
       cmsg != nullptr; 
       cmsg = CMSG_NXTHDR(msg, cmsg)) {
    
    if (cmsg->cmsg_level != SOL_SOCKET) continue;
    
    int ret = parseCmsg(cmsg, scatter_list, current_dst_offset);
    if (ret < 0) {
      stats_.parse_errors++;
      return ret;
    }
  }
  
  return 0;
}

int CmsgParser::parseCmsg(const struct cmsghdr* cmsg, ScatterList& scatter_list,
                          uint32_t& current_dst_offset) {
  ScatterEntry entry;
  
  switch (cmsg->cmsg_type) {
    case SCM_DEVMEM_DMABUF: {
      DevMemFragment frag = utils::extractDevMemFragment(cmsg);
      
      if (!validateFragment(frag)) {
        stats_.validation_errors++;
        return -2;
      }
      
      entry.src_ptr = static_cast<char*>(config_.dmabuf_base) + frag.frag_offset;
      entry.src_offset = frag.frag_offset;
      entry.dst_offset = current_dst_offset;
      entry.length = frag.frag_size;
      entry.is_devmem = true;
      entry.token = frag.frag_token;
      
      scatter_list.entries.push_back(entry);
      scatter_list.devmem_bytes += frag.frag_size;
      scatter_list.total_bytes += frag.frag_size;
      current_dst_offset += frag.frag_size;
      
      stats_.devmem_fragments++;
      break;
    }
    
    case SCM_DEVMEM_LINEAR: {
      // Linear fragment - data is in bounce buffer
      if (cmsg->cmsg_len < CMSG_LEN(sizeof(uint32_t))) {
        stats_.parse_errors++;
        return -3;
      }
      
      uint32_t frag_size = *reinterpret_cast<const uint32_t*>(CMSG_DATA(cmsg));
      
      if (!validateLinearFragment(frag_size, current_dst_offset)) {
        stats_.validation_errors++;
        return -4;
      }
      
      entry.src_ptr = config_.bounce_buffer;
      entry.src_offset = current_dst_offset; // Linear data maps 1:1
      entry.dst_offset = current_dst_offset;
      entry.length = frag_size;
      entry.is_devmem = false;
      entry.token = 0; // No token for linear fragments
      
      scatter_list.entries.push_back(entry);
      scatter_list.linear_bytes += frag_size;
      scatter_list.total_bytes += frag_size;
      current_dst_offset += frag_size;
      
      stats_.linear_fragments++;
      break;
    }
    
    default:
      // Unknown cmsg type, skip
      break;
  }
  
  return 0;
}

bool CmsgParser::validateFragment(const DevMemFragment& frag) const {
  // Check dmabuf ID
  if (frag.dmabuf_id != config_.expected_dmabuf_id) {
    return false;
  }
  
  // Check bounds
  if (frag.frag_offset + frag.frag_size > config_.dmabuf_size) {
    return false;
  }
  
  // Check alignment (optional, depends on requirements)
  if (frag.frag_offset % 4 != 0) {
    return false;
  }
  
  return true;
}

bool CmsgParser::validateLinearFragment(uint32_t size, uint32_t offset) const {
  if (!config_.bounce_buffer) return false;
  
  if (offset + size > config_.bounce_size) {
    return false;
  }
  
  return true;
}

bool CmsgParser::validate(const ScatterList& scatter_list, 
                          size_t expected_total_bytes) const {
  if (scatter_list.total_bytes != expected_total_bytes) {
    return false;
  }
  
  // Check for gaps or overlaps
  if (!utils::validateScatterList(scatter_list, expected_total_bytes)) {
    return false;
  }
  
  return true;
}

namespace utils {

DevMemFragment extractDevMemFragment(const struct cmsghdr* cmsg) {
  DevMemFragment frag = {};
  
  if (cmsg->cmsg_len >= CMSG_LEN(sizeof(DevMemFragment))) {
    const DevMemFragment* data = 
      reinterpret_cast<const DevMemFragment*>(CMSG_DATA(cmsg));
    frag = *data;
  }
  
  return frag;
}

size_t calculateTotalSize(const ScatterList& scatter_list) {
  return scatter_list.total_bytes;
}

bool validateScatterList(const ScatterList& scatter_list, size_t expected_size) {
  if (scatter_list.entries.empty()) {
    return expected_size == 0;
  }
  
  // Sort entries by destination offset for validation
  auto entries = scatter_list.entries;
  std::sort(entries.begin(), entries.end(), 
    [](const ScatterEntry& a, const ScatterEntry& b) {
      return a.dst_offset < b.dst_offset;
    });
  
  // Check for gaps and overlaps
  uint32_t expected_offset = 0;
  for (const auto& entry : entries) {
    if (entry.dst_offset != expected_offset) {
      return false; // Gap or overlap
    }
    expected_offset += entry.length;
  }
  
  return expected_offset == expected_size;
}

std::string dumpScatterList(const ScatterList& scatter_list) {
  std::ostringstream oss;
  oss << "ScatterList: " << scatter_list.entries.size() << " entries, "
      << scatter_list.total_bytes << " total bytes\n";
  oss << "  DevMem: " << scatter_list.devmem_bytes << " bytes\n";
  oss << "  Linear: " << scatter_list.linear_bytes << " bytes\n";
  
  for (size_t i = 0; i < scatter_list.entries.size(); ++i) {
    const auto& entry = scatter_list.entries[i];
    oss << "  [" << i << "] " 
        << (entry.is_devmem ? "DEVMEM" : "LINEAR")
        << " src_off=" << entry.src_offset
        << " dst_off=" << entry.dst_offset  
        << " len=" << entry.length;
    if (entry.is_devmem) {
      oss << " token=" << entry.token;
    }
    oss << "\n";
  }
  
  return oss.str();
}

} // namespace utils
} // namespace rx
} // namespace tcpx
