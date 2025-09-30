/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef TCPX_RX_DESCRIPTOR_H_
#define TCPX_RX_DESCRIPTOR_H_

#include <stdint.h>
#include "include/tcpx_structs.h"

namespace tcpx {
namespace rx {

// Use TCPX plugin's loadMeta as the descriptor type (avoids duplication)
using UnpackDescriptor = tcpx::plugin::loadMeta;

// Maximum descriptors per unpack operation
#define MAX_UNPACK_DESCRIPTORS 2048

// Descriptor block for GPU kernel
struct UnpackDescriptorBlock {
  UnpackDescriptor descriptors[MAX_UNPACK_DESCRIPTORS];
  uint32_t count;           // Number of valid descriptors
  uint32_t total_bytes;     // Total bytes to unpack
  void* bounce_buffer;      // Source bounce buffer base
  void* dst_buffer;         // Destination buffer base
  void* ready_flag;         // Device pointer to a 64-bit counter/flag (optional)
  uint64_t ready_threshold; // Optional: expected minimal value to consider ready

  UnpackDescriptorBlock()
    : count(0), total_bytes(0), bounce_buffer(nullptr), dst_buffer(nullptr)
    , ready_flag(nullptr), ready_threshold(0) {}
};

// Simple utility function to build descriptor block from loadMeta array
inline void buildDescriptorBlock(
    const tcpx::plugin::loadMeta* meta_entries,
    uint32_t count,
    void* bounce_buffer,
    void* dst_buffer,
    UnpackDescriptorBlock& desc_block) {
  desc_block.count = count;
  desc_block.total_bytes = 0;
  desc_block.bounce_buffer = bounce_buffer;
  desc_block.dst_buffer = dst_buffer;

  for (uint32_t i = 0; i < count && i < MAX_UNPACK_DESCRIPTORS; ++i) {
    desc_block.descriptors[i] = meta_entries[i];
    desc_block.total_bytes += meta_entries[i].len;
  }
}

} // namespace rx
} // namespace tcpx

#endif // TCPX_RX_DESCRIPTOR_H_

