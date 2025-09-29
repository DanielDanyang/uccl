/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "unpack_launch.h"
#include <iostream>
#include <algorithm>

// Forward declare CUDA kernels
extern "C" {
__global__ void tcpxUnpackKernel(const tcpx::rx::UnpackDescriptorBlock* desc_block);
__global__ void tcpxUnpackKernelSmall(const tcpx::rx::UnpackDescriptorBlock* desc_block);
}

// Forward declare utility function from unpack_kernels.cu
extern "C" {
tcpx::device::KernelLaunchParams calculateLaunchParams(const tcpx::rx::UnpackDescriptorBlock& desc_block);
}

namespace tcpx {
namespace device {

UnpackLauncher::UnpackLauncher(const UnpackLaunchConfig& config)
  : config_(config), d_desc_block_(nullptr), d_desc_block_size_(0)
  , start_event_(nullptr), stop_event_(nullptr), events_created_(false) {
  
  if (config_.enable_profiling) {
    cudaError_t err1 = cudaEventCreate(&start_event_);
    cudaError_t err2 = cudaEventCreate(&stop_event_);
    events_created_ = (err1 == cudaSuccess && err2 == cudaSuccess);
    
    if (!events_created_) {
      std::cerr << "Warning: Failed to create CUDA events for profiling\n";
    }
  }
}

UnpackLauncher::~UnpackLauncher() {
  if (d_desc_block_) {
    cudaFree(d_desc_block_);
  }
  
  if (events_created_) {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }
}

int UnpackLauncher::launch(const tcpx::rx::UnpackDescriptorBlock& desc_block) {
  return launch(desc_block, config_.stream);
}

int UnpackLauncher::launch(const tcpx::rx::UnpackDescriptorBlock& desc_block,
                           cudaStream_t stream) {
  if (desc_block.count == 0) {
    return 0; // Nothing to do
  }
  
  if (desc_block.count > config_.max_descriptors) {
    stats_.kernel_errors++;
    return -1;
  }
  
  // Copy descriptor block to device
  int ret = copyDescriptorBlockToDevice(desc_block);
  if (ret < 0) {
    stats_.kernel_errors++;
    return ret;
  }
  
  // Calculate launch parameters
  KernelLaunchParams params = calculateLaunchParams(desc_block);
  if (!launch_utils::validateLaunchParams(params)) {
    stats_.kernel_errors++;
    return -3;
  }
  
  // Start profiling if enabled
  float execution_time = 0.0f;
  if (config_.enable_profiling && events_created_) {
    cudaEventRecord(start_event_, stream);
  }
  
  // Launch kernel
  ret = launchKernel(params);
  if (ret < 0) {
    stats_.kernel_errors++;
    return ret;
  }
  
  // Stop profiling if enabled
  if (config_.enable_profiling && events_created_) {
    cudaEventRecord(stop_event_, stream);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&execution_time, start_event_, stop_event_);
  }
  
  // Update statistics
  updateStats(desc_block, execution_time);
  
  return 0;
}

int UnpackLauncher::launchSync(const tcpx::rx::UnpackDescriptorBlock& desc_block) {
  int ret = launch(desc_block);
  if (ret < 0) {
    return ret;
  }
  
  cudaError_t err = waitForCompletion();
  return (err == cudaSuccess) ? 0 : -4;
}

bool UnpackLauncher::isComplete() const {
  if (!config_.stream) {
    return true; // Default stream is always synchronous
  }
  
  cudaError_t err = cudaStreamQuery(config_.stream);
  return (err == cudaSuccess);
}

cudaError_t UnpackLauncher::waitForCompletion() {
  if (config_.stream) {
    return cudaStreamSynchronize(config_.stream);
  } else {
    return cudaDeviceSynchronize();
  }
}

int UnpackLauncher::allocateDeviceMemory(size_t size) {
  if (d_desc_block_size_ >= size) {
    return 0; // Already have enough memory
  }
  
  // Free existing memory
  if (d_desc_block_) {
    cudaFree(d_desc_block_);
    d_desc_block_ = nullptr;
    d_desc_block_size_ = 0;
  }
  
  // Allocate new memory with some padding
  size_t padded_size = size + (size / 4); // 25% padding
  cudaError_t err = cudaMalloc(&d_desc_block_, padded_size);
  if (err != cudaSuccess) {
    return -1;
  }
  
  d_desc_block_size_ = padded_size;
  return 0;
}

int UnpackLauncher::copyDescriptorBlockToDevice(
    const tcpx::rx::UnpackDescriptorBlock& desc_block) {
  
  size_t required_size = sizeof(tcpx::rx::UnpackDescriptorBlock);
  
  int ret = allocateDeviceMemory(required_size);
  if (ret < 0) {
    return ret;
  }
  
  cudaError_t err = cudaMemcpy(d_desc_block_, &desc_block, required_size,
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return -2;
  }
  
  return 0;
}

KernelLaunchParams UnpackLauncher::calculateLaunchParams(
    const tcpx::rx::UnpackDescriptorBlock& desc_block) const {

  return calculateLaunchParams(desc_block);
}

int UnpackLauncher::launchKernel(const KernelLaunchParams& params) {
  const tcpx::rx::UnpackDescriptorBlock* d_desc_ptr = 
    static_cast<const tcpx::rx::UnpackDescriptorBlock*>(d_desc_block_);
  
  cudaError_t err;
  
  if (config_.use_small_kernel) {
    tcpxUnpackKernelSmall<<<params.grid_size, params.block_size, 
                           params.shared_mem_size, config_.stream>>>(d_desc_ptr);
  } else {
    tcpxUnpackKernel<<<params.grid_size, params.block_size,
                      params.shared_mem_size, config_.stream>>>(d_desc_ptr);
  }
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
  
  return 0;
}

void UnpackLauncher::updateStats(const tcpx::rx::UnpackDescriptorBlock& desc_block,
                                 float execution_time_ms) {
  stats_.launches++;
  stats_.descriptors_processed += desc_block.count;
  stats_.bytes_unpacked += desc_block.total_bytes;
  
  if (execution_time_ms > 0.0f) {
    stats_.total_time_ms += execution_time_ms;
    
    // Calculate bandwidth in GB/s
    float bandwidth_gbps = (desc_block.total_bytes / (1024.0f * 1024.0f * 1024.0f)) /
                          (execution_time_ms / 1000.0f);
    
    // Update average bandwidth
    stats_.avg_bandwidth_gbps = 
      (stats_.avg_bandwidth_gbps * (stats_.launches - 1) + bandwidth_gbps) / 
      stats_.launches;
  }
}

void UnpackLauncher::updateConfig(const UnpackLaunchConfig& config) {
  config_ = config;
  
  // Recreate events if profiling setting changed
  if (config_.enable_profiling && !events_created_) {
    cudaError_t err1 = cudaEventCreate(&start_event_);
    cudaError_t err2 = cudaEventCreate(&stop_event_);
    events_created_ = (err1 == cudaSuccess && err2 == cudaSuccess);
  } else if (!config_.enable_profiling && events_created_) {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    events_created_ = false;
  }
}

namespace launch_utils {

int calculateOptimalBlockSize(uint32_t descriptor_count, uint32_t avg_descriptor_size) {
  // For small descriptors, use more threads per block
  if (avg_descriptor_size < 256) {
    return 256;
  } else if (avg_descriptor_size < 1024) {
    return 512;
  } else {
    return 1024;
  }
}

float estimateExecutionTime(const tcpx::rx::UnpackDescriptorBlock& desc_block) {
  // Simple estimation based on data size and theoretical bandwidth
  float theoretical_bandwidth = calculateTheoreticalBandwidth();
  float data_gb = desc_block.total_bytes / (1024.0f * 1024.0f * 1024.0f);
  
  // Add some overhead for kernel launch and small transfers
  float base_overhead_ms = 0.01f; // 10 microseconds
  float transfer_time_ms = (data_gb / theoretical_bandwidth) * 1000.0f;
  
  return base_overhead_ms + transfer_time_ms;
}

bool validateLaunchParams(const KernelLaunchParams& params) {
  if (params.grid_size.x == 0 || params.block_size.x == 0) {
    return false;
  }
  
  if (params.block_size.x > 1024) {
    return false; // Exceeds maximum block size
  }
  
  return true;
}

cudaDeviceProp getDeviceProperties() {
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  return prop;
}

float calculateTheoreticalBandwidth() {
  cudaDeviceProp prop = getDeviceProperties();
  
  // Calculate memory bandwidth in GB/s
  float memory_clock_khz = prop.memoryClockRate;
  int memory_bus_width = prop.memoryBusWidth;
  
  // Bandwidth = (memory_clock * 2) * (bus_width / 8) / 1e6
  float bandwidth_gbps = (memory_clock_khz * 2.0f * memory_bus_width / 8.0f) / 1e6f;
  
  // Apply efficiency factor (typically 80-90% for memory copy)
  return bandwidth_gbps * 0.85f;
}

} // namespace launch_utils
} // namespace device
} // namespace tcpx
