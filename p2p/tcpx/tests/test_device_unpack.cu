/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Test suite for Device Unpack Kernels
 ************************************************************************/
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <chrono>
#include "../device/unpack_launch.h"
#include "../rx/rx_descriptor.h"

using namespace tcpx::device;
using namespace tcpx::rx;

class DeviceUnpackTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
    
    // Allocate host memory
    h_bounce_buffer_ = new char[BUFFER_SIZE];
    h_dst_buffer_ = new char[BUFFER_SIZE];
    h_expected_buffer_ = new char[BUFFER_SIZE];
    
    // Allocate device memory
    err = cudaMalloc(&d_bounce_buffer_, BUFFER_SIZE);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device bounce buffer";
    
    err = cudaMalloc(&d_dst_buffer_, BUFFER_SIZE);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device dst buffer";
    
    // Create CUDA stream
    err = cudaStreamCreate(&stream_);
    ASSERT_EQ(err, cudaSuccess) << "Failed to create CUDA stream";
    
    // Setup launcher configuration
    launch_config_.stream = stream_;
    launch_config_.enable_profiling = true;
    launcher_ = std::make_unique<UnpackLauncher>(launch_config_);
    
    // Initialize test data
    initializeTestData();
  }
  
  void TearDown() override {
    if (launcher_) {
      launcher_.reset();
    }
    
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
    
    if (d_bounce_buffer_) {
      cudaFree(d_bounce_buffer_);
    }
    
    if (d_dst_buffer_) {
      cudaFree(d_dst_buffer_);
    }
    
    delete[] h_bounce_buffer_;
    delete[] h_dst_buffer_;
    delete[] h_expected_buffer_;
  }
  
  void initializeTestData() {
    // Fill bounce buffer with test pattern
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      h_bounce_buffer_[i] = static_cast<char>(i % 256);
    }
    
    // Clear destination buffers
    memset(h_dst_buffer_, 0, BUFFER_SIZE);
    memset(h_expected_buffer_, 0, BUFFER_SIZE);
    
    // Copy to device
    cudaMemcpy(d_bounce_buffer_, h_bounce_buffer_, BUFFER_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(d_dst_buffer_, 0, BUFFER_SIZE);
  }
  
  // Helper function to create descriptor block
  UnpackDescriptorBlock createDescriptorBlock(const std::vector<TestDescriptor>& descriptors) {
    UnpackDescriptorBlock desc_block;
    desc_block.count = descriptors.size();
    desc_block.total_bytes = 0;
    desc_block.bounce_buffer = d_bounce_buffer_;
    desc_block.dst_buffer = d_dst_buffer_;
    
    for (size_t i = 0; i < descriptors.size(); ++i) {
      const auto& test_desc = descriptors[i];
      desc_block.descriptors[i] = UnpackDescriptor(
        test_desc.src_offset, test_desc.length, test_desc.dst_offset);
      desc_block.total_bytes += test_desc.length;
      
      // Update expected buffer
      memcpy(h_expected_buffer_ + test_desc.dst_offset,
             h_bounce_buffer_ + test_desc.src_offset,
             test_desc.length);
    }
    
    return desc_block;
  }
  
  // Verify results
  bool verifyResults(const UnpackDescriptorBlock& desc_block) {
    // Copy result from device
    cudaMemcpy(h_dst_buffer_, d_dst_buffer_, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Compare with expected
    for (uint32_t i = 0; i < desc_block.count; ++i) {
      const auto& desc = desc_block.descriptors[i];
      
      if (memcmp(h_dst_buffer_ + desc.dst_off,
                 h_expected_buffer_ + desc.dst_off,
                 desc.len) != 0) {
        return false;
      }
    }
    
    return true;
  }
  
  struct TestDescriptor {
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t length;
  };
  
  static constexpr size_t BUFFER_SIZE = 64 * 1024;  // 64KB
  
  char* h_bounce_buffer_;
  char* h_dst_buffer_;
  char* h_expected_buffer_;
  
  void* d_bounce_buffer_;
  void* d_dst_buffer_;
  
  cudaStream_t stream_;
  UnpackLaunchConfig launch_config_;
  std::unique_ptr<UnpackLauncher> launcher_;
};

TEST_F(DeviceUnpackTest, EmptyDescriptorBlock) {
  UnpackDescriptorBlock desc_block;
  desc_block.count = 0;
  desc_block.total_bytes = 0;
  desc_block.bounce_buffer = d_bounce_buffer_;
  desc_block.dst_buffer = d_dst_buffer_;
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
}

TEST_F(DeviceUnpackTest, SingleDescriptor) {
  std::vector<TestDescriptor> descriptors = {
    {0, 0, 256}  // Copy 256 bytes from offset 0 to offset 0
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, MultipleDescriptors) {
  std::vector<TestDescriptor> descriptors = {
    {0, 1000, 256},      // Copy from 0 to 1000
    {512, 2000, 512},    // Copy from 512 to 2000
    {1024, 3000, 1024}   // Copy from 1024 to 3000
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, AlignedTransfers) {
  // Test various alignment scenarios
  std::vector<TestDescriptor> descriptors = {
    {0, 0, 16},          // 16-byte aligned
    {16, 16, 32},        // 16-byte aligned
    {64, 64, 64},        // 16-byte aligned
    {128, 128, 128}      // 16-byte aligned
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, UnalignedTransfers) {
  // Test unaligned transfers
  std::vector<TestDescriptor> descriptors = {
    {1, 1001, 255},      // Unaligned start and size
    {257, 2001, 511},    // Unaligned start and size
    {769, 3001, 1023}    // Unaligned start and size
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, SmallTransfers) {
  // Test small transfers (< 16 bytes)
  std::vector<TestDescriptor> descriptors = {
    {0, 1000, 1},        // 1 byte
    {1, 1001, 2},        // 2 bytes
    {3, 1003, 4},        // 4 bytes
    {7, 1007, 8},        // 8 bytes
    {15, 1015, 15}       // 15 bytes
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, LargeTransfers) {
  // Test large transfers
  std::vector<TestDescriptor> descriptors = {
    {0, 10000, 8192},     // 8KB
    {8192, 20000, 16384}, // 16KB
    {24576, 40000, 32768} // 32KB
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, ManySmallDescriptors) {
  // Test many small descriptors (should use small kernel)
  std::vector<TestDescriptor> descriptors;
  
  for (int i = 0; i < 100; ++i) {
    descriptors.push_back({
      static_cast<uint32_t>(i * 64),      // src_offset
      static_cast<uint32_t>(20000 + i * 64), // dst_offset
      64                                   // length
    });
  }
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, AsyncExecution) {
  std::vector<TestDescriptor> descriptors = {
    {0, 1000, 1024}
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  // Launch asynchronously
  int ret = launcher_->launch(desc_block);
  EXPECT_EQ(ret, 0);
  
  // Check if still running
  bool complete = launcher_->isComplete();
  // May or may not be complete depending on timing
  
  // Wait for completion
  cudaError_t err = launcher_->waitForCompletion();
  EXPECT_EQ(err, cudaSuccess);
  
  // Should be complete now
  complete = launcher_->isComplete();
  EXPECT_TRUE(complete);
  
  bool correct = verifyResults(desc_block);
  EXPECT_TRUE(correct);
}

TEST_F(DeviceUnpackTest, StatisticsTracking) {
  std::vector<TestDescriptor> descriptors = {
    {0, 1000, 512},
    {512, 2000, 1024}
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  // Reset stats
  launcher_->resetStats();
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_EQ(ret, 0);
  
  const auto& stats = launcher_->getStats();
  EXPECT_EQ(stats.launches, 1);
  EXPECT_EQ(stats.descriptors_processed, 2);
  EXPECT_EQ(stats.bytes_unpacked, 1536);
  EXPECT_EQ(stats.kernel_errors, 0);
  EXPECT_GT(stats.total_time_ms, 0.0f);
  EXPECT_GT(stats.avg_bandwidth_gbps, 0.0f);
}

TEST_F(DeviceUnpackTest, ErrorHandling) {
  // Test with too many descriptors
  launch_config_.max_descriptors = 2;
  launcher_->updateConfig(launch_config_);
  
  std::vector<TestDescriptor> descriptors = {
    {0, 1000, 256},
    {256, 2000, 256},
    {512, 3000, 256}  // Exceeds max_descriptors
  };
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  int ret = launcher_->launchSync(desc_block);
  EXPECT_LT(ret, 0);  // Should fail
  
  const auto& stats = launcher_->getStats();
  EXPECT_GT(stats.kernel_errors, 0);
}

// Performance test
TEST_F(DeviceUnpackTest, PerformanceTest) {
  const int NUM_ITERATIONS = 100;
  const int NUM_DESCRIPTORS = 50;
  
  std::vector<TestDescriptor> descriptors;
  for (int i = 0; i < NUM_DESCRIPTORS; ++i) {
    descriptors.push_back({
      static_cast<uint32_t>(i * 256),
      static_cast<uint32_t>(20000 + i * 256),
      256
    });
  }
  
  UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
  
  launcher_->resetStats();
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    int ret = launcher_->launchSync(desc_block);
    EXPECT_EQ(ret, 0);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_us = static_cast<double>(duration.count()) / NUM_ITERATIONS;
  std::cout << "Average unpack time: " << avg_time_us << " microseconds" << std::endl;
  
  const auto& stats = launcher_->getStats();
  std::cout << "Average bandwidth: " << stats.avg_bandwidth_gbps << " GB/s" << std::endl;
  
  // Should be reasonably fast
  EXPECT_LT(avg_time_us, 1000.0);  // < 1ms per unpack
  EXPECT_GT(stats.avg_bandwidth_gbps, 1.0f);  // > 1 GB/s
}

// Stress test with random data
TEST_F(DeviceUnpackTest, StressTest) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> offset_dist(0, BUFFER_SIZE / 2);
  std::uniform_int_distribution<uint32_t> size_dist(1, 1024);
  
  const int NUM_DESCRIPTORS = 200;
  std::vector<TestDescriptor> descriptors;
  
  uint32_t dst_offset = 0;
  for (int i = 0; i < NUM_DESCRIPTORS && dst_offset < BUFFER_SIZE / 2; ++i) {
    uint32_t src_offset = offset_dist(gen);
    uint32_t length = std::min(size_dist(gen), 
                              static_cast<uint32_t>(BUFFER_SIZE / 2 - dst_offset));
    
    if (src_offset + length <= BUFFER_SIZE / 2) {
      descriptors.push_back({src_offset, dst_offset, length});
      dst_offset += length;
    }
  }
  
  if (!descriptors.empty()) {
    UnpackDescriptorBlock desc_block = createDescriptorBlock(descriptors);
    
    int ret = launcher_->launchSync(desc_block);
    EXPECT_EQ(ret, 0);
    
    bool correct = verifyResults(desc_block);
    EXPECT_TRUE(correct);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  
  // Check CUDA availability
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA devices available, skipping tests" << std::endl;
    return 0;
  }
  
  return RUN_ALL_TESTS();
}
