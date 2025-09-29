/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Integration test for complete TCPX unpack pipeline
 ************************************************************************/
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <vector>
#include <memory>
#include <chrono>

#include "../rx/rx_cmsg_parser.h"
#include "../rx/rx_descriptor.h"
#include "../device/unpack_launch.h"

using namespace tcpx::rx;
using namespace tcpx::device;

class TcpxIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
    
    // Allocate host memory
    h_bounce_buffer_ = new char[BUFFER_SIZE];
    h_dmabuf_buffer_ = new char[BUFFER_SIZE];
    h_dst_buffer_ = new char[BUFFER_SIZE];
    h_expected_buffer_ = new char[BUFFER_SIZE];
    
    // Allocate device memory
    err = cudaMalloc(&d_bounce_buffer_, BUFFER_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    err = cudaMalloc(&d_dmabuf_buffer_, BUFFER_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    err = cudaMalloc(&d_dst_buffer_, BUFFER_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    // Create CUDA stream
    err = cudaStreamCreate(&stream_);
    ASSERT_EQ(err, cudaSuccess);
    
    // Setup components
    setupParser();
    setupDescriptorBuilder();
    setupUnpackLauncher();
    
    // Initialize test data
    initializeTestData();
  }
  
  void TearDown() override {
    launcher_.reset();
    builder_.reset();
    parser_.reset();
    
    if (stream_) cudaStreamDestroy(stream_);
    if (d_dst_buffer_) cudaFree(d_dst_buffer_);
    if (d_dmabuf_buffer_) cudaFree(d_dmabuf_buffer_);
    if (d_bounce_buffer_) cudaFree(d_bounce_buffer_);
    
    delete[] h_expected_buffer_;
    delete[] h_dst_buffer_;
    delete[] h_dmabuf_buffer_;
    delete[] h_bounce_buffer_;
  }
  
  void setupParser() {
    ParserConfig parser_config;
    parser_config.bounce_buffer = h_bounce_buffer_;
    parser_config.bounce_size = BUFFER_SIZE;
    parser_config.dmabuf_base = h_dmabuf_buffer_;
    parser_config.dmabuf_size = BUFFER_SIZE;
    parser_config.expected_dmabuf_id = TEST_DMABUF_ID;
    
    parser_ = std::make_unique<CmsgParser>(parser_config);
  }
  
  void setupDescriptorBuilder() {
    DescriptorConfig desc_config;
    desc_config.bounce_buffer = d_bounce_buffer_;
    desc_config.dst_buffer = d_dst_buffer_;
    desc_config.max_descriptors = 1024;
    
    builder_ = std::make_unique<DescriptorBuilder>(desc_config);
  }
  
  void setupUnpackLauncher() {
    UnpackLaunchConfig launch_config;
    launch_config.stream = stream_;
    launch_config.enable_profiling = true;
    launch_config.use_small_kernel = true;
    
    launcher_ = std::make_unique<UnpackLauncher>(launch_config);
  }
  
  void initializeTestData() {
    // Fill buffers with test patterns
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      h_bounce_buffer_[i] = static_cast<char>(i % 256);
      h_dmabuf_buffer_[i] = static_cast<char>((i + 128) % 256);
    }
    
    memset(h_dst_buffer_, 0, BUFFER_SIZE);
    memset(h_expected_buffer_, 0, BUFFER_SIZE);
    
    // Copy to device
    cudaMemcpy(d_bounce_buffer_, h_bounce_buffer_, BUFFER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dmabuf_buffer_, h_dmabuf_buffer_, BUFFER_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(d_dst_buffer_, 0, BUFFER_SIZE);
  }
  
  // Create mock control message
  struct msghdr createMockCmsg(const std::vector<MockFragment>& fragments) {
    struct msghdr msg = {};
    
    size_t total_size = 0;
    for (const auto& frag : fragments) {
      if (frag.is_devmem) {
        total_size += CMSG_SPACE(sizeof(DevMemFragment));
      } else {
        total_size += CMSG_SPACE(sizeof(uint32_t));
      }
    }
    
    msg.msg_control = control_buffer_;
    msg.msg_controllen = total_size;
    
    char* ptr = control_buffer_;
    for (const auto& frag : fragments) {
      struct cmsghdr* cmsg = reinterpret_cast<struct cmsghdr*>(ptr);
      cmsg->cmsg_level = SOL_SOCKET;
      
      if (frag.is_devmem) {
        cmsg->cmsg_type = SCM_DEVMEM_DMABUF;
        cmsg->cmsg_len = CMSG_LEN(sizeof(DevMemFragment));
        
        DevMemFragment* data = reinterpret_cast<DevMemFragment*>(CMSG_DATA(cmsg));
        data->frag_offset = frag.offset;
        data->frag_size = frag.size;
        data->frag_token = frag.token;
        data->dmabuf_id = TEST_DMABUF_ID;
        
        ptr += CMSG_SPACE(sizeof(DevMemFragment));
      } else {
        cmsg->cmsg_type = SCM_DEVMEM_LINEAR;
        cmsg->cmsg_len = CMSG_LEN(sizeof(uint32_t));
        
        uint32_t* data = reinterpret_cast<uint32_t*>(CMSG_DATA(cmsg));
        *data = frag.size;
        
        ptr += CMSG_SPACE(sizeof(uint32_t));
      }
    }
    
    return msg;
  }
  
  // Run complete pipeline
  int runPipeline(const std::vector<MockFragment>& fragments) {
    // Step 1: Parse control messages
    struct msghdr msg = createMockCmsg(fragments);
    ScatterList scatter_list;
    
    int ret = parser_->parse(&msg, scatter_list);
    if (ret < 0) return ret;
    
    // Step 2: Build descriptors
    UnpackDescriptorBlock desc_block;
    ret = builder_->buildDescriptors(scatter_list, desc_block);
    if (ret < 0) return ret;
    
    // Step 3: Copy devmem data to bounce buffer (simulate kernel behavior)
    for (const auto& entry : scatter_list.entries) {
      if (entry.is_devmem) {
        cudaMemcpy(
          static_cast<char*>(d_bounce_buffer_) + entry.src_offset,
          static_cast<char*>(d_dmabuf_buffer_) + entry.src_offset,
          entry.length,
          cudaMemcpyDeviceToDevice
        );
      }
    }
    
    // Step 4: Launch unpack kernel
    ret = launcher_->launchSync(desc_block);
    if (ret < 0) return ret;
    
    // Step 5: Verify results
    return verifyResults(scatter_list);
  }
  
  int verifyResults(const ScatterList& scatter_list) {
    // Copy result from device
    cudaMemcpy(h_dst_buffer_, d_dst_buffer_, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    
    // Build expected result
    memset(h_expected_buffer_, 0, BUFFER_SIZE);
    for (const auto& entry : scatter_list.entries) {
      if (entry.is_devmem) {
        memcpy(h_expected_buffer_ + entry.dst_offset,
               h_dmabuf_buffer_ + entry.src_offset,
               entry.length);
      } else {
        memcpy(h_expected_buffer_ + entry.dst_offset,
               h_bounce_buffer_ + entry.src_offset,
               entry.length);
      }
    }
    
    // Compare results
    for (const auto& entry : scatter_list.entries) {
      if (memcmp(h_dst_buffer_ + entry.dst_offset,
                 h_expected_buffer_ + entry.dst_offset,
                 entry.length) != 0) {
        return -1;
      }
    }
    
    return 0;
  }
  
  struct MockFragment {
    uint32_t offset;
    uint32_t size;
    uint32_t token;
    bool is_devmem;
  };
  
  static constexpr size_t BUFFER_SIZE = 32 * 1024;
  static constexpr uint32_t TEST_DMABUF_ID = 42;
  
  char* h_bounce_buffer_;
  char* h_dmabuf_buffer_;
  char* h_dst_buffer_;
  char* h_expected_buffer_;
  
  void* d_bounce_buffer_;
  void* d_dmabuf_buffer_;
  void* d_dst_buffer_;
  
  cudaStream_t stream_;
  char control_buffer_[2048];
  
  std::unique_ptr<CmsgParser> parser_;
  std::unique_ptr<DescriptorBuilder> builder_;
  std::unique_ptr<UnpackLauncher> launcher_;
};

TEST_F(TcpxIntegrationTest, SingleDevMemFragment) {
  std::vector<MockFragment> fragments = {
    {0, 1024, 12345, true}  // 1KB devmem fragment
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, MultipleDevMemFragments) {
  std::vector<MockFragment> fragments = {
    {0, 512, 11111, true},
    {1024, 1024, 22222, true},
    {2048, 2048, 33333, true}
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, MixedFragments) {
  std::vector<MockFragment> fragments = {
    {0, 256, 11111, true},    // DevMem
    {0, 512, 0, false},       // Linear (will be skipped by unpack)
    {512, 1024, 22222, true}  // DevMem
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, SmallFragments) {
  std::vector<MockFragment> fragments = {
    {0, 16, 11111, true},
    {16, 32, 22222, true},
    {48, 64, 33333, true},
    {112, 128, 44444, true}
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, LargeFragments) {
  std::vector<MockFragment> fragments = {
    {0, 8192, 11111, true},     // 8KB
    {8192, 16384, 22222, true}  // 16KB
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, UnalignedFragments) {
  std::vector<MockFragment> fragments = {
    {1, 255, 11111, true},      // Unaligned start and size
    {257, 511, 22222, true},    // Unaligned start and size
    {769, 1023, 33333, true}    // Unaligned start and size
  };
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, ManySmallFragments) {
  std::vector<MockFragment> fragments;
  
  for (int i = 0; i < 100; ++i) {
    fragments.push_back({
      static_cast<uint32_t>(i * 64),
      64,
      static_cast<uint32_t>(10000 + i),
      true
    });
  }
  
  int ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);
}

TEST_F(TcpxIntegrationTest, PerformanceTest) {
  const int NUM_ITERATIONS = 50;
  
  std::vector<MockFragment> fragments = {
    {0, 4096, 11111, true},
    {4096, 4096, 22222, true},
    {8192, 4096, 33333, true},
    {12288, 4096, 44444, true}
  };
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    int ret = runPipeline(fragments);
    EXPECT_EQ(ret, 0);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_us = static_cast<double>(duration.count()) / NUM_ITERATIONS;
  std::cout << "Average pipeline time: " << avg_time_us << " microseconds" << std::endl;
  
  // Print component statistics
  const auto& parser_stats = parser_->getStats();
  const auto& builder_stats = builder_->getStats();
  const auto& launcher_stats = launcher_->getStats();
  
  std::cout << "Parser stats: " << parser_stats.total_messages << " messages, "
            << parser_stats.devmem_fragments << " devmem fragments" << std::endl;
  std::cout << "Builder stats: " << builder_stats.blocks_built << " blocks, "
            << builder_stats.descriptors_created << " descriptors" << std::endl;
  std::cout << "Launcher stats: " << launcher_stats.launches << " launches, "
            << launcher_stats.avg_bandwidth_gbps << " GB/s avg bandwidth" << std::endl;
  
  // Should be reasonably fast (< 100 microseconds per pipeline)
  EXPECT_LT(avg_time_us, 100.0);
}

TEST_F(TcpxIntegrationTest, StressTest) {
  const int NUM_FRAGMENTS = 500;
  std::vector<MockFragment> fragments;
  
  uint32_t offset = 0;
  for (int i = 0; i < NUM_FRAGMENTS && offset < BUFFER_SIZE / 2; ++i) {
    uint32_t size = 32;  // Small fragments
    if (offset + size <= BUFFER_SIZE / 2) {
      fragments.push_back({offset, size, static_cast<uint32_t>(i + 1000), true});
      offset += size;
    }
  }
  
  if (!fragments.empty()) {
    int ret = runPipeline(fragments);
    EXPECT_EQ(ret, 0);
  }
}

TEST_F(TcpxIntegrationTest, ErrorRecovery) {
  // Test with invalid dmabuf ID (should fail at parser stage)
  std::vector<MockFragment> fragments = {
    {0, 1024, 12345, true}
  };
  
  // Temporarily change expected dmabuf ID
  ParserConfig config;
  config.bounce_buffer = h_bounce_buffer_;
  config.bounce_size = BUFFER_SIZE;
  config.dmabuf_base = h_dmabuf_buffer_;
  config.dmabuf_size = BUFFER_SIZE;
  config.expected_dmabuf_id = 999;  // Wrong ID
  
  parser_->updateConfig(config);
  
  int ret = runPipeline(fragments);
  EXPECT_LT(ret, 0);  // Should fail
  
  // Restore correct config
  config.expected_dmabuf_id = TEST_DMABUF_ID;
  parser_->updateConfig(config);
  
  ret = runPipeline(fragments);
  EXPECT_EQ(ret, 0);  // Should succeed now
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
