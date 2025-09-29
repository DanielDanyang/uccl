/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Test suite for RX Descriptor module
 ************************************************************************/
#include <gtest/gtest.h>
#include <chrono>
#include "../rx/rx_descriptor.h"

using namespace tcpx::rx;

class RxDescriptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Setup test configuration
    config_.bounce_buffer = bounce_buffer_;
    config_.dst_buffer = dst_buffer_;
    config_.max_descriptors = 1024;
    
    builder_ = std::make_unique<DescriptorBuilder>(config_);
    
    // Initialize test buffers
    memset(bounce_buffer_, 0xAA, sizeof(bounce_buffer_));
    memset(dst_buffer_, 0xBB, sizeof(dst_buffer_));
  }
  
  void TearDown() override {
    builder_.reset();
  }
  
  // Helper function to create test scatter list
  ScatterList createTestScatterList(const std::vector<TestEntry>& entries) {
    ScatterList scatter_list;
    
    uint32_t dst_offset = 0;
    for (const auto& test_entry : entries) {
      ScatterEntry entry;
      entry.src_ptr = bounce_buffer_ + test_entry.src_offset;
      entry.src_offset = test_entry.src_offset;
      entry.dst_offset = dst_offset;
      entry.length = test_entry.length;
      entry.is_devmem = test_entry.is_devmem;
      entry.token = test_entry.token;
      
      scatter_list.entries.push_back(entry);
      scatter_list.total_bytes += entry.length;
      
      if (entry.is_devmem) {
        scatter_list.devmem_bytes += entry.length;
      } else {
        scatter_list.linear_bytes += entry.length;
      }
      
      dst_offset += entry.length;
    }
    
    return scatter_list;
  }
  
  struct TestEntry {
    uint32_t src_offset;
    uint32_t length;
    bool is_devmem;
    uint32_t token;
  };
  
  static constexpr size_t BUFFER_SIZE = 8192;
  
  DescriptorConfig config_;
  std::unique_ptr<DescriptorBuilder> builder_;
  
  char bounce_buffer_[BUFFER_SIZE];
  char dst_buffer_[BUFFER_SIZE];
};

TEST_F(RxDescriptorTest, EmptyScatterList) {
  ScatterList empty_scatter_list;
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(empty_scatter_list, desc_block);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(desc_block.count, 0);
  EXPECT_EQ(desc_block.total_bytes, 0);
  EXPECT_EQ(desc_block.bounce_buffer, bounce_buffer_);
  EXPECT_EQ(desc_block.dst_buffer, dst_buffer_);
}

TEST_F(RxDescriptorTest, SingleDevMemEntry) {
  std::vector<TestEntry> entries = {
    {100, 256, true, 12345}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(desc_block.count, 1);
  EXPECT_EQ(desc_block.total_bytes, 256);
  
  const auto& desc = desc_block.descriptors[0];
  EXPECT_EQ(desc.src_off, 100);
  EXPECT_EQ(desc.dst_off, 0);
  EXPECT_EQ(desc.len, 256);
}

TEST_F(RxDescriptorTest, MultipleDevMemEntries) {
  std::vector<TestEntry> entries = {
    {0, 128, true, 11111},
    {128, 256, true, 22222},
    {384, 512, true, 33333}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(desc_block.count, 3);
  EXPECT_EQ(desc_block.total_bytes, 896);
  
  // Check each descriptor
  EXPECT_EQ(desc_block.descriptors[0].src_off, 0);
  EXPECT_EQ(desc_block.descriptors[0].dst_off, 0);
  EXPECT_EQ(desc_block.descriptors[0].len, 128);
  
  EXPECT_EQ(desc_block.descriptors[1].src_off, 128);
  EXPECT_EQ(desc_block.descriptors[1].dst_off, 128);
  EXPECT_EQ(desc_block.descriptors[1].len, 256);
  
  EXPECT_EQ(desc_block.descriptors[2].src_off, 384);
  EXPECT_EQ(desc_block.descriptors[2].dst_off, 384);
  EXPECT_EQ(desc_block.descriptors[2].len, 512);
}

TEST_F(RxDescriptorTest, SkipLinearEntries) {
  std::vector<TestEntry> entries = {
    {0, 128, true, 11111},      // DevMem - should be included
    {128, 256, false, 0},       // Linear - should be skipped
    {384, 512, true, 33333}     // DevMem - should be included
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(desc_block.count, 2);  // Only DevMem entries
  EXPECT_EQ(desc_block.total_bytes, 640);  // 128 + 512
  
  EXPECT_EQ(desc_block.descriptors[0].len, 128);
  EXPECT_EQ(desc_block.descriptors[1].len, 512);
}

TEST_F(RxDescriptorTest, TooManyDescriptors) {
  // Create more entries than max_descriptors
  config_.max_descriptors = 2;
  builder_->updateConfig(config_);
  
  std::vector<TestEntry> entries = {
    {0, 128, true, 11111},
    {128, 256, true, 22222},
    {384, 512, true, 33333}  // This exceeds max_descriptors
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  
  EXPECT_LT(ret, 0);  // Should fail
  EXPECT_GT(builder_->getStats().build_errors, 0);
}

TEST_F(RxDescriptorTest, DescriptorValidation) {
  std::vector<TestEntry> entries = {
    {0, 256, true, 12345}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  EXPECT_EQ(ret, 0);
  
  bool valid = builder_->validateDescriptors(desc_block);
  EXPECT_TRUE(valid);
}

TEST_F(RxDescriptorTest, DescriptorAlignment) {
  // Test alignment calculation
  uint32_t align1 = descriptor_utils::calculateOptimalAlignment(0, 16);
  EXPECT_EQ(align1, 16);
  
  uint32_t align2 = descriptor_utils::calculateOptimalAlignment(4, 8);
  EXPECT_EQ(align2, 4);
  
  uint32_t align3 = descriptor_utils::calculateOptimalAlignment(1, 7);
  EXPECT_EQ(align3, 1);
}

TEST_F(RxDescriptorTest, DescriptorMerging) {
  // Create adjacent descriptors that can be merged
  UnpackDescriptorBlock desc_block;
  desc_block.count = 3;
  desc_block.total_bytes = 768;
  desc_block.bounce_buffer = bounce_buffer_;
  desc_block.dst_buffer = dst_buffer_;
  
  // Adjacent descriptors
  desc_block.descriptors[0] = UnpackDescriptor(0, 256, 0);
  desc_block.descriptors[1] = UnpackDescriptor(256, 256, 256);  // Adjacent to first
  desc_block.descriptors[2] = UnpackDescriptor(1024, 256, 512); // Not adjacent
  
  int merged_count = descriptor_utils::mergeDescriptors(desc_block);
  
  EXPECT_EQ(merged_count, 1);  // One descriptor was merged
  EXPECT_EQ(desc_block.count, 2);  // Two descriptors remain
  
  // Check merged descriptor
  EXPECT_EQ(desc_block.descriptors[0].src_off, 0);
  EXPECT_EQ(desc_block.descriptors[0].dst_off, 0);
  EXPECT_EQ(desc_block.descriptors[0].len, 512);  // 256 + 256
  
  // Check non-merged descriptor
  EXPECT_EQ(desc_block.descriptors[1].src_off, 1024);
  EXPECT_EQ(desc_block.descriptors[1].dst_off, 512);
  EXPECT_EQ(desc_block.descriptors[1].len, 256);
}

TEST_F(RxDescriptorTest, DescriptorSplitting) {
  UnpackDescriptorBlock desc_block;
  desc_block.count = 1;
  desc_block.total_bytes = 1024;
  desc_block.bounce_buffer = bounce_buffer_;
  desc_block.dst_buffer = dst_buffer_;
  
  // Large descriptor that needs splitting
  desc_block.descriptors[0] = UnpackDescriptor(0, 1024, 0);
  
  int split_count = descriptor_utils::splitDescriptors(desc_block, 256);
  
  EXPECT_EQ(split_count, 3);  // 1024 / 256 = 4 chunks, so 3 additional descriptors
  EXPECT_EQ(desc_block.count, 4);
  
  // Check split descriptors
  for (uint32_t i = 0; i < desc_block.count; ++i) {
    EXPECT_EQ(desc_block.descriptors[i].src_off, i * 256);
    EXPECT_EQ(desc_block.descriptors[i].dst_off, i * 256);
    EXPECT_EQ(desc_block.descriptors[i].len, 256);
  }
}

TEST_F(RxDescriptorTest, DescriptorBlockValidation) {
  UnpackDescriptorBlock desc_block;
  desc_block.count = 2;
  desc_block.total_bytes = 512;
  desc_block.bounce_buffer = bounce_buffer_;
  desc_block.dst_buffer = dst_buffer_;
  
  desc_block.descriptors[0] = UnpackDescriptor(0, 256, 0);
  desc_block.descriptors[1] = UnpackDescriptor(256, 256, 256);
  
  bool valid = descriptor_utils::validateDescriptorBlock(desc_block);
  EXPECT_TRUE(valid);
  
  // Test invalid case - wrong total_bytes
  desc_block.total_bytes = 1000;
  valid = descriptor_utils::validateDescriptorBlock(desc_block);
  EXPECT_FALSE(valid);
}

TEST_F(RxDescriptorTest, DescriptorDump) {
  std::vector<TestEntry> entries = {
    {0, 256, true, 12345}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  builder_->buildDescriptors(scatter_list, desc_block);
  
  std::string dump = descriptor_utils::dumpDescriptorBlock(desc_block);
  EXPECT_FALSE(dump.empty());
  EXPECT_NE(dump.find("UnpackDescriptorBlock"), std::string::npos);
  EXPECT_NE(dump.find("src_off=0"), std::string::npos);
  EXPECT_NE(dump.find("len=256"), std::string::npos);
}

TEST_F(RxDescriptorTest, StatisticsTracking) {
  std::vector<TestEntry> entries = {
    {0, 256, true, 12345},
    {256, 512, true, 67890}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  // Reset stats
  builder_->resetStats();
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  EXPECT_EQ(ret, 0);
  
  const auto& stats = builder_->getStats();
  EXPECT_EQ(stats.blocks_built, 1);
  EXPECT_EQ(stats.descriptors_created, 2);
  EXPECT_EQ(stats.bytes_processed, 768);
  EXPECT_EQ(stats.build_errors, 0);
}

// Performance test
TEST_F(RxDescriptorTest, PerformanceTest) {
  const int NUM_ITERATIONS = 1000;
  
  std::vector<TestEntry> entries = {
    {0, 256, true, 11111},
    {256, 512, true, 22222},
    {768, 1024, true, 33333}
  };
  
  ScatterList scatter_list = createTestScatterList(entries);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    UnpackDescriptorBlock desc_block;
    int ret = builder_->buildDescriptors(scatter_list, desc_block);
    EXPECT_EQ(ret, 0);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_us = static_cast<double>(duration.count()) / NUM_ITERATIONS;
  std::cout << "Average build time: " << avg_time_us << " microseconds" << std::endl;
  
  // Should be fast (< 5 microseconds per build)
  EXPECT_LT(avg_time_us, 5.0);
}

// Stress test with many descriptors
TEST_F(RxDescriptorTest, StressTest) {
  const int NUM_DESCRIPTORS = 1000;
  
  std::vector<TestEntry> entries;
  for (int i = 0; i < NUM_DESCRIPTORS; ++i) {
    entries.push_back({static_cast<uint32_t>(i * 64), 64, true, 
                      static_cast<uint32_t>(i + 1000)});
  }
  
  ScatterList scatter_list = createTestScatterList(entries);
  UnpackDescriptorBlock desc_block;
  
  int ret = builder_->buildDescriptors(scatter_list, desc_block);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(desc_block.count, NUM_DESCRIPTORS);
  EXPECT_EQ(desc_block.total_bytes, NUM_DESCRIPTORS * 64);
  
  bool valid = builder_->validateDescriptors(desc_block);
  EXPECT_TRUE(valid);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
