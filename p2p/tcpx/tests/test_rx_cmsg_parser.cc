/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Test suite for RX CMSG Parser module
 ************************************************************************/
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <cstring>
#include "../rx/rx_cmsg_parser.h"

using namespace tcpx::rx;

class RxCmsgParserTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Setup test configuration
    config_.bounce_buffer = bounce_buffer_;
    config_.bounce_size = sizeof(bounce_buffer_);
    config_.dmabuf_base = dmabuf_buffer_;
    config_.dmabuf_size = sizeof(dmabuf_buffer_);
    
    parser_ = std::make_unique<CmsgParser>(config_);
    
    // Initialize test data
    memset(bounce_buffer_, 0xAA, sizeof(bounce_buffer_));
    memset(dmabuf_buffer_, 0xBB, sizeof(dmabuf_buffer_));
  }
  
  void TearDown() override {
    parser_.reset();
  }
  
  // Helper function to create mock msghdr with control messages
  struct msghdr createMockMessage(const std::vector<MockCmsg>& cmsgs) {
    struct msghdr msg = {};
    
    // Calculate total control message size
    size_t total_size = 0;
    for (const auto& cmsg : cmsgs) {
      total_size += CMSG_SPACE(cmsg.data_size);
    }
    
    if (total_size > sizeof(control_buffer_)) {
      throw std::runtime_error("Control buffer too small");
    }
    
    msg.msg_control = control_buffer_;
    msg.msg_controllen = total_size;
    
    // Build control messages
    char* ptr = control_buffer_;
    for (const auto& mock_cmsg : cmsgs) {
      struct cmsghdr* cmsg = reinterpret_cast<struct cmsghdr*>(ptr);
      cmsg->cmsg_len = CMSG_LEN(mock_cmsg.data_size);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = mock_cmsg.type;
      
      memcpy(CMSG_DATA(cmsg), mock_cmsg.data, mock_cmsg.data_size);
      
      ptr += CMSG_SPACE(mock_cmsg.data_size);
    }
    
    return msg;
  }
  
  struct MockCmsg {
    int type;
    size_t data_size;
    const void* data;
  };
  
  static constexpr size_t BUFFER_SIZE = 4096;
  
  ParserConfig config_;
  std::unique_ptr<CmsgParser> parser_;
  
  char bounce_buffer_[BUFFER_SIZE];
  char dmabuf_buffer_[BUFFER_SIZE];
  char control_buffer_[1024];
};

TEST_F(RxCmsgParserTest, EmptyMessage) {
  struct msghdr msg = {};
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  
  EXPECT_EQ(ret, 0);
  EXPECT_TRUE(scatter_list.entries.empty());
  EXPECT_EQ(scatter_list.total_bytes, 0);
  EXPECT_EQ(scatter_list.devmem_bytes, 0);
  EXPECT_EQ(scatter_list.linear_bytes, 0);
}

TEST_F(RxCmsgParserTest, SingleDevMemFragment) {
  DevMemFragment frag = {
    .frag_offset = 100,
    .frag_size = 256,
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(scatter_list.entries.size(), 1);
  EXPECT_EQ(scatter_list.total_bytes, 256);
  EXPECT_EQ(scatter_list.devmem_bytes, 256);
  EXPECT_EQ(scatter_list.linear_bytes, 0);
  
  const auto& entry = scatter_list.entries[0];
  EXPECT_TRUE(entry.is_devmem);
  EXPECT_EQ(entry.src_offset, 100);
  EXPECT_EQ(entry.dst_offset, 0);
  EXPECT_EQ(entry.length, 256);
  EXPECT_EQ(entry.token, 12345);
  EXPECT_EQ(entry.src_ptr, dmabuf_buffer_ + 100);
}

TEST_F(RxCmsgParserTest, SingleLinearFragment) {
  uint32_t frag_size = 128;
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_LINEAR, sizeof(frag_size), &frag_size}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(scatter_list.entries.size(), 1);
  EXPECT_EQ(scatter_list.total_bytes, 128);
  EXPECT_EQ(scatter_list.devmem_bytes, 0);
  EXPECT_EQ(scatter_list.linear_bytes, 128);
  
  const auto& entry = scatter_list.entries[0];
  EXPECT_FALSE(entry.is_devmem);
  EXPECT_EQ(entry.src_offset, 0);
  EXPECT_EQ(entry.dst_offset, 0);
  EXPECT_EQ(entry.length, 128);
  EXPECT_EQ(entry.token, 0);
  EXPECT_EQ(entry.src_ptr, bounce_buffer_);
}

TEST_F(RxCmsgParserTest, MixedFragments) {
  DevMemFragment devmem_frag = {
    .frag_offset = 0,
    .frag_size = 512,
    .frag_token = 11111,
    };
  
  uint32_t linear_size = 256;
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(devmem_frag), &devmem_frag},
    {SCM_DEVMEM_LINEAR, sizeof(linear_size), &linear_size}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(scatter_list.entries.size(), 2);
  EXPECT_EQ(scatter_list.total_bytes, 768);
  EXPECT_EQ(scatter_list.devmem_bytes, 512);
  EXPECT_EQ(scatter_list.linear_bytes, 256);
  
  // Check devmem entry
  const auto& devmem_entry = scatter_list.entries[0];
  EXPECT_TRUE(devmem_entry.is_devmem);
  EXPECT_EQ(devmem_entry.dst_offset, 0);
  EXPECT_EQ(devmem_entry.length, 512);
  
  // Check linear entry
  const auto& linear_entry = scatter_list.entries[1];
  EXPECT_FALSE(linear_entry.is_devmem);
  EXPECT_EQ(linear_entry.dst_offset, 512);
  EXPECT_EQ(linear_entry.length, 256);
}

TEST_F(RxCmsgParserTest, MissingDmabufBase) {
  ParserConfig cfg;
  cfg.bounce_buffer = bounce_buffer_;
  cfg.bounce_size = sizeof(bounce_buffer_);
  cfg.dmabuf_base = nullptr;
  cfg.dmabuf_size = 0;
  parser_->updateConfig(cfg);

  DevMemFragment frag = {
    .frag_offset = 0,
    .frag_size = 256,
    .frag_token = 12345,
    .dmabuf_id = 0
  };

  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };

  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;

  int ret = parser_->parse(&msg, scatter_list);

  EXPECT_LT(ret, 0);  // Should fail due to missing dmabuf backing
  EXPECT_GT(parser_->getStats().validation_errors, 0);

  parser_->updateConfig(config_);
}

TEST_F(RxCmsgParserTest, OutOfBoundsFragment) {
  DevMemFragment frag = {
    .frag_offset = BUFFER_SIZE - 100,  // Near end of buffer
    .frag_size = 256,                  // Extends beyond buffer
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  
  EXPECT_LT(ret, 0);  // Should fail
  EXPECT_GT(parser_->getStats().validation_errors, 0);
}

TEST_F(RxCmsgParserTest, ValidationSuccess) {
  DevMemFragment frag = {
    .frag_offset = 0,
    .frag_size = 256,
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  EXPECT_EQ(ret, 0);
  
  bool valid = parser_->validate(scatter_list, 256);
  EXPECT_TRUE(valid);
}

TEST_F(RxCmsgParserTest, ValidationSizeMismatch) {
  DevMemFragment frag = {
    .frag_offset = 0,
    .frag_size = 256,
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  int ret = parser_->parse(&msg, scatter_list);
  EXPECT_EQ(ret, 0);
  
  bool valid = parser_->validate(scatter_list, 512);  // Wrong expected size
  EXPECT_FALSE(valid);
}

TEST_F(RxCmsgParserTest, UtilityFunctions) {
  DevMemFragment frag = {
    .frag_offset = 100,
    .frag_size = 256,
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  ScatterList scatter_list;
  
  parser_->parse(&msg, scatter_list);
  
  // Test utility functions
  size_t total_size = utils::calculateTotalSize(scatter_list);
  EXPECT_EQ(total_size, 256);
  
  bool valid = utils::validateScatterList(scatter_list, 256);
  EXPECT_TRUE(valid);
  
  std::string dump = utils::dumpScatterList(scatter_list);
  EXPECT_FALSE(dump.empty());
  EXPECT_NE(dump.find("DEVMEM"), std::string::npos);
}

// Performance test
TEST_F(RxCmsgParserTest, PerformanceTest) {
  const int NUM_ITERATIONS = 1000;
  
  DevMemFragment frag = {
    .frag_offset = 0,
    .frag_size = 1024,
    .frag_token = 12345,
    .dmabuf_id = 0
  };
  
  std::vector<MockCmsg> cmsgs = {
    {SCM_DEVMEM_DMABUF, sizeof(frag), &frag}
  };
  
  struct msghdr msg = createMockMessage(cmsgs);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    ScatterList scatter_list;
    int ret = parser_->parse(&msg, scatter_list);
    EXPECT_EQ(ret, 0);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_us = static_cast<double>(duration.count()) / NUM_ITERATIONS;
  std::cout << "Average parse time: " << avg_time_us << " microseconds" << std::endl;
  
  // Should be fast (< 10 microseconds per parse)
  EXPECT_LT(avg_time_us, 10.0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

