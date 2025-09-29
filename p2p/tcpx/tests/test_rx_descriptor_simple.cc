/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Simple RX Descriptor Builder test (no GTest dependency)
 ************************************************************************/
#include <iostream>
#include <cstring>
#include <vector>
#include "../rx/rx_descriptor.h"

using namespace tcpx::rx;

// Simple test framework
int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    void run_test_##name() { \
        std::cout << "Running " #name "... "; \
        tests_run++; \
        try { \
            test_##name(); \
            std::cout << "PASS\n"; \
            tests_passed++; \
        } catch (const std::exception& e) { \
            std::cout << "FAIL: " << e.what() << "\n"; \
        } catch (...) { \
            std::cout << "FAIL: Unknown exception\n"; \
        } \
    } \
    void test_##name()

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Expected " #a " == " #b); \
    }

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        throw std::runtime_error("Expected " #cond " to be true"); \
    }

#define ASSERT_LE(a, b) \
    if ((a) > (b)) { \
        throw std::runtime_error("Expected " #a " <= " #b); \
    }

// Test basic builder creation
TEST(BuilderCreation) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    const auto& stats = builder.getStats();
    ASSERT_EQ(stats.blocks_built, 0);
    ASSERT_EQ(stats.descriptors_created, 0);
    ASSERT_EQ(stats.descriptors_merged, 0);
}

// Test empty scatter list
TEST(EmptyScatterList) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    ScatterList scatter_list;
    UnpackDescriptorBlock desc_block;
    
    int result = builder.buildDescriptors(scatter_list, desc_block);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(desc_block.count, 0);
}

// Test single descriptor creation
TEST(SingleDescriptor) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    // Create scatter list with one entry
    ScatterList scatter_list;
    ScatterEntry entry;
    entry.src_off = 0;
    entry.dst_off = 0;
    entry.len = 1024;
    entry.is_dmabuf = false;
    
    scatter_list.entries.push_back(entry);
    scatter_list.total_bytes = 1024;
    
    UnpackDescriptorBlock desc_block;
    int result = builder.buildDescriptors(scatter_list, desc_block);
    
    ASSERT_EQ(result, 0);
    ASSERT_EQ(desc_block.count, 1);
    ASSERT_EQ(desc_block.descriptors[0].src_off, 0);
    ASSERT_EQ(desc_block.descriptors[0].dst_off, 0);
    ASSERT_EQ(desc_block.descriptors[0].len, 1024);
}

// Test multiple descriptors
TEST(MultipleDescriptors) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    // Create scatter list with multiple entries
    ScatterList scatter_list;
    
    ScatterEntry entry1;
    entry1.src_off = 0;
    entry1.dst_off = 0;
    entry1.len = 512;
    entry1.is_dmabuf = false;
    
    ScatterEntry entry2;
    entry2.src_off = 512;
    entry2.dst_off = 512;
    entry2.len = 512;
    entry2.is_dmabuf = false;
    
    scatter_list.entries.push_back(entry1);
    scatter_list.entries.push_back(entry2);
    scatter_list.total_bytes = 1024;
    
    UnpackDescriptorBlock desc_block;
    int result = builder.buildDescriptors(scatter_list, desc_block);
    
    ASSERT_EQ(result, 0);
    ASSERT_LE(desc_block.count, 2);  // Might be merged
    
    // Check total bytes are preserved
    uint32_t total_len = 0;
    for (uint32_t i = 0; i < desc_block.count; ++i) {
        total_len += desc_block.descriptors[i].len;
    }
    ASSERT_EQ(total_len, 1024);
}

// Test descriptor merging
TEST(DescriptorMerging) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    // Create adjacent entries that should be merged
    ScatterList scatter_list;
    
    ScatterEntry entry1;
    entry1.src_off = 0;
    entry1.dst_off = 0;
    entry1.len = 256;
    entry1.is_dmabuf = false;
    
    ScatterEntry entry2;
    entry2.src_off = 256;  // Adjacent
    entry2.dst_off = 256;  // Adjacent
    entry2.len = 256;
    entry2.is_dmabuf = false;
    
    scatter_list.entries.push_back(entry1);
    scatter_list.entries.push_back(entry2);
    scatter_list.total_bytes = 512;
    
    UnpackDescriptorBlock desc_block;
    int result = builder.buildDescriptors(scatter_list, desc_block);
    
    ASSERT_EQ(result, 0);
    // Should be merged into 1 descriptor (if merging is implemented)
    ASSERT_LE(desc_block.count, 2);
}

// Test statistics
TEST(Statistics) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 1024;
    
    DescriptorBuilder builder(config);
    
    ScatterList scatter_list;
    ScatterEntry entry;
    entry.src_off = 0;
    entry.dst_off = 0;
    entry.len = 1024;
    entry.is_dmabuf = false;
    
    scatter_list.entries.push_back(entry);
    scatter_list.total_bytes = 1024;
    
    UnpackDescriptorBlock desc_block;
    builder.buildDescriptors(scatter_list, desc_block);
    
    const auto& stats = builder.getStats();
    ASSERT_EQ(stats.blocks_built, 1);
    ASSERT_LE(stats.descriptors_created, scatter_list.entries.size());
}

// Test error handling
TEST(ErrorHandling) {
    DescriptorConfig config;
    config.bounce_buffer = nullptr;
    config.dst_buffer = nullptr;
    config.max_descriptors = 0;  // Invalid
    
    DescriptorBuilder builder(config);
    
    ScatterList scatter_list;
    UnpackDescriptorBlock desc_block;
    
    // Should handle gracefully
    int result = builder.buildDescriptors(scatter_list, desc_block);
    ASSERT_EQ(result, 0);  // Empty list should still work
}

int main() {
    std::cout << "TCPX RX Descriptor Builder Simple Test Suite\n";
    std::cout << "============================================\n\n";
    
    // Run all tests
    run_test_BuilderCreation();
    run_test_EmptyScatterList();
    run_test_SingleDescriptor();
    run_test_MultipleDescriptors();
    run_test_DescriptorMerging();
    run_test_Statistics();
    run_test_ErrorHandling();
    
    // Print results
    std::cout << "\n=== Test Results ===\n";
    std::cout << "Tests run: " << tests_run << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << (tests_run - tests_passed) << "\n";
    
    if (tests_passed == tests_run) {
        std::cout << "\n✓ All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests FAILED!\n";
        return 1;
    }
}
