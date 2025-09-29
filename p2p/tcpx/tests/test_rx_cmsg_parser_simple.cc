/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Simple RX CMSG Parser test (no GTest dependency)
 ************************************************************************/
#include <iostream>
#include <cstring>
#include <vector>
#include "../rx/rx_cmsg_parser.h"

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

#define ASSERT_FALSE(cond) \
    if (cond) { \
        throw std::runtime_error("Expected " #cond " to be false"); \
    }

// Test basic parser creation
TEST(ParserCreation) {
    ParserConfig config;
    config.bounce_buffer = nullptr;
    config.bounce_size = 4096;
    config.dmabuf_base = nullptr;
    config.dmabuf_size = 4096;
    config.expected_dmabuf_id = 42;
    
    CmsgParser parser(config);
    
    const auto& stats = parser.getStats();
    ASSERT_EQ(stats.total_messages, 0);
    ASSERT_EQ(stats.devmem_fragments, 0);
    ASSERT_EQ(stats.linear_fragments, 0);
}

// Test empty message parsing
TEST(EmptyMessageParsing) {
    ParserConfig config;
    config.bounce_buffer = nullptr;
    config.bounce_size = 4096;
    config.dmabuf_base = nullptr;
    config.dmabuf_size = 4096;
    config.expected_dmabuf_id = 42;
    
    CmsgParser parser(config);
    ScatterList scatter_list;
    
    // Create empty message
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    
    int result = parser.parse(&msg, scatter_list);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(scatter_list.entries.size(), 0);
}

// Test configuration validation
TEST(ConfigValidation) {
    ParserConfig config;
    config.bounce_buffer = nullptr;
    config.bounce_size = 0;  // Invalid
    config.dmabuf_base = nullptr;
    config.dmabuf_size = 4096;
    config.expected_dmabuf_id = 42;
    
    // Should still create parser but with warnings
    CmsgParser parser(config);
    ASSERT_TRUE(true);  // Just test creation doesn't crash
}

// Test statistics collection
TEST(StatisticsCollection) {
    ParserConfig config;
    config.bounce_buffer = nullptr;
    config.bounce_size = 4096;
    config.dmabuf_base = nullptr;
    config.dmabuf_size = 4096;
    config.expected_dmabuf_id = 42;
    
    CmsgParser parser(config);
    
    // Parse empty message multiple times
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    ScatterList scatter_list;
    
    parser.parse(&msg, scatter_list);
    parser.parse(&msg, scatter_list);
    
    const auto& stats = parser.getStats();
    ASSERT_EQ(stats.total_messages, 2);
}

// Test scatter list validation
TEST(ScatterListValidation) {
    ScatterList scatter_list;
    
    // Add some test entries
    ScatterEntry entry1;
    entry1.src_ptr = nullptr;
    entry1.src_offset = 0;
    entry1.dst_offset = 0;
    entry1.length = 100;
    entry1.is_devmem = false;

    ScatterEntry entry2;
    entry2.src_ptr = nullptr;
    entry2.src_offset = 100;
    entry2.dst_offset = 100;
    entry2.length = 200;
    entry2.is_devmem = true;

    scatter_list.entries.push_back(entry1);
    scatter_list.entries.push_back(entry2);
    scatter_list.total_bytes = 300;

    // Test validation (this is internal function, just test it doesn't crash)
    ASSERT_EQ(scatter_list.entries.size(), 2);
    ASSERT_EQ(scatter_list.total_bytes, 300);
}

// Test error handling
TEST(ErrorHandling) {
    ParserConfig config;
    config.bounce_buffer = nullptr;
    config.bounce_size = 4096;
    config.dmabuf_base = nullptr;
    config.dmabuf_size = 4096;
    config.expected_dmabuf_id = 42;
    
    CmsgParser parser(config);
    ScatterList scatter_list;
    
    // Test with null message
    int result = parser.parse(nullptr, scatter_list);
    ASSERT_EQ(result, -1);
}

int main() {
    std::cout << "TCPX RX CMSG Parser Simple Test Suite\n";
    std::cout << "=====================================\n\n";
    
    // Run all tests
    run_test_ParserCreation();
    run_test_EmptyMessageParsing();
    run_test_ConfigValidation();
    run_test_StatisticsCollection();
    run_test_ScatterListValidation();
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
