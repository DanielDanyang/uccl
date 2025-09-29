/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Simple Device Unpack test (no GTest dependency)
 ************************************************************************/
#include <iostream>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../device/unpack_launch.h"

using namespace tcpx::device;
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// Test launcher creation
TEST(LauncherCreation) {
    UnpackLaunchConfig config;
    config.stream = nullptr;
    config.use_small_kernel = false;
    config.enable_profiling = false;
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    
    const auto& stats = launcher.getStats();
    ASSERT_EQ(stats.launches, 0);
    ASSERT_EQ(stats.total_bytes, 0);
}

// Test empty descriptor block
TEST(EmptyDescriptorBlock) {
    UnpackLaunchConfig config;
    config.stream = nullptr;
    config.use_small_kernel = false;
    config.enable_profiling = false;
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    
    UnpackDescriptorBlock desc_block;
    desc_block.count = 0;
    desc_block.bounce_buffer = nullptr;
    desc_block.dst_buffer = nullptr;
    
    int result = launcher.launchSync(desc_block);
    ASSERT_EQ(result, 0);  // Should handle empty gracefully
}

// Test GPU memory allocation
TEST(GPUMemoryAllocation) {
    void* d_bounce = nullptr;
    void* d_dst = nullptr;
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_bounce, 4096));
    CUDA_CHECK(cudaMalloc(&d_dst, 4096));
    
    ASSERT_TRUE(d_bounce != nullptr);
    ASSERT_TRUE(d_dst != nullptr);
    
    // Clean up
    CUDA_CHECK(cudaFree(d_bounce));
    CUDA_CHECK(cudaFree(d_dst));
}

// Test simple unpack operation
TEST(SimpleUnpackOperation) {
    const size_t buffer_size = 4096;
    void* d_bounce = nullptr;
    void* d_dst = nullptr;
    void* h_src = nullptr;
    void* h_dst = nullptr;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_bounce, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_dst, buffer_size));
    CUDA_CHECK(cudaMallocHost(&h_src, buffer_size));
    CUDA_CHECK(cudaMallocHost(&h_dst, buffer_size));
    
    // Initialize source data
    memset(h_src, 0xAB, buffer_size);
    memset(h_dst, 0x00, buffer_size);
    
    // Copy to GPU bounce buffer
    CUDA_CHECK(cudaMemcpy(d_bounce, h_src, buffer_size, cudaMemcpyHostToDevice));
    
    // Create descriptor
    UnpackDescriptorBlock desc_block;
    desc_block.count = 1;
    desc_block.bounce_buffer = d_bounce;
    desc_block.dst_buffer = d_dst;
    desc_block.descriptors[0].src_off = 0;
    desc_block.descriptors[0].dst_off = 0;
    desc_block.descriptors[0].len = buffer_size;
    
    // Launch unpack
    UnpackLaunchConfig config;
    config.stream = nullptr;
    config.use_small_kernel = false;
    config.enable_profiling = false;
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    int result = launcher.launchSync(desc_block);
    ASSERT_EQ(result, 0);
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, buffer_size, cudaMemcpyDeviceToHost));
    
    // Check first few bytes
    unsigned char* src_bytes = static_cast<unsigned char*>(h_src);
    unsigned char* dst_bytes = static_cast<unsigned char*>(h_dst);
    
    for (int i = 0; i < 16; ++i) {
        if (src_bytes[i] != dst_bytes[i]) {
            throw std::runtime_error("Data mismatch at byte " + std::to_string(i));
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_bounce));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));
}

// Test multiple descriptors
TEST(MultipleDescriptors) {
    const size_t buffer_size = 4096;
    void* d_bounce = nullptr;
    void* d_dst = nullptr;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_bounce, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_dst, buffer_size));
    
    // Create descriptor block with multiple descriptors
    UnpackDescriptorBlock desc_block;
    desc_block.count = 4;
    desc_block.bounce_buffer = d_bounce;
    desc_block.dst_buffer = d_dst;
    
    // Split into 4 1KB chunks
    for (int i = 0; i < 4; ++i) {
        desc_block.descriptors[i].src_off = i * 1024;
        desc_block.descriptors[i].dst_off = i * 1024;
        desc_block.descriptors[i].len = 1024;
    }
    
    // Launch unpack
    UnpackLaunchConfig config;
    config.stream = nullptr;
    config.use_small_kernel = true;  // Use small kernel for multiple small chunks
    config.enable_profiling = false;
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    int result = launcher.launchSync(desc_block);
    ASSERT_EQ(result, 0);
    
    // Clean up
    CUDA_CHECK(cudaFree(d_bounce));
    CUDA_CHECK(cudaFree(d_dst));
}

// Test profiling
TEST(Profiling) {
    UnpackLaunchConfig config;
    config.stream = nullptr;
    config.use_small_kernel = false;
    config.enable_profiling = true;  // Enable profiling
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    
    UnpackDescriptorBlock desc_block;
    desc_block.count = 0;
    desc_block.bounce_buffer = nullptr;
    desc_block.dst_buffer = nullptr;
    
    launcher.launchSync(desc_block);
    
    const auto& stats = launcher.getStats();
    ASSERT_EQ(stats.launches, 1);
}

// Test CUDA stream
TEST(CUDAStream) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    UnpackLaunchConfig config;
    config.stream = stream;
    config.use_small_kernel = false;
    config.enable_profiling = false;
    config.max_descriptors = 1024;
    
    UnpackLauncher launcher(config);
    
    UnpackDescriptorBlock desc_block;
    desc_block.count = 0;
    desc_block.bounce_buffer = nullptr;
    desc_block.dst_buffer = nullptr;
    
    int result = launcher.launch(desc_block, stream);
    ASSERT_EQ(result, 0);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    std::cout << "TCPX Device Unpack Simple Test Suite\n";
    std::cout << "====================================\n\n";
    
    // Check CUDA availability
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "No CUDA devices available, skipping tests\n";
        return 0;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)\n\n";
    
    // Run all tests
    run_test_LauncherCreation();
    run_test_EmptyDescriptorBlock();
    run_test_GPUMemoryAllocation();
    run_test_SimpleUnpackOperation();
    run_test_MultipleDescriptors();
    run_test_Profiling();
    run_test_CUDAStream();
    
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
