/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Simple build test for TCPX Unpack Architecture
 ************************************************************************/
#include <iostream>
#include <vector>

// Test basic includes
#include "../rx/rx_cmsg_parser.h"
#include "../rx/rx_descriptor.h"

#ifdef __CUDACC__
#include "../device/unpack_launch.h"
#endif

using namespace tcpx::rx;

int main() {
    std::cout << "TCPX Unpack Architecture Build Test\n";
    std::cout << "===================================\n\n";
    
    // Test RX CMSG Parser
    std::cout << "Testing RX CMSG Parser...\n";
    ParserConfig parser_config;
    parser_config.bounce_buffer = nullptr;
    parser_config.bounce_size = 4096;
    parser_config.dmabuf_base = nullptr;
    parser_config.dmabuf_size = 4096;
    parser_config.expected_dmabuf_id = 42;
    
    CmsgParser parser(parser_config);
    std::cout << "✓ CmsgParser created successfully\n";
    
    // Test RX Descriptor Builder
    std::cout << "Testing RX Descriptor Builder...\n";
    DescriptorConfig desc_config;
    desc_config.bounce_buffer = nullptr;
    desc_config.dst_buffer = nullptr;
    desc_config.max_descriptors = 1024;
    
    DescriptorBuilder builder(desc_config);
    std::cout << "✓ DescriptorBuilder created successfully\n";
    
    // Test basic functionality
    std::cout << "Testing basic functionality...\n";
    ScatterList scatter_list;
    UnpackDescriptorBlock desc_block;
    
    int ret = builder.buildDescriptors(scatter_list, desc_block);
    if (ret == 0) {
        std::cout << "✓ Empty descriptor build successful\n";
    } else {
        std::cout << "✗ Empty descriptor build failed\n";
        return 1;
    }
    
#ifdef __CUDACC__
    // Test Device Unpack Launcher (CUDA only)
    std::cout << "Testing Device Unpack Launcher...\n";
    
    using namespace tcpx::device;
    
    UnpackLaunchConfig launch_config;
    launch_config.stream = nullptr;
    launch_config.enable_profiling = false;
    
    UnpackLauncher launcher(launch_config);
    std::cout << "✓ UnpackLauncher created successfully\n";
    
    // Test empty launch
    ret = launcher.launchSync(desc_block);
    if (ret == 0) {
        std::cout << "✓ Empty kernel launch successful\n";
    } else {
        std::cout << "✗ Empty kernel launch failed\n";
        return 1;
    }
#else
    std::cout << "Skipping CUDA tests (not compiled with NVCC)\n";
#endif
    
    // Test statistics
    std::cout << "Testing statistics...\n";
    const auto& parser_stats = parser.getStats();
    const auto& builder_stats = builder.getStats();
    
    std::cout << "Parser stats: " << parser_stats.total_messages << " messages\n";
    std::cout << "Builder stats: " << builder_stats.blocks_built << " blocks\n";
    
#ifdef __CUDACC__
    const auto& launcher_stats = launcher.getStats();
    std::cout << "Launcher stats: " << launcher_stats.launches << " launches\n";
#endif
    
    std::cout << "\n✓ All tests passed!\n";
    std::cout << "TCPX Unpack Architecture build is working correctly.\n";
    
    return 0;
}
