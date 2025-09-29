/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 *
 * Performance benchmark for TCPX unpack pipeline
 ************************************************************************/
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#include "../rx/rx_cmsg_parser.h"
#include "../rx/rx_descriptor.h"
#include "../device/unpack_launch.h"

using namespace tcpx::rx;
using namespace tcpx::device;

class TcpxBenchmark {
public:
  TcpxBenchmark() {
    setupCuda();
    setupComponents();
    allocateMemory();
  }
  
  ~TcpxBenchmark() {
    cleanup();
  }
  
  void runBenchmarks() {
    std::cout << "TCPX Unpack Performance Benchmark\n";
    std::cout << "==================================\n\n";
    
    benchmarkSingleFragment();
    benchmarkMultipleFragments();
    benchmarkSmallFragments();
    benchmarkLargeFragments();
    benchmarkMixedSizes();
    benchmarkScalability();
    
    printSummary();
  }

private:
  void setupCuda() {
    cudaSetDevice(0);
    cudaStreamCreate(&stream_);
    
    // Get device properties
    cudaGetDeviceProperties(&device_prop_, 0);
    std::cout << "GPU: " << device_prop_.name << std::endl;
    std::cout << "Memory: " << device_prop_.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Memory Clock: " << device_prop_.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << device_prop_.memoryBusWidth << " bits\n\n";
  }
  
  void setupComponents() {
    // Parser config
    ParserConfig parser_config;
    parser_config.bounce_buffer = h_bounce_buffer_;
    parser_config.bounce_size = BUFFER_SIZE;
    parser_config.dmabuf_base = h_dmabuf_buffer_;
    parser_config.dmabuf_size = BUFFER_SIZE;
    parser_config.expected_dmabuf_id = 42;
    parser_ = std::make_unique<CmsgParser>(parser_config);
    
    // Descriptor builder config
    DescriptorConfig desc_config;
    desc_config.bounce_buffer = d_bounce_buffer_;
    desc_config.dst_buffer = d_dst_buffer_;
    desc_config.max_descriptors = 2048;
    builder_ = std::make_unique<DescriptorBuilder>(desc_config);
    
    // Launcher config
    UnpackLaunchConfig launch_config;
    launch_config.stream = stream_;
    launch_config.enable_profiling = true;
    launcher_ = std::make_unique<UnpackLauncher>(launch_config);
  }
  
  void allocateMemory() {
    // Host memory
    h_bounce_buffer_ = new char[BUFFER_SIZE];
    h_dmabuf_buffer_ = new char[BUFFER_SIZE];
    
    // Device memory
    cudaMalloc(&d_bounce_buffer_, BUFFER_SIZE);
    cudaMalloc(&d_dmabuf_buffer_, BUFFER_SIZE);
    cudaMalloc(&d_dst_buffer_, BUFFER_SIZE);
    
    // Initialize with test pattern
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      h_bounce_buffer_[i] = static_cast<char>(i % 256);
      h_dmabuf_buffer_[i] = static_cast<char>((i + 128) % 256);
    }
    
    cudaMemcpy(d_bounce_buffer_, h_bounce_buffer_, BUFFER_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dmabuf_buffer_, h_dmabuf_buffer_, BUFFER_SIZE, cudaMemcpyHostToDevice);
  }
  
  void cleanup() {
    delete[] h_bounce_buffer_;
    delete[] h_dmabuf_buffer_;
    
    if (d_dst_buffer_) cudaFree(d_dst_buffer_);
    if (d_dmabuf_buffer_) cudaFree(d_dmabuf_buffer_);
    if (d_bounce_buffer_) cudaFree(d_bounce_buffer_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  
  BenchmarkResult runSingleBenchmark(const std::string& name,
                                     const std::vector<FragmentSpec>& fragments,
                                     int iterations) {
    std::cout << "Running " << name << " (" << iterations << " iterations)..." << std::flush;
    
    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;
    result.total_bytes = 0;
    
    for (const auto& frag : fragments) {
      result.total_bytes += frag.size;
    }
    
    // Reset component stats
    parser_->resetStats();
    builder_->resetStats();
    launcher_->resetStats();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
      runSingleIteration(fragments);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    result.total_time_us = duration.count();
    result.avg_time_us = static_cast<double>(result.total_time_us) / iterations;
    result.bandwidth_gbps = (result.total_bytes * iterations / (1024.0 * 1024.0 * 1024.0)) /
                           (result.total_time_us / 1e6);
    
    // Get component stats
    result.parser_stats = parser_->getStats();
    result.builder_stats = builder_->getStats();
    result.launcher_stats = launcher_->getStats();
    
    std::cout << " Done\n";
    return result;
  }
  
  void runSingleIteration(const std::vector<FragmentSpec>& fragments) {
    // Create scatter list
    ScatterList scatter_list;
    uint32_t dst_offset = 0;
    
    for (const auto& frag : fragments) {
      ScatterEntry entry;
      entry.src_ptr = h_dmabuf_buffer_ + frag.offset;
      entry.src_offset = frag.offset;
      entry.dst_offset = dst_offset;
      entry.length = frag.size;
      entry.is_devmem = true;
      entry.token = frag.token;
      
      scatter_list.entries.push_back(entry);
      scatter_list.total_bytes += frag.size;
      scatter_list.devmem_bytes += frag.size;
      dst_offset += frag.size;
    }
    
    // Build descriptors
    UnpackDescriptorBlock desc_block;
    builder_->buildDescriptors(scatter_list, desc_block);
    
    // Copy devmem data to bounce buffer
    for (const auto& entry : scatter_list.entries) {
      cudaMemcpy(
        static_cast<char*>(d_bounce_buffer_) + entry.src_offset,
        static_cast<char*>(d_dmabuf_buffer_) + entry.src_offset,
        entry.length,
        cudaMemcpyDeviceToDevice
      );
    }
    
    // Launch unpack
    launcher_->launchSync(desc_block);
  }
  
  void benchmarkSingleFragment() {
    std::vector<FragmentSpec> fragments = {{0, 4096, 1000}};
    auto result = runSingleBenchmark("Single 4KB Fragment", fragments, 1000);
    results_.push_back(result);
  }
  
  void benchmarkMultipleFragments() {
    std::vector<FragmentSpec> fragments = {
      {0, 1024, 1001},
      {1024, 1024, 1002},
      {2048, 1024, 1003},
      {3072, 1024, 1004}
    };
    auto result = runSingleBenchmark("4x 1KB Fragments", fragments, 1000);
    results_.push_back(result);
  }
  
  void benchmarkSmallFragments() {
    std::vector<FragmentSpec> fragments;
    for (int i = 0; i < 64; ++i) {
      fragments.push_back({static_cast<uint32_t>(i * 64), 64, 
                          static_cast<uint32_t>(2000 + i)});
    }
    auto result = runSingleBenchmark("64x 64B Fragments", fragments, 500);
    results_.push_back(result);
  }
  
  void benchmarkLargeFragments() {
    std::vector<FragmentSpec> fragments = {
      {0, 65536, 3001},      // 64KB
      {65536, 131072, 3002}  // 128KB
    };
    auto result = runSingleBenchmark("Large Fragments (64KB+128KB)", fragments, 200);
    results_.push_back(result);
  }
  
  void benchmarkMixedSizes() {
    std::vector<FragmentSpec> fragments = {
      {0, 16, 4001},         // 16B
      {16, 256, 4002},       // 256B
      {272, 4096, 4003},     // 4KB
      {4368, 65536, 4004}    // 64KB
    };
    auto result = runSingleBenchmark("Mixed Sizes", fragments, 500);
    results_.push_back(result);
  }
  
  void benchmarkScalability() {
    std::vector<int> fragment_counts = {1, 10, 50, 100, 500, 1000};
    
    for (int count : fragment_counts) {
      std::vector<FragmentSpec> fragments;
      for (int i = 0; i < count; ++i) {
        fragments.push_back({static_cast<uint32_t>(i * 128), 128,
                            static_cast<uint32_t>(5000 + i)});
      }
      
      std::string name = std::to_string(count) + "x 128B Fragments";
      auto result = runSingleBenchmark(name, fragments, 100);
      results_.push_back(result);
    }
  }
  
  void printSummary() {
    std::cout << "\nBenchmark Results Summary\n";
    std::cout << "=========================\n\n";
    
    std::cout << std::left << std::setw(25) << "Test Name"
              << std::right << std::setw(12) << "Avg Time (μs)"
              << std::setw(15) << "Bandwidth (GB/s)"
              << std::setw(12) << "Total Bytes"
              << std::setw(10) << "Fragments" << std::endl;
    std::cout << std::string(74, '-') << std::endl;
    
    for (const auto& result : results_) {
      std::cout << std::left << std::setw(25) << result.name
                << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
                << result.avg_time_us
                << std::setw(15) << std::setprecision(2) << result.bandwidth_gbps
                << std::setw(12) << result.total_bytes
                << std::setw(10) << result.launcher_stats.descriptors_processed / result.iterations
                << std::endl;
    }
    
    std::cout << "\nComponent Performance:\n";
    std::cout << "=====================\n";
    
    // Find best performing test for component analysis
    auto best_result = *std::max_element(results_.begin(), results_.end(),
      [](const BenchmarkResult& a, const BenchmarkResult& b) {
        return a.bandwidth_gbps < b.bandwidth_gbps;
      });
    
    std::cout << "Best performance: " << best_result.name << std::endl;
    std::cout << "  Parser: " << best_result.parser_stats.total_messages << " messages processed\n";
    std::cout << "  Builder: " << best_result.builder_stats.blocks_built << " blocks built\n";
    std::cout << "  Launcher: " << best_result.launcher_stats.avg_bandwidth_gbps << " GB/s avg\n";
    
    // Calculate theoretical bandwidth
    float theoretical_bw = calculateTheoreticalBandwidth();
    std::cout << "\nTheoretical memory bandwidth: " << theoretical_bw << " GB/s\n";
    std::cout << "Peak efficiency: " << (best_result.bandwidth_gbps / theoretical_bw * 100) 
              << "%\n";
  }
  
  float calculateTheoreticalBandwidth() {
    float memory_clock_khz = device_prop_.memoryClockRate;
    int memory_bus_width = device_prop_.memoryBusWidth;
    return (memory_clock_khz * 2.0f * memory_bus_width / 8.0f) / 1e6f;
  }
  
  struct FragmentSpec {
    uint32_t offset;
    uint32_t size;
    uint32_t token;
  };
  
  struct BenchmarkResult {
    std::string name;
    int iterations;
    uint64_t total_time_us;
    double avg_time_us;
    size_t total_bytes;
    double bandwidth_gbps;
    ParserStats parser_stats;
    DescriptorStats builder_stats;
    UnpackStats launcher_stats;
  };
  
  static constexpr size_t BUFFER_SIZE = 1024 * 1024;  // 1MB
  
  char* h_bounce_buffer_;
  char* h_dmabuf_buffer_;
  void* d_bounce_buffer_;
  void* d_dmabuf_buffer_;
  void* d_dst_buffer_;
  
  cudaStream_t stream_;
  cudaDeviceProp device_prop_;
  
  std::unique_ptr<CmsgParser> parser_;
  std::unique_ptr<DescriptorBuilder> builder_;
  std::unique_ptr<UnpackLauncher> launcher_;
  
  std::vector<BenchmarkResult> results_;
};

int main() {
  // Check CUDA availability
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA devices available" << std::endl;
    return 1;
  }
  
  try {
    TcpxBenchmark benchmark;
    benchmark.runBenchmarks();
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
