# TCPX Unpack Architecture Makefile

# Toolchain
CXX        := g++
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

# Flags
CXXFLAGS   := -std=c++17 -fPIC -O2 -Wall -I../include -I. -I$(CUDA_HOME)/include
NVCCFLAGS  := -std=c++17 -Xcompiler "-fPIC -O2 -Wall" -I../include -I. -I$(CUDA_HOME)/include
LIBS       := -ldl -lpthread
CUDA_LIBS  := -L$(CUDA_HOME)/lib64 -lcuda -lcudart

# Source files
RX_SRCS    := rx/rx_cmsg_parser.cc rx/rx_descriptor.cc
DEVICE_SRCS := device/unpack_kernels.cu device/unpack_launch.cu
UNPACK_SRCS := $(RX_SRCS) $(DEVICE_SRCS)

# Object files
RX_OBJS    := $(RX_SRCS:.cc=.o)
DEVICE_OBJS := device/unpack_kernels.o device/unpack_launch.o

# Build object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

device/unpack_kernels.o: device/unpack_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

device/unpack_launch.o: device/unpack_launch.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Original tests (legacy)
test_device_discovery:
	$(CXX) $(CXXFLAGS) -o tests/test_device_discovery tests/test_device_discovery.cc tcpx_impl.cc $(LIBS)

test_connection:
	$(CXX) $(CXXFLAGS) -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS) $(CUDA_LIBS)

test_tcpx:
	$(CXX) $(CXXFLAGS) -o tests/test_tcpx tests/test_tcpx.cc tcpx_impl.cc $(LIBS)

test_performance:
	$(CXX) $(CXXFLAGS) -o tests/test_performance tests/test_performance.cc tcpx_impl.cc $(LIBS)

# Enhanced test with unpack architecture
test_tcpx_transfer: $(RX_OBJS) $(DEVICE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_transfer tests/test_tcpx_transfer.cc tcpx_impl.cc $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS)

# New modular tests
test_rx_cmsg_parser: $(RX_OBJS)
	$(CXX) $(CXXFLAGS) -o tests/test_rx_cmsg_parser tests/test_rx_cmsg_parser.cc $(RX_OBJS) $(LIBS) -lgtest -lgtest_main

test_rx_descriptor: $(RX_OBJS)
	$(CXX) $(CXXFLAGS) -o tests/test_rx_descriptor tests/test_rx_descriptor.cc $(RX_OBJS) $(LIBS) -lgtest -lgtest_main

test_device_unpack: $(RX_OBJS) $(DEVICE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o tests/test_device_unpack tests/test_device_unpack.cu $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS) -lgtest -lgtest_main

test_tcpx_integration: $(RX_OBJS) $(DEVICE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_integration tests/test_tcpx_integration.cc $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS) -lgtest -lgtest_main

benchmark_tcpx_unpack: $(RX_OBJS) $(DEVICE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o tests/benchmark_tcpx_unpack tests/benchmark_tcpx_unpack.cc $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS)

# Simple build test
test_build_simple: $(RX_OBJS)
	$(CXX) $(CXXFLAGS) -o tests/test_build_simple tests/test_build_simple.cc $(RX_OBJS) $(LIBS)

test_build_simple_cuda: $(RX_OBJS) $(DEVICE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o tests/test_build_simple_cuda tests/test_build_simple.cc $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS)

# Build targets
all: $(RX_OBJS) $(DEVICE_OBJS) test_device_discovery test_connection test_tcpx_transfer test_performance test_build_simple

# New modular architecture tests
unpack_tests: test_rx_cmsg_parser test_rx_descriptor test_device_unpack test_tcpx_integration

# All tests including new architecture
all_tests: all unpack_tests benchmark_tcpx_unpack

# Build tests without CUDA dependency
no-cuda: test_device_discovery test_tcpx test_performance test_build_simple

# Quick build test
quick_test: test_build_simple

# Simple build everything
build_all: $(RX_OBJS) $(DEVICE_OBJS) test_tcpx_transfer
	@echo "All TCPX unpack components built successfully!"
	@echo "Available executables:"
	@ls -la tests/test_* 2>/dev/null || echo "No test executables found"

# Build connection test without CUDA (host memory only)
test_connection_host:
	$(CXX) $(CXXFLAGS) -DNO_CUDA -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS)

# Run tests
test: test_device_discovery
	./tests/test_device_discovery

run_unpack_tests: unpack_tests
	@echo "Running RX CMSG Parser tests..."
	./tests/test_rx_cmsg_parser
	@echo "Running RX Descriptor tests..."
	./tests/test_rx_descriptor
	@echo "Running Device Unpack tests..."
	./tests/test_device_unpack
	@echo "Running Integration tests..."
	./tests/test_tcpx_integration

run_benchmark: benchmark_tcpx_unpack
	@echo "Running performance benchmark..."
	./tests/benchmark_tcpx_unpack

# Clean built files
clean:
	@echo "Cleaning build artifacts..."
	rm -f tests/test_* tests/benchmark_* $(RX_OBJS) $(DEVICE_OBJS) *.o
	@echo "Clean completed."

.PHONY: all test clean unpack_tests all_tests no-cuda run_unpack_tests run_benchmark build_all quick_test
