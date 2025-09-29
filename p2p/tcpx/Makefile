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

# Core components only
core: $(RX_OBJS) $(DEVICE_OBJS)
	@echo "Core TCPX unpack components built successfully!"
	@echo "Built objects: $(RX_OBJS) $(DEVICE_OBJS)"

# Simple build test (no external dependencies)
test_build_simple: $(RX_OBJS)
	$(CXX) $(CXXFLAGS) -o tests/test_build_simple tests/test_build_simple.cc $(RX_OBJS) $(LIBS)

# Enhanced test with unpack architecture (check if files exist first)
test_tcpx_transfer: $(RX_OBJS) $(DEVICE_OBJS)
	@if [ -f tests/test_tcpx_transfer.cc ] && [ -f tcpx_impl.cc ]; then \
		echo "Building test_tcpx_transfer..."; \
		$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_transfer tests/test_tcpx_transfer.cc tcpx_impl.cc $(RX_OBJS) $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS); \
	else \
		echo "Missing files for test_tcpx_transfer"; \
		echo "Required: tests/test_tcpx_transfer.cc, tcpx_impl.cc"; \
		ls -la tests/test_tcpx_transfer.cc tcpx_impl.cc 2>/dev/null || echo "Files not found"; \
	fi

# Legacy tests (check if files exist)
test_device_discovery:
	@if [ -f tests/test_device_discovery.cc ] && [ -f tcpx_impl.cc ]; then \
		$(CXX) $(CXXFLAGS) -o tests/test_device_discovery tests/test_device_discovery.cc tcpx_impl.cc $(LIBS); \
	else \
		echo "Missing files for test_device_discovery"; \
	fi

test_connection:
	@if [ -f tests/test_connection.cc ] && [ -f tcpx_impl.cc ]; then \
		$(CXX) $(CXXFLAGS) -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS) $(CUDA_LIBS); \
	else \
		echo "Missing files for test_connection"; \
	fi

test_tcpx:
	@if [ -f tests/test_tcpx.cc ] && [ -f tcpx_impl.cc ]; then \
		$(CXX) $(CXXFLAGS) -o tests/test_tcpx tests/test_tcpx.cc tcpx_impl.cc $(LIBS); \
	else \
		echo "Missing files for test_tcpx"; \
	fi

test_performance:
	@if [ -f tests/test_performance.cc ] && [ -f tcpx_impl.cc ]; then \
		$(CXX) $(CXXFLAGS) -o tests/test_performance tests/test_performance.cc tcpx_impl.cc $(LIBS); \
	else \
		echo "Missing files for test_performance"; \
	fi

# Build targets
all: core test_build_simple test_tcpx_transfer

# Quick test - just core components
quick: core test_build_simple
	@echo "Quick build completed!"

# Check what files we have
check:
	@echo "=== File Check ==="
	@echo "Core source files:"
	@ls -la rx/*.cc rx/*.h device/*.cu device/*.h 2>/dev/null || echo "Missing core files"
	@echo ""
	@echo "Implementation files:"
	@ls -la tcpx_impl.cc tcpx_interface.h 2>/dev/null || echo "Missing implementation files"
	@echo ""
	@echo "Test files:"
	@ls -la tests/*.cc tests/*.cu 2>/dev/null || echo "Missing test files"
	@echo ""
	@echo "Built objects:"
	@ls -la $(RX_OBJS) $(DEVICE_OBJS) 2>/dev/null || echo "No objects built yet"

# Clean built files
clean:
	@echo "Cleaning build artifacts..."
	rm -f tests/test_* tests/benchmark_* $(RX_OBJS) $(DEVICE_OBJS) *.o
	@echo "Clean completed."

.PHONY: all core quick check clean test_build_simple test_tcpx_transfer test_device_discovery test_connection test_tcpx test_performance
