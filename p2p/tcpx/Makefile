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
# Note: rx/rx_descriptor.h is now header-only, no .cc files needed
DEVICE_SRCS := device/unpack_kernels.cu device/unpack_launch.cu

# Object files
DEVICE_OBJS := device/unpack_kernels.o device/unpack_launch.o

# Build object files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

device/unpack_kernels.o: device/unpack_kernels.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

device/unpack_launch.o: device/unpack_launch.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Core components
core: $(DEVICE_OBJS)
	@echo "Core TCPX unpack components built successfully!"
	@echo "Built objects: $(DEVICE_OBJS)"

# Main transfer test with unpack architecture
test_tcpx_transfer: $(DEVICE_OBJS)
	@echo "Building test_tcpx_transfer..."
	$(NVCC) $(NVCCFLAGS) -o tests/test_tcpx_transfer tests/test_tcpx_transfer.cc tcpx_impl.cc $(DEVICE_OBJS) $(LIBS) $(CUDA_LIBS)

# Connection handshake test
test_connection:
	@echo "Building test_connection..."
	$(CXX) $(CXXFLAGS) -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS) $(CUDA_LIBS)

# Unit tests removed - rx module is now header-only and tested via integration tests

# Build targets
all: core test_tcpx_transfer test_connection

# All tests (unit tests removed, only integration tests remain)
all_tests: all

# Quick test - just core components
quick: core
	@echo "Quick build completed!"

# Check what files we have
check:
	@echo "=== File Check ==="
	@echo "Core source files:"
	@ls -la rx/*.h device/*.cu device/*.h 2>/dev/null || echo "Missing core files"
	@echo ""
	@echo "Implementation files:"
	@ls -la tcpx_impl.cc tcpx_interface.h 2>/dev/null || echo "Missing implementation files"
	@echo ""
	@echo "Test files:"
	@ls -la tests/*.cc 2>/dev/null || echo "Missing test files"
	@echo ""
	@echo "Built objects:"
	@ls -la $(DEVICE_OBJS) 2>/dev/null || echo "No objects built yet"

# Clean built files (ONLY executables and objects, NOT source files)
clean:
	@echo "Cleaning build artifacts..."
	@echo "Removing executable files..."
	rm -f tests/test_connection tests/test_tcpx_transfer
	@echo "Removing object files..."
	rm -f $(DEVICE_OBJS) *.o
	@echo "Clean completed. Source files preserved."

.PHONY: all core quick check clean all_tests test_tcpx_transfer test_connection
