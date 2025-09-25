# TCPX P2P Transport Makefile
CXX = g++
CUDA_HOME ?= /usr/local/cuda
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -I../include -I. -I$(CUDA_HOME)/include
LIBS = -ldl -lpthread
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -lcuda

# Build tests
test_device_discovery:
	$(CXX) $(CXXFLAGS) -o tests/test_device_discovery tests/test_device_discovery.cc tcpx_impl.cc $(LIBS)

test_connection:
	$(CXX) $(CXXFLAGS) -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS) $(CUDA_LIBS)

test_tcpx:
	$(CXX) $(CXXFLAGS) -o tests/test_tcpx tests/test_tcpx.cc tcpx_impl.cc $(LIBS)

test_performance:
	$(CXX) $(CXXFLAGS) -o tests/test_performance tests/test_performance.cc tcpx_impl.cc $(LIBS)

# Build all core tests
all: test_device_discovery test_connection test_performance

# Build tests without CUDA dependency
no-cuda: test_device_discovery test_tcpx test_performance

# Build connection test without CUDA (host memory only)
test_connection_host:
	$(CXX) $(CXXFLAGS) -DNO_CUDA -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS)

# Run device discovery test
test: test_device_discovery
	./tests/test_device_discovery

# Clean built files (only executables, not source code)
clean:
	rm -f tests/test_device_discovery tests/test_connection tests/test_tcpx tests/test_performance *.o

.PHONY: all test clean
