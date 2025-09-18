# TCPX P2P Transport Makefile
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -I../include -I.
LIBS = -ldl -lpthread

# Build tests
test_device_discovery:
	$(CXX) $(CXXFLAGS) -o tests/test_device_discovery tests/test_device_discovery.cc tcpx_impl.cc $(LIBS)

test_connection:
	$(CXX) $(CXXFLAGS) -o tests/test_connection tests/test_connection.cc tcpx_impl.cc $(LIBS)

test_tcpx:
	$(CXX) $(CXXFLAGS) -o tests/test_tcpx tests/test_tcpx.cc tcpx_impl.cc $(LIBS)

test_data_transfer:
	$(CXX) $(CXXFLAGS) -o tests/test_data_transfer tests/test_data_transfer.cc tcpx_impl.cc $(LIBS)

# Build all core tests
all: test_device_discovery test_connection test_data_transfer

# Run device discovery test
test: test_device_discovery
	./tests/test_device_discovery

# Clean built files (only executables, not source code)
clean:
	rm -f tests/test_device_discovery tests/test_connection tests/test_tcpx tests/test_data_transfer *.o

.PHONY: all test clean
