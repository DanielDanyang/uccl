# TCPX P2P Transport Makefile
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -I../include
LIBS = -ldl -lpthread

# Test output directory
TEST_DIR = tests
TEST_BINS = $(TEST_DIR)/test_device_discovery $(TEST_DIR)/test_connection

# Core TCPX tests
test_device_discovery: $(TEST_DIR)/test_device_discovery.cc tcpx_impl.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/test_device_discovery $(TEST_DIR)/test_device_discovery.cc tcpx_impl.cc $(LIBS)

test_connection: $(TEST_DIR)/test_connection.cc tcpx_impl.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/test_connection $(TEST_DIR)/test_connection.cc tcpx_impl.cc $(LIBS)

# Additional tests (development/experimental)
test_tcpx: $(TEST_DIR)/test_tcpx.cc tcpx_impl.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/test_tcpx $(TEST_DIR)/test_tcpx.cc tcpx_impl.cc $(LIBS)

test_endpoint_tcpx: $(TEST_DIR)/test_endpoint_tcpx.cc tcpx_impl.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_DIR)/test_endpoint_tcpx $(TEST_DIR)/test_endpoint_tcpx.cc tcpx_impl.cc $(LIBS)

# Convenience targets
all: test_device_discovery test_connection

# Run automated tests
test: test_device_discovery
	@echo "=== Running TCPX Tests ==="
	@echo "1. Device discovery test:"
	./$(TEST_DIR)/test_device_discovery
	@echo ""
	@echo "2. Connection test requires two nodes - run manually:"
	@echo "   Node 1: ./$(TEST_DIR)/test_connection server"
	@echo "   Node 2: ./$(TEST_DIR)/test_connection client <node1_ip>"

# Help information
help:
	@echo "TCPX P2P Transport Build Targets:"
	@echo ""
	@echo "Core Tests:"
	@echo "  test_device_discovery - Test TCPX device discovery"
	@echo "  test_connection      - Test connection between two nodes"
	@echo ""
	@echo "Development Tests:"
	@echo "  test_tcpx           - Basic TCPX plugin loading test"
	@echo "  test_endpoint_tcpx  - Endpoint integration demo"
	@echo ""
	@echo "Utility:"
	@echo "  all                 - Build core tests"
	@echo "  test                - Run automated tests"
	@echo "  clean               - Remove all built files"
	@echo "  help                - Show this help"

clean:
	rm -f $(TEST_DIR)/test_* *.o

.PHONY: all test help clean
