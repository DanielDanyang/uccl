# TCPX NIXL Plugin Makefile

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++14 -g -O2 -I. -I.. -I../../nccl-plugin-gpudirecttcpx/src -I../../rdma
LDFLAGS = -ldl -lpthread

# TCPX engine shared library
ENGINE_SRCS = engine.cc tcpx_transport_minimal.cc pybind_engine.cc
ENGINE_LIB = libuccl_tcpx_engine.so

# Python test entrypoint
PYTHON_TEST = test_tcpx_write.py

.PHONY: all clean test run-test

all: $(ENGINE_LIB)

# Build the TCPX engine shared library used by the Python test harness
$(ENGINE_LIB): $(ENGINE_SRCS)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $^ $(LDFLAGS)

# Run Python smoke test
test: run-test

run-test: $(ENGINE_LIB)
	@echo "Running TCPX Python smoke test..."
	@echo "Ensure environment variables are set:"
	@echo "  export UCCL_TCPX_PLUGIN_PATH=/path/to/your/tcpx/lib"
	@echo "  export UCCL_TCPX_DEV=0"
	@echo ""
	python3 $(PYTHON_TEST)

# Clean artifacts
clean:
	rm -f $(ENGINE_LIB)

# Help
help:
	@echo "TCPX Engine build options:"
	@echo "  all       - build TCPX engine shared library"
	@echo "  test      - run the Python smoke test"
	@echo "  run-test  - run the Python smoke test"
	@echo "  clean     - remove build artifacts"
	@echo "  help      - show this help"
	@echo ""
	@echo "Environment variables:"
	@echo "  UCCL_TCPX_PLUGIN_PATH - path to NCCL TCPX plugin"
	@echo "  UCCL_TCPX_DEV         - TCPX device index"
