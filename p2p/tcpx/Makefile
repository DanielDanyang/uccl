# TCPX Engine Makefile

# Compiler and flags
CUDA_HOME ?= /usr/local/cuda
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64
CXX := g++
CXXFLAGS := -O3 -shared -std=c++17 -fPIC -I../../include -I../../rdma -I$(CUDA_INC) \
	-I../../nccl-plugin-gpudirecttcpx/src -I. -I.. \
	-Wno-pointer-arith -Wno-sign-compare -Wno-unused-variable \
	-Wl,-rpath=/usr/lib/x86_64-linux-gnu -lglog -lgflags -lgtest -lz -lelf -libverbs -lpthread

# Python and pybind11 configuration
PYTHON          ?= python3
PYTHON_CONFIG    = $(PYTHON)-config
PYEXT           := $(shell $(PYTHON_CONFIG) --extension-suffix)
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)

LDFLAGS = -L$(CUDA_LIB) -lcudart -lcuda \
          -Wl,-rpath,$(CUDA_LIB) -lglog -lgflags -lgtest \
          -lz -lelf -libverbs -lpthread -ldl -lpthread

# TCPX engine shared library
ENGINE_SRCS = engine.cc tcpx_transport_minimal.cc pybind_engine.cc
ENGINE_LIB = libuccl_tcpx_engine.so

# Python test entrypoint
PYTHON_TEST = test_tcpx_write.py

.PHONY: all clean test run-test

all: $(ENGINE_LIB)

# Build the TCPX engine shared library used by the Python test harness
$(ENGINE_LIB): $(ENGINE_SRCS)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -o $@ $^ $(LDFLAGS)

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
