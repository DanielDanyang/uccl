# TCPX NIXL Plugin Makefile

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -g -O2 -I. -I..
LDFLAGS = -ldl -lpthread

# 目标文件
PLUGIN_SRC = uccl_engine_tcpx_nixl.cc
PLUGIN_OBJ = uccl_engine_tcpx_nixl.o
PLUGIN_LIB = libuccl_tcpx_nixl.so

# Python 测试
PYTHON_TEST = test_tcpx_write.py

.PHONY: all clean test plugin run-test

all: plugin

# 编译插件
plugin: $(PLUGIN_LIB)

$(PLUGIN_LIB): $(PLUGIN_OBJ)
	$(CXX) -shared -o $@ $< $(LDFLAGS)

$(PLUGIN_OBJ): $(PLUGIN_SRC)
	$(CXX) $(CXXFLAGS) -fPIC -c -o $@ $<

# 运行 Python 测试
test: run-test

run-test: plugin
	@echo "运行 TCPX Python 测试..."
	@echo "确保设置了正确的环境变量:"
	@echo "  export UCCL_TCPX_PLUGIN_PATH=/path/to/your/tcpx/lib"
	@echo "  export UCCL_TCPX_DEV=0"
	@echo ""
	python3 $(PYTHON_TEST)

# 清理
clean:
	rm -f $(PLUGIN_OBJ) $(PLUGIN_LIB)

# 显示帮助
help:
	@echo "TCPX NIXL Plugin 构建选项:"
	@echo "  all       - 编译插件"
	@echo "  plugin    - 编译插件"
	@echo "  test      - 运行 Python 测试"
	@echo "  run-test  - 运行 Python 测试"
	@echo "  clean     - 清理编译文件"
	@echo "  help      - 显示此帮助"
	@echo ""
	@echo "环境变量:"
	@echo "  UCCL_TCPX_PLUGIN_PATH - TCPX 插件库路径"
	@echo "  UCCL_TCPX_DEV         - TCPX 设备 ID"
