#include "tcpx_interface.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 全局变量
static void* tcpx_plugin_handle = NULL;
static int tcpx_initialized = 0;

// TCPX 插件函数指针
static int (*tcpx_devices_func)(int* ndev) = NULL;

// 实现基础函数
int tcpx_load_plugin(char const* plugin_path) {
  printf("[TCPX] Loading plugin: %s\n", plugin_path);

  tcpx_plugin_handle = dlopen(plugin_path, RTLD_LAZY);
  if (!tcpx_plugin_handle) {
    printf("[TCPX] Failed to load plugin: %s\n", dlerror());
    return -1;
  }

  // 获取 NCCL 插件结构体
  void* plugin_symbol = dlsym(tcpx_plugin_handle, "ncclNetPlugin_v7");
  if (!plugin_symbol) {
    printf("[TCPX] ncclNetPlugin_v7 not found: %s\n", dlerror());
    dlclose(tcpx_plugin_handle);
    tcpx_plugin_handle = NULL;
    return -1;
  }

  // 从插件结构体中提取函数指针
  void** plugin_funcs = (void**)plugin_symbol;
  tcpx_devices_func = (int (*)(int*))plugin_funcs[2];  // devices 函数在索引 2

  if (!tcpx_devices_func) {
    printf("[TCPX] tcpxDevices function not found\n");
    dlclose(tcpx_plugin_handle);
    tcpx_plugin_handle = NULL;
    return -1;
  }

  printf("[TCPX] Plugin loaded successfully\n");
  tcpx_initialized = 1;
  return 0;
}

int tcpx_get_device_count() {
  if (!tcpx_initialized) {
    // 尝试加载默认插件
    if (tcpx_load_plugin("/usr/local/tcpx/lib64/libnccl-net-tcpx.so") != 0) {
      printf("[TCPX] Failed to load default plugin, returning -1\n");
      return -1;
    }
  }

  int ndev = 0;
  if (tcpx_devices_func) {
    printf("[TCPX] Calling tcpxDevices function...\n");
    int result = tcpx_devices_func(&ndev);
    printf("[TCPX] tcpxDevices returned: %d, ndev: %d\n", result, ndev);

    if (result == 0) {
      printf("[TCPX] Found %d TCPX devices\n", ndev);
      return ndev;
    } else {
      printf("[TCPX] tcpxDevices failed with result: %d\n", result);
      return -1;
    }
  }

  printf("[TCPX] tcpx_devices_func is NULL\n");
  return -1;
}
