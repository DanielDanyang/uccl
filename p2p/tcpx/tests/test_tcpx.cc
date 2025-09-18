#include "../tcpx_interface.h"
#include <stdio.h>

int main() {
  printf("🧪 TCPX 简化测试\n");
  printf("==================================================\n");

  // Test plugin loading
  printf("🔄 加载 TCPX 插件...\n");
  int result = tcpx_load_plugin("/usr/local/tcpx/lib64/libnccl-net-tcpx.so");
  if (result == 0) {
    printf("✅ TCPX 插件加载成功\n");
  } else {
    printf("❌ TCPX 插件加载失败\n");
  }

  // Test device discovery
  printf("🔄 获取设备数量...\n");
  int device_count = tcpx_get_device_count();
  printf("📊 发现 %d 个 TCPX 设备\n", device_count);

  printf("==================================================\n");
  if (result == 0 && device_count > 0) {
    printf("🎉 TCPX 基础功能测试成功!\n");
    return 0;
  } else {
    printf("❌ TCPX 基础功能测试失败\n");
    return 1;
  }
}
