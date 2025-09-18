#pragma once

// 最简化的 TCPX 接口
// 目标：替换 RDMA 传输层，让现有 p2p/engine.cc 能使用 TCPX

extern "C" {
// 基础函数
int tcpx_get_device_count();
int tcpx_load_plugin(char const* plugin_path);
}
