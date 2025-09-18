#pragma once

// TCPX 接口 - 替换 RDMA 传输层
// 目标：让现有 p2p/engine.cc 能使用 TCPX 进行节点间连接

#include <cstddef>
#include <cstdint>

extern "C" {
// 基础函数
int tcpx_get_device_count();
int tcpx_load_plugin(char const* plugin_path);

// 连接管理
int tcpx_listen(int dev, void* handle, void** listen_comm);
int tcpx_connect(int dev, void* handle, void** send_comm, void** recv_comm);
int tcpx_accept(void* listen_comm, void** recv_comm, void** send_comm);

// 内存注册
int tcpx_reg_mr(void* comm, void* data, size_t size, int type, void** mhandle);
int tcpx_dereg_mr(void* comm, void* mhandle);

// 数据传输
int tcpx_isend(void* send_comm, void* data, int size, int tag, void* mhandle,
               void** request);
int tcpx_irecv(void* recv_comm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request);
int tcpx_test(void* request, int* done, int* size);

// 连接清理
int tcpx_close_send(void* send_comm);
int tcpx_close_recv(void* recv_comm);
int tcpx_close_listen(void* listen_comm);
}
