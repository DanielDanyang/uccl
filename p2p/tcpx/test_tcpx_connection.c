#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// 包含 UCCL 引擎头文件
#include "../uccl_engine.h"

void print_usage(const char* prog_name) {
    printf("Usage:\n");
    printf("  Server: %s server\n", prog_name);
    printf("  Client: %s client <server_ip>\n", prog_name);
    printf("\n");
    printf("Environment variables:\n");
    printf("  UCCL_TCPX_PLUGIN_PATH - Path to TCPX plugin (default: libnccl-net.so)\n");
    printf("  UCCL_TCPX_DEV - TCPX device ID (default: 0)\n");
}

int test_server() {
    printf("=== TCPX Server Test ===\n");
    
    // 1. 创建引擎
    printf("1. 创建 TCPX 引擎...\n");
    uccl_engine_t* engine = uccl_engine_create(0, 4);
    if (!engine) {
        printf("❌ 引擎创建失败\n");
        return -1;
    }
    printf("✅ 引擎创建成功\n");
    
    // 2. 获取元数据
    printf("2. 获取连接元数据...\n");
    char* metadata = NULL;
    if (uccl_engine_get_metadata(engine, &metadata) != 0 || !metadata) {
        printf("❌ 元数据获取失败\n");
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 元数据: %s\n", metadata);
    
    // 3. 等待客户端连接
    printf("3. 等待客户端连接...\n");
    printf("请在另一个终端运行: %s client <server_ip>\n", "test_tcpx_connection");
    
    char client_ip[64];
    int client_gpu_idx;
    uccl_conn_t* conn = uccl_engine_accept(engine, client_ip, sizeof(client_ip), &client_gpu_idx);
    if (!conn) {
        printf("❌ 接受连接失败\n");
        delete[] metadata;
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 客户端连接成功: %s (GPU %d)\n", client_ip, client_gpu_idx);
    
    // 4. 准备接收数据
    printf("4. 准备接收数据...\n");
    char recv_buffer[1024];
    memset(recv_buffer, 0, sizeof(recv_buffer));
    
    uccl_mr_t* mr = uccl_engine_reg(engine, (uintptr_t)recv_buffer, sizeof(recv_buffer));
    if (!mr) {
        printf("❌ 内存注册失败\n");
        uccl_engine_conn_destroy(conn);
        delete[] metadata;
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 内存注册成功\n");
    
    // 5. 接收数据
    printf("5. 接收数据...\n");
    if (uccl_engine_recv(conn, mr, recv_buffer, sizeof(recv_buffer)) != 0) {
        printf("❌ 数据接收失败\n");
    } else {
        printf("✅ 接收到数据: %s\n", recv_buffer);
    }
    
    // 6. 清理
    printf("6. 清理资源...\n");
    uccl_engine_mr_destroy(mr);
    uccl_engine_conn_destroy(conn);
    delete[] metadata;
    uccl_engine_destroy(engine);
    printf("✅ 清理完成\n");
    
    return 0;
}

int test_client(const char* server_ip) {
    printf("=== TCPX Client Test ===\n");
    printf("连接到服务器: %s\n", server_ip);
    
    // 1. 创建引擎
    printf("1. 创建 TCPX 引擎...\n");
    uccl_engine_t* engine = uccl_engine_create(0, 4);
    if (!engine) {
        printf("❌ 引擎创建失败\n");
        return -1;
    }
    printf("✅ 引擎创建成功\n");
    
    // 2. 连接到服务器
    printf("2. 连接到服务器...\n");
    // 假设服务器监听在端口 12345
    uccl_conn_t* conn = uccl_engine_connect(engine, server_ip, 0, 12345);
    if (!conn) {
        printf("❌ 连接服务器失败\n");
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 连接服务器成功\n");
    
    // 3. 准备发送数据
    printf("3. 准备发送数据...\n");
    const char* message = "Hello from TCPX client!";
    char send_buffer[1024];
    strncpy(send_buffer, message, sizeof(send_buffer) - 1);
    send_buffer[sizeof(send_buffer) - 1] = '\0';
    
    uccl_mr_t* mr = uccl_engine_reg(engine, (uintptr_t)send_buffer, strlen(send_buffer) + 1);
    if (!mr) {
        printf("❌ 内存注册失败\n");
        uccl_engine_conn_destroy(conn);
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 内存注册成功\n");
    
    // 4. 发送数据
    printf("4. 发送数据: %s\n", send_buffer);
    uint64_t transfer_id;
    if (uccl_engine_write(conn, mr, send_buffer, strlen(send_buffer) + 1, &transfer_id) != 0) {
        printf("❌ 数据发送失败\n");
        uccl_engine_mr_destroy(mr);
        uccl_engine_conn_destroy(conn);
        uccl_engine_destroy(engine);
        return -1;
    }
    printf("✅ 数据发送启动，传输ID: %lu\n", transfer_id);
    
    // 5. 等待传输完成
    printf("5. 等待传输完成...\n");
    int timeout = 100; // 10秒超时
    while (timeout-- > 0) {
        if (uccl_engine_xfer_status(conn, transfer_id)) {
            printf("✅ 数据传输完成\n");
            break;
        }
        usleep(100000); // 100ms
    }
    if (timeout <= 0) {
        printf("⚠️ 传输超时，但可能仍在进行中\n");
    }
    
    // 6. 清理
    printf("6. 清理资源...\n");
    uccl_engine_mr_destroy(mr);
    uccl_engine_conn_destroy(conn);
    uccl_engine_destroy(engine);
    printf("✅ 清理完成\n");
    
    return 0;
}

int main(int argc, char* argv[]) {
    printf("TCPX Connection Test\n");
    printf("====================\n");
    
    // 显示环境变量
    printf("环境变量:\n");
    printf("  UCCL_TCPX_PLUGIN_PATH = %s\n", 
           getenv("UCCL_TCPX_PLUGIN_PATH") ? getenv("UCCL_TCPX_PLUGIN_PATH") : "未设置");
    printf("  UCCL_TCPX_DEV = %s\n", 
           getenv("UCCL_TCPX_DEV") ? getenv("UCCL_TCPX_DEV") : "未设置");
    printf("\n");
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (strcmp(argv[1], "server") == 0) {
        return test_server();
    } else if (strcmp(argv[1], "client") == 0) {
        if (argc < 3) {
            printf("错误: 客户端模式需要指定服务器IP\n");
            print_usage(argv[0]);
            return 1;
        }
        return test_client(argv[2]);
    } else {
        printf("错误: 未知模式 '%s'\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }
}
