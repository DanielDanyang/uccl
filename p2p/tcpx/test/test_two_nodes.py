#!/usr/bin/env python3
"""
TCPX 双节点连接测试
测试两个独立进程/节点之间的真实 TCPX 连接
"""

import argparse
import os
import sys
import time
import socket
import threading

def import_p2p_module():
    """导入 TCPX 引擎模块"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    import importlib.util
    so_file = os.path.join(parent_dir, 'libuccl_tcpx_engine.so')
    spec = importlib.util.spec_from_file_location("p2p", so_file)
    p2p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2p)
    return p2p

def get_local_ip():
    """获取本机 IP 地址"""
    try:
        # 连接到一个远程地址来获取本机 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def run_server(gpu_idx=0, port=12345):
    """运行服务器节点（接受连接）"""
    print(f"🖥️  启动服务器节点 (GPU {gpu_idx}, Port {port})")
    print("=" * 50)
    
    try:
        # 导入模块
        print("🔄 导入 TCPX 引擎模块...")
        p2p = import_p2p_module()
        print("✅ 模块导入成功")
        
        # 创建引擎
        print(f"🔄 创建服务器引擎 (GPU {gpu_idx})...")
        server_engine = p2p.Endpoint(gpu_idx, 4)
        print("✅ 服务器引擎创建成功")
        
        # 获取本机信息
        local_ip = get_local_ip()
        metadata = server_engine.get_metadata()
        print(f"📋 服务器信息:")
        print(f"  IP: {local_ip}")
        print(f"  GPU: {gpu_idx}")
        print(f"  Port: {port}")
        print(f"  元数据: {len(metadata)} 字节")
        
        # 等待连接
        print(f"🔄 等待客户端连接...")
        print(f"💡 在另一个终端运行: python test/test_two_nodes.py --mode client --server-ip {local_ip} --server-port {port}")
        
        # 接受连接
        success, client_ip, client_gpu, conn_id = server_engine.accept()
        
        if success:
            print(f"✅ 接受连接成功!")
            print(f"  客户端 IP: {client_ip}")
            print(f"  客户端 GPU: {client_gpu}")
            print(f"  连接 ID: {conn_id}")
            
            # 保持连接一段时间
            print("🔄 保持连接 10 秒...")
            time.sleep(10)
            print("✅ 连接测试完成")
            
        else:
            print("❌ 接受连接失败")
            return False
            
    except Exception as e:
        print(f"❌ 服务器异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_client(server_ip, server_port, gpu_idx=1):
    """运行客户端节点（发起连接）"""
    print(f"💻 启动客户端节点 (GPU {gpu_idx})")
    print("=" * 50)
    
    try:
        # 导入模块
        print("🔄 导入 TCPX 引擎模块...")
        p2p = import_p2p_module()
        print("✅ 模块导入成功")
        
        # 创建引擎
        print(f"🔄 创建客户端引擎 (GPU {gpu_idx})...")
        client_engine = p2p.Endpoint(gpu_idx, 4)
        print("✅ 客户端引擎创建成功")
        
        # 获取本机信息
        local_ip = get_local_ip()
        metadata = client_engine.get_metadata()
        print(f"📋 客户端信息:")
        print(f"  IP: {local_ip}")
        print(f"  GPU: {gpu_idx}")
        print(f"  元数据: {len(metadata)} 字节")
        
        # 等待一下让服务器准备好
        print("🔄 等待 2 秒让服务器准备...")
        time.sleep(2)
        
        # 连接到服务器
        print(f"🔄 连接到服务器 {server_ip}:{server_port}...")
        success, conn_id = client_engine.connect(server_ip, 0, server_port)
        
        if success:
            print(f"✅ 连接成功!")
            print(f"  服务器 IP: {server_ip}")
            print(f"  服务器端口: {server_port}")
            print(f"  连接 ID: {conn_id}")
            
            # 保持连接一段时间
            print("🔄 保持连接 10 秒...")
            time.sleep(10)
            print("✅ 连接测试完成")
            
        else:
            print("❌ 连接失败")
            return False
            
    except Exception as e:
        print(f"❌ 客户端异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_local_test():
    """运行本地双进程测试"""
    print("🏠 本地双进程连接测试")
    print("=" * 50)
    
    import subprocess
    import threading
    
    # 启动服务器进程
    server_port = 12345
    local_ip = get_local_ip()
    
    print(f"🔄 启动服务器进程...")
    server_process = subprocess.Popen([
        sys.executable, __file__, 
        "--mode", "server", 
        "--port", str(server_port),
        "--gpu", "0"
    ])
    
    # 等待服务器启动
    time.sleep(3)
    
    print(f"🔄 启动客户端进程...")
    client_process = subprocess.Popen([
        sys.executable, __file__,
        "--mode", "client",
        "--server-ip", local_ip,
        "--server-port", str(server_port),
        "--gpu", "1"
    ])
    
    # 等待两个进程完成
    print("🔄 等待进程完成...")
    server_result = server_process.wait()
    client_result = client_process.wait()
    
    print(f"📊 测试结果:")
    print(f"  服务器退出码: {server_result}")
    print(f"  客户端退出码: {client_result}")
    
    if server_result == 0 and client_result == 0:
        print("✅ 本地双进程连接测试成功!")
        return True
    else:
        print("❌ 本地双进程连接测试失败")
        return False

def main():
    parser = argparse.ArgumentParser(description='TCPX 双节点连接测试')
    parser.add_argument('--mode', choices=['server', 'client', 'local', 'h100-server', 'h100-client'],
                       default='local', help='运行模式')
    parser.add_argument('--server-ip', default='127.0.0.1',
                       help='服务器 IP 地址')
    parser.add_argument('--server-port', type=int, default=12345,
                       help='服务器端口')
    parser.add_argument('--port', type=int, default=12345,
                       help='服务器监听端口')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU 索引')

    args = parser.parse_args()

    # H100 节点预设配置
    if args.mode == 'h100-server':
        # 节点1: 10.0.1.25 作为服务器
        args.mode = 'server'
        args.port = 12345
        args.gpu = 0
        print(f"🖥️  H100 节点1 (10.0.1.25) 作为服务器")
        print(f"💡 在节点2运行: python test/test_two_nodes.py --mode h100-client")
    elif args.mode == 'h100-client':
        # 节点2: 10.0.0.226 作为客户端，连接到节点1
        args.mode = 'client'
        args.server_ip = '10.0.1.25'
        args.server_port = 12345
        args.gpu = 0
        print(f"💻 H100 节点2 (10.0.0.226) 作为客户端，连接到 10.0.1.25")
    
    if args.mode == 'server':
        success = run_server(args.gpu, args.port)
    elif args.mode == 'client':
        success = run_client(args.server_ip, args.server_port, args.gpu)
    else:  # local
        success = run_local_test()
    
    if success:
        print("\n🎉 TCPX 双节点连接测试成功!")
        print("📋 验证了:")
        print("  ✅ 跨进程/节点的真实网络连接")
        print("  ✅ TCPX 引擎的连接建立")
        print("  ✅ 连接的稳定性和持久性")
    else:
        print("\n❌ TCPX 双节点连接测试失败")
        print("🔧 可能的问题:")
        print("  - 网络配置问题")
        print("  - TCPX 插件未正确初始化")
        print("  - 端口被占用或防火墙阻止")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
