#!/usr/bin/env python3
"""
真实网络连接诊断测试
验证 TCPX 是否真正建立了跨节点网络连接
"""

import argparse
import os
import sys
import time
import socket
import subprocess

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

def check_network_connectivity(target_ip, target_port):
    """检查网络连通性"""
    print(f"🔍 检查网络连通性: {target_ip}:{target_port}")
    
    try:
        # 尝试 TCP 连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((target_ip, target_port))
        sock.close()
        
        if result == 0:
            print(f"  ✅ TCP 连接成功")
            return True
        else:
            print(f"  ❌ TCP 连接失败 (错误码: {result})")
            return False
    except Exception as e:
        print(f"  ❌ 连接异常: {e}")
        return False

def run_network_diagnostic():
    """运行网络诊断"""
    print("🔬 TCPX 真实网络连接诊断")
    print("=" * 50)
    
    # 检查本机网络信息
    print("📋 本机网络信息:")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"  主机名: {hostname}")
        print(f"  本机 IP: {local_ip}")
        
        # 获取所有网络接口
        result = subprocess.run(['ip', 'addr', 'show'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'inet ' in line and '127.0.0.1' not in line:
                    print(f"  网络接口: {line.strip()}")
        
    except Exception as e:
        print(f"  ⚠️  获取网络信息失败: {e}")
    
    print()
    
    # 测试 TCPX 引擎的网络行为
    print("🧪 测试 TCPX 引擎网络行为:")
    
    try:
        p2p = import_p2p_module()
        
        # 创建引擎
        print("  🔄 创建 TCPX 引擎...")
        engine = p2p.Endpoint(0, 4)
        print("  ✅ 引擎创建成功")
        
        # 测试连接到不存在的服务器
        print("  🔄 测试连接到不存在的服务器...")
        fake_ip = "192.168.999.999"  # 不存在的 IP
        fake_port = 99999
        
        start_time = time.time()
        success, conn_id = engine.connect(fake_ip, 0, fake_port)
        end_time = time.time()
        
        print(f"  📊 连接结果:")
        print(f"    成功: {success}")
        print(f"    连接 ID: {conn_id}")
        print(f"    耗时: {end_time - start_time:.3f} 秒")
        
        if success and (end_time - start_time) < 0.1:
            print("  ⚠️  警告: 连接过快，可能是模拟连接而非真实网络连接")
        elif success:
            print("  ✅ 连接成功，似乎是真实网络连接")
        else:
            print("  ❌ 连接失败，这是预期的（因为目标不存在）")
        
        # 测试接受连接
        print("  🔄 测试接受连接...")
        start_time = time.time()
        success, ip_addr, gpu_idx, conn_id = engine.accept()
        end_time = time.time()
        
        print(f"  📊 接受结果:")
        print(f"    成功: {success}")
        print(f"    客户端 IP: {ip_addr}")
        print(f"    客户端 GPU: {gpu_idx}")
        print(f"    连接 ID: {conn_id}")
        print(f"    耗时: {end_time - start_time:.3f} 秒")
        
        if success and ip_addr == "127.0.0.1" and (end_time - start_time) < 0.1:
            print("  ⚠️  警告: 立即返回 127.0.0.1，这是模拟接受而非真实网络接受")
        elif success:
            print("  ✅ 接受成功，似乎是真实网络接受")
        else:
            print("  ❌ 接受失败")
        
    except Exception as e:
        print(f"  ❌ TCPX 引擎测试异常: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 诊断结论
    print("🔍 诊断结论:")
    print("  当前 TCPX 实现的问题:")
    print("  1. ❌ 连接函数立即返回成功，没有真实网络连接")
    print("  2. ❌ 接受函数立即返回 127.0.0.1，没有真实网络监听")
    print("  3. ❌ 没有调用真正的 TCPX 插件 API")
    print()
    print("  需要的改进:")
    print("  1. ✅ 实现真正的 tcpxListen() 调用")
    print("  2. ✅ 实现真正的 tcpxConnect_v5() 调用")
    print("  3. ✅ 实现真正的 tcpxAccept_v5() 调用")
    print("  4. ✅ 从真实连接中获取客户端 IP 信息")

def run_real_server(port=12345):
    """运行真实的 TCP 服务器用于对比"""
    print(f"🖥️  启动真实 TCP 服务器 (端口 {port})")
    print("=" * 50)
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(1)
        
        print(f"✅ 服务器监听在端口 {port}")
        print("💡 在另一个终端运行: telnet <server_ip> 12345")
        print("🔄 等待连接...")
        
        client_socket, client_address = server_socket.accept()
        print(f"✅ 接受连接来自: {client_address[0]}:{client_address[1]}")
        
        # 发送欢迎消息
        client_socket.send(b"Hello from real TCP server!\n")
        
        # 保持连接 10 秒
        time.sleep(10)
        
        client_socket.close()
        server_socket.close()
        
        print("✅ 真实 TCP 服务器测试完成")
        print(f"📋 对比: 真实服务器能正确获取客户端 IP {client_address[0]}")
        
    except Exception as e:
        print(f"❌ 真实 TCP 服务器异常: {e}")

def main():
    parser = argparse.ArgumentParser(description='TCPX 真实网络连接诊断')
    parser.add_argument('--mode', choices=['diagnostic', 'tcp-server'], 
                       default='diagnostic', help='运行模式')
    parser.add_argument('--port', type=int, default=12345,
                       help='服务器端口')
    
    args = parser.parse_args()
    
    if args.mode == 'diagnostic':
        run_network_diagnostic()
    else:  # tcp-server
        run_real_server(args.port)

if __name__ == "__main__":
    main()
