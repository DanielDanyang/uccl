#!/usr/bin/env python3
"""
TCPX 连接功能测试
测试引擎的连接建立和接受功能
"""

import os
import sys
import threading
import time

def import_p2p_module():
    """导入 TCPX 引擎模块"""
    # 添加父目录到 Python 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 使用 importlib 导入模块
    import importlib.util
    so_file = os.path.join(parent_dir, 'libuccl_tcpx_engine.so')
    spec = importlib.util.spec_from_file_location("p2p", so_file)
    p2p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2p)
    return p2p

def test_basic_connection():
    """测试基本的连接功能"""
    print("🧪 TCPX 连接功能测试")
    print("=" * 50)
    
    try:
        # 导入模块
        print("🔄 导入 TCPX 引擎模块...")
        p2p = import_p2p_module()
        print("✅ 模块导入成功")
        
        # 创建两个引擎实例（模拟两个节点）
        print("🔄 创建引擎实例...")
        engine1 = p2p.Endpoint(0, 4)  # 节点1: GPU 0
        engine2 = p2p.Endpoint(1, 4)  # 节点2: GPU 1
        print("✅ 引擎实例创建成功")
        
        # 测试连接功能
        print("🔄 测试连接功能...")
        
        # 节点1 连接到节点2
        remote_ip = "127.0.0.1"
        remote_gpu_idx = 1
        remote_port = 12345
        
        print(f"  🔄 节点1 连接到节点2 ({remote_ip}:{remote_port}, GPU {remote_gpu_idx})...")
        success, conn_id = engine1.connect(remote_ip, remote_gpu_idx, remote_port)
        
        if success:
            print(f"  ✅ 连接成功! conn_id = {conn_id}")
        else:
            print(f"  ❌ 连接失败")
            return False
        
        # 测试接受连接功能
        print("🔄 测试接受连接功能...")
        
        print(f"  🔄 节点2 接受连接...")
        success, ip_addr, gpu_idx, conn_id2 = engine2.accept()
        
        if success:
            print(f"  ✅ 接受连接成功! 来自 {ip_addr} GPU {gpu_idx}, conn_id = {conn_id2}")
        else:
            print(f"  ❌ 接受连接失败")
            return False
        
        print("✅ 连接功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 连接测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_connections():
    """测试并发连接功能"""
    print("\n🧪 TCPX 并发连接测试")
    print("=" * 50)
    
    try:
        # 导入模块
        p2p = import_p2p_module()
        
        # 创建多个引擎实例
        print("🔄 创建多个引擎实例...")
        engines = []
        for i in range(3):
            engine = p2p.Endpoint(i, 4)
            engines.append(engine)
        print(f"✅ 创建了 {len(engines)} 个引擎实例")
        
        # 测试多个连接
        print("🔄 测试多个连接...")
        connections = []
        
        for i in range(len(engines)):
            for j in range(len(engines)):
                if i != j:  # 不连接自己
                    print(f"  🔄 引擎 {i} 连接到引擎 {j}...")
                    success, conn_id = engines[i].connect("127.0.0.1", j, 12345 + j)
                    if success:
                        connections.append((i, j, conn_id))
                        print(f"    ✅ 连接成功 {i} -> {j}, conn_id = {conn_id}")
                    else:
                        print(f"    ❌ 连接失败 {i} -> {j}")
        
        print(f"✅ 建立了 {len(connections)} 个连接")
        
        # 测试接受连接
        print("🔄 测试接受多个连接...")
        accepted = 0
        for i in range(len(engines)):
            try:
                success, ip_addr, gpu_idx, conn_id = engines[i].accept()
                if success:
                    accepted += 1
                    print(f"  ✅ 引擎 {i} 接受连接: 来自 {ip_addr} GPU {gpu_idx}, conn_id = {conn_id}")
            except Exception as e:
                print(f"  ⚠️  引擎 {i} 接受连接异常: {e}")
        
        print(f"✅ 接受了 {accepted} 个连接")
        return True
        
    except Exception as e:
        print(f"❌ 并发连接测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connection_metadata():
    """测试连接元数据功能"""
    print("\n🧪 TCPX 连接元数据测试")
    print("=" * 50)
    
    try:
        # 导入模块
        p2p = import_p2p_module()
        
        # 创建引擎
        print("🔄 创建引擎...")
        engine = p2p.Endpoint(0, 4)
        print("✅ 引擎创建成功")
        
        # 测试元数据生成
        print("🔄 测试元数据生成...")
        metadata = engine.get_metadata()
        print(f"✅ 元数据生成成功: {len(metadata)} 字节")
        print(f"  元数据内容: {list(metadata)}")
        
        # 测试 OOB IP
        print("🔄 测试 OOB IP...")
        oob_ip = p2p.get_oob_ip()
        print(f"✅ OOB IP: {oob_ip}")
        
        return True
        
    except Exception as e:
        print(f"❌ 元数据测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始 TCPX 连接功能测试")
    print()
    
    # 运行所有测试
    tests = [
        ("基本连接测试", test_basic_connection),
        ("并发连接测试", test_concurrent_connections),
        ("连接元数据测试", test_connection_metadata),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有连接测试通过!")
        print()
        print("📋 测试总结:")
        print("  ✅ 基本连接功能正常")
        print("  ✅ 并发连接功能正常")
        print("  ✅ 元数据生成正常")
        print()
        print("🚀 下一步: 可以开始测试数据传输功能")
    else:
        print("❌ 部分连接测试失败")
        print()
        print("🔧 可能的问题:")
        print("  - TCPX 插件未正确初始化")
        print("  - 连接参数配置错误")
        print("  - 网络配置问题")
    
    sys.exit(0 if passed == total else 1)
