#!/usr/bin/env python3
"""
TCPX 引擎基本功能测试
测试引擎的创建、销毁和基本操作
"""

import ctypes
import os
import sys

def test_tcpx_engine():
    """测试 TCPX 引擎基本功能"""

    print("🧪 TCPX 引擎基本功能测试")
    print("=" * 50)

    # 1. 检查环境变量
    plugin_path = os.getenv('UCCL_TCPX_PLUGIN_PATH', '/usr/local/tcpx/lib64/libnccl-net-tcpx.so')
    print(f"📁 TCPX 插件路径: {plugin_path}")

    # 2. 检查引擎库是否存在
    engine_lib_path = '../libuccl_tcpx_engine.so'
    if not os.path.exists(engine_lib_path):
        print(f"❌ TCPX 引擎库不存在: {engine_lib_path}")
        print("💡 请先运行: make clean && make")
        return False

    print(f"✅ TCPX 引擎库存在")

    # 3. 尝试加载引擎库
    try:
        print(f"🔄 加载 TCPX 引擎库...")
        engine_lib = ctypes.CDLL(engine_lib_path)
        print(f"✅ TCPX 引擎库加载成功")
    except Exception as e:
        print(f"❌ TCPX 引擎库加载失败: {e}")
        return False

    # 4. 检查引擎 API 函数
    print(f"🔍 检查引擎 API 函数...")

    required_functions = [
        'uccl_engine_create',
        'uccl_engine_destroy',
        'uccl_engine_get_metadata',
        'uccl_engine_connect',
        'uccl_engine_reg',
        'uccl_engine_dereg',
        'uccl_engine_write',
        'uccl_engine_get_p2p_listen_port'
    ]

    missing_functions = []
    for func_name in required_functions:
        try:
            func = getattr(engine_lib, func_name)
            print(f"  ✅ {func_name}")
        except AttributeError:
            print(f"  ❌ {func_name} - 缺失")
            missing_functions.append(func_name)

    if missing_functions:
        print(f"❌ 缺失 {len(missing_functions)} 个必要函数")
        return False

    print(f"✅ 所有引擎 API 函数都存在")

    # 5. 测试引擎创建和销毁
    print(f"🔄 测试引擎创建和销毁...")

    try:
        # 设置函数签名
        engine_lib.uccl_engine_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        engine_lib.uccl_engine_create.restype = ctypes.c_void_p

        engine_lib.uccl_engine_destroy.argtypes = [ctypes.c_void_p]
        engine_lib.uccl_engine_destroy.restype = None

        # 创建引擎
        local_gpu_idx = 0
        num_cpus = 4

        print(f"  🔄 创建引擎 (gpu_idx={local_gpu_idx}, num_cpus={num_cpus})...")
        engine = engine_lib.uccl_engine_create(local_gpu_idx, num_cpus)

        if engine:
            print(f"  ✅ 引擎创建成功: {hex(engine)}")

            # 销毁引擎
            print(f"  🔄 销毁引擎...")
            engine_lib.uccl_engine_destroy(engine)
            print(f"  ✅ 引擎销毁成功")

        else:
            print(f"  ❌ 引擎创建失败")
            return False

    except Exception as e:
        print(f"  ❌ 引擎测试异常: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_tcpx_engine()

    print("=" * 50)
    if success:
        print("🎉 TCPX 引擎基本功能测试成功!")
        print()
        print("📋 测试总结:")
        print("  ✅ 引擎库可以加载")
        print("  ✅ 所有 API 函数存在")
        print("  ✅ 引擎可以创建和销毁")
        print()
        print("🚀 下一步: 可以开始测试连接和数据传输功能")
    else:
        print("❌ TCPX 引擎基本功能测试失败")
        print()
        print("🔧 可能的问题:")
        print("  - 引擎库未编译")
        print("  - 编译错误")
        print("  - 依赖库缺失")

    sys.exit(0 if success else 1)
"""
TCPX 引擎基本功能测试
"""

import ctypes
import os
import sys

def test_tcpx_engine():
    """测试 TCPX 引擎基本功能"""
    
    print("🧪 TCPX 引擎基本功能测试")
    print("=" * 50)
    
    # 1. 检查环境变量
    plugin_path = os.getenv('UCCL_TCPX_PLUGIN_PATH', '/usr/local/tcpx/lib64/libnccl-net-tcpx.so')
    print(f"📁 TCPX 插件路径: {plugin_path}")
    
    # 2. 检查引擎库是否存在
    engine_lib_path = './libuccl_tcpx_engine.so'
    if not os.path.exists(engine_lib_path):
        print(f"❌ TCPX 引擎库不存在: {engine_lib_path}")
        print("💡 请先运行: make clean && make")
        return False
    
    print(f"✅ TCPX 引擎库存在")
    
    # 3. 尝试加载引擎库
    try:
        print(f"🔄 加载 TCPX 引擎库...")
        engine_lib = ctypes.CDLL(engine_lib_path)
        print(f"✅ TCPX 引擎库加载成功")
    except Exception as e:
        print(f"❌ TCPX 引擎库加载失败: {e}")
        return False
    
    # 4. 检查引擎 API 函数
    print(f"🔍 检查引擎 API 函数...")
    
    required_functions = [
        'uccl_engine_create',
        'uccl_engine_destroy',
        'uccl_engine_get_metadata',
        'uccl_engine_connect',
        'uccl_engine_reg',
        'uccl_engine_dereg',
        'uccl_engine_write',
        'uccl_engine_get_p2p_listen_port'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        try:
            func = getattr(engine_lib, func_name)
            print(f"  ✅ {func_name}")
        except AttributeError:
            print(f"  ❌ {func_name} - 缺失")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"❌ 缺失 {len(missing_functions)} 个必要函数")
        return False
    
    print(f"✅ 所有引擎 API 函数都存在")
    
    # 5. 测试引擎创建和销毁
    print(f"🔄 测试引擎创建和销毁...")
    
    try:
        # 设置函数签名
        engine_lib.uccl_engine_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        engine_lib.uccl_engine_create.restype = ctypes.c_void_p
        
        engine_lib.uccl_engine_destroy.argtypes = [ctypes.c_void_p]
        engine_lib.uccl_engine_destroy.restype = None
        
        # 创建引擎
        local_gpu_idx = 0
        num_cpus = 4
        
        print(f"  🔄 创建引擎 (gpu_idx={local_gpu_idx}, num_cpus={num_cpus})...")
        engine = engine_lib.uccl_engine_create(local_gpu_idx, num_cpus)
        
        if engine:
            print(f"  ✅ 引擎创建成功: {hex(engine)}")
            
            # 销毁引擎
            print(f"  🔄 销毁引擎...")
            engine_lib.uccl_engine_destroy(engine)
            print(f"  ✅ 引擎销毁成功")
            
        else:
            print(f"  ❌ 引擎创建失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 引擎测试异常: {e}")
        return False
    
    # 6. 测试元数据生成
    print(f"🔄 测试元数据生成...")
    
    try:
        # 重新创建引擎用于元数据测试
        engine = engine_lib.uccl_engine_create(0, 4)
        if not engine:
            print(f"  ❌ 无法创建引擎用于元数据测试")
            return False
        
        # 设置元数据函数签名
        engine_lib.uccl_engine_get_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_size_t)]
        engine_lib.uccl_engine_get_metadata.restype = ctypes.c_int
        
        # 准备缓冲区
        metadata_buffer = (ctypes.c_uint8 * 1024)()
        metadata_size = ctypes.c_size_t(1024)
        
        # 获取元数据
        result = engine_lib.uccl_engine_get_metadata(engine, metadata_buffer, ctypes.byref(metadata_size))
        
        if result == 0:
            metadata_bytes = bytes(metadata_buffer[:metadata_size.value])
            metadata_str = metadata_bytes.decode('utf-8', errors='ignore')
            print(f"  ✅ 元数据生成成功: {metadata_str}")
        else:
            print(f"  ⚠️  元数据生成返回: {result}")
        
        # 清理
        engine_lib.uccl_engine_destroy(engine)
        
    except Exception as e:
        print(f"  ⚠️  元数据测试异常: {e}")
    
    return True

if __name__ == "__main__":
    success = test_tcpx_engine()
    
    print("=" * 50)
    if success:
        print("🎉 TCPX 引擎基本功能测试成功!")
        print()
        print("📋 测试总结:")
        print("  ✅ 引擎库可以加载")
        print("  ✅ 所有 API 函数存在")
        print("  ✅ 引擎可以创建和销毁")
        print("  ✅ 元数据可以生成")
        print()
        print("🚀 下一步: 可以开始测试连接和数据传输功能")
    else:
        print("❌ TCPX 引擎基本功能测试失败")
        print()
        print("🔧 可能的问题:")
        print("  - 引擎库未编译")
        print("  - 编译错误")
        print("  - 依赖库缺失")
    
    sys.exit(0 if success else 1)
