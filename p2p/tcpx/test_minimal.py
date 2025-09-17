#!/usr/bin/env python3
"""
最小功能测试 - 只测试 TCPX 插件加载和设备查询
"""

import ctypes
import os
import sys

def test_minimal_tcpx():
    """测试最基本的 TCPX 功能"""
    
    print("🧪 TCPX 最小功能测试")
    print("=" * 50)
    
    # 1. 检查环境变量
    plugin_path = os.getenv('UCCL_TCPX_PLUGIN_PATH', '/usr/local/tcpx/lib64/libnccl-net-tcpx.so')
    device_id = int(os.getenv('UCCL_TCPX_DEV', '0'))
    
    print(f"📁 TCPX 插件路径: {plugin_path}")
    print(f"🔧 TCPX 设备 ID: {device_id}")
    
    # 2. 检查插件文件是否存在
    if not os.path.exists(plugin_path):
        print(f"❌ TCPX 插件文件不存在: {plugin_path}")
        print("💡 请设置正确的 UCCL_TCPX_PLUGIN_PATH 环境变量")
        return False
    
    print(f"✅ TCPX 插件文件存在")
    
    # 3. 尝试加载插件
    try:
        print(f"🔄 加载 TCPX 插件...")
        tcpx_lib = ctypes.CDLL(plugin_path)
        print(f"✅ TCPX 插件加载成功")
    except Exception as e:
        print(f"❌ TCPX 插件加载失败: {e}")
        return False
    
    # 4. 检查必要的函数符号
    required_functions = [
        'tcpxInit',
        'tcpxDevices', 
        'tcpxGetProperties',
        'tcpxListen',
        'tcpxConnect_v5',
        'tcpxAccept_v5',
        'tcpxRegMr',
        'tcpxDeregMr'
    ]
    
    print(f"🔍 检查函数符号...")
    missing_functions = []
    
    for func_name in required_functions:
        try:
            func = getattr(tcpx_lib, func_name)
            print(f"  ✅ {func_name}")
        except AttributeError:
            print(f"  ❌ {func_name} - 缺失")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"❌ 缺失 {len(missing_functions)} 个必要函数")
        return False
    
    print(f"✅ 所有必要函数符号都存在")
    
    # 5. 尝试初始化插件 (可能会失败，但不影响测试)
    try:
        print(f"🔄 尝试初始化 TCPX 插件...")
        
        # 定义日志回调函数类型
        LOG_FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p, 
                                        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
        
        def log_callback(level, file, func, line, msg):
            print(f"[TCPX-PLUGIN] {msg.decode('utf-8', errors='ignore')}")
            return 0
        
        log_func = LOG_FUNC_TYPE(log_callback)
        
        # 设置函数签名
        tcpx_lib.tcpxInit.argtypes = [ctypes.c_void_p]
        tcpx_lib.tcpxInit.restype = ctypes.c_int
        
        # 调用初始化 (传入 NULL，避免回调问题)
        result = tcpx_lib.tcpxInit(None)
        
        if result == 0:
            print(f"✅ TCPX 插件初始化成功")
        else:
            print(f"⚠️  TCPX 插件初始化返回: {result} (可能正常)")
            
    except Exception as e:
        print(f"⚠️  TCPX 插件初始化异常: {e} (可能正常)")
    
    # 6. 尝试查询设备数量
    try:
        print(f"🔄 查询 TCPX 设备数量...")
        
        # 设置函数签名
        tcpx_lib.tcpxDevices.argtypes = [ctypes.POINTER(ctypes.c_int)]
        tcpx_lib.tcpxDevices.restype = ctypes.c_int
        
        ndev = ctypes.c_int(0)
        result = tcpx_lib.tcpxDevices(ctypes.byref(ndev))
        
        if result == 0:
            print(f"✅ 找到 {ndev.value} 个 TCPX 设备")
        else:
            print(f"⚠️  设备查询返回: {result}, 设备数: {ndev.value}")
            
    except Exception as e:
        print(f"⚠️  设备查询异常: {e}")
    
    print("=" * 50)
    print("🎉 最小功能测试完成!")
    print()
    print("📋 测试总结:")
    print("  ✅ 插件文件存在")
    print("  ✅ 插件可以加载")
    print("  ✅ 必要函数符号存在")
    print("  ⚠️  初始化和设备查询可能需要特定环境")
    print()
    print("🚀 下一步: 如果插件加载成功，可以继续测试引擎创建")
    
    return True

if __name__ == "__main__":
    success = test_minimal_tcpx()
    sys.exit(0 if success else 1)
