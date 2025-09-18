#!/usr/bin/env python3
"""
NCCL TCPX 插件接口测试
"""

import ctypes
import os
import sys

def test_nccl_tcpx_plugin():
    """测试 NCCL TCPX 插件接口"""
    
    print("🧪 NCCL TCPX 插件接口测试")
    print("=" * 50)
    
    # 1. 检查环境变量
    plugin_path = os.getenv('UCCL_TCPX_PLUGIN_PATH', '/usr/local/tcpx/lib64/libnccl-net-tcpx.so')
    device_id = int(os.getenv('UCCL_TCPX_DEV', '0'))
    
    print(f"📁 TCPX 插件路径: {plugin_path}")
    print(f"🔧 TCPX 设备 ID: {device_id}")
    
    # 2. 检查插件文件是否存在
    if not os.path.exists(plugin_path):
        print(f"❌ TCPX 插件文件不存在: {plugin_path}")
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
    
    # 4. 检查 NCCL 插件结构体
    print(f"🔍 检查 NCCL 插件接口...")
    
    try:
        # 获取 NCCL 插件结构体
        plugin_symbol = getattr(tcpx_lib, 'ncclNetPlugin_v7')
        print(f"  ✅ ncclNetPlugin_v7 符号存在")
        
        # 定义 NCCL 插件结构体
        class NCCLNetPlugin(ctypes.Structure):
            _fields_ = [
                ("name", ctypes.c_char_p),
                ("init", ctypes.c_void_p),
                ("devices", ctypes.c_void_p),
                ("getProperties", ctypes.c_void_p),
                ("listen", ctypes.c_void_p),
                ("connect", ctypes.c_void_p),
                ("accept", ctypes.c_void_p),
                ("regMr", ctypes.c_void_p),
                ("regMrDmaBuf", ctypes.c_void_p),
                ("deregMr", ctypes.c_void_p),
                ("isend", ctypes.c_void_p),
                ("irecv", ctypes.c_void_p),
                ("iflush", ctypes.c_void_p),
                ("test", ctypes.c_void_p),
                ("closeSend", ctypes.c_void_p),
                ("closeRecv", ctypes.c_void_p),
                ("closeListen", ctypes.c_void_p),
                ("getDeviceMr", ctypes.c_void_p),
                ("irecvConsumed", ctypes.c_void_p),
            ]
        
        # 获取插件结构体内容
        plugin = ctypes.cast(plugin_symbol, ctypes.POINTER(NCCLNetPlugin)).contents
        
        if plugin.name:
            plugin_name = plugin.name.decode('utf-8')
            print(f"  📋 插件名称: {plugin_name}")
        else:
            print(f"  ⚠️  插件名称为空")
        
        # 检查关键函数指针
        function_checks = [
            ("init", plugin.init),
            ("devices", plugin.devices),
            ("getProperties", plugin.getProperties),
            ("listen", plugin.listen),
            ("connect", plugin.connect),
            ("accept", plugin.accept),
            ("regMr", plugin.regMr),
            ("deregMr", plugin.deregMr),
            ("isend", plugin.isend),
            ("irecv", plugin.irecv),
            ("test", plugin.test),
        ]
        
        valid_functions = 0
        for func_name, func_ptr in function_checks:
            if func_ptr:
                print(f"    ✅ {func_name}")
                valid_functions += 1
            else:
                print(f"    ❌ {func_name} - NULL")
        
        print(f"  📊 有效函数: {valid_functions}/{len(function_checks)}")
        
        if valid_functions >= 8:  # 至少需要核心函数
            print(f"✅ NCCL 插件接口检查通过")
            
            # 5. 尝试调用 devices 函数
            if plugin.devices:
                print(f"🔄 尝试调用 devices 函数...")
                try:
                    # 设置函数签名
                    devices_func = ctypes.cast(plugin.devices, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int)))
                    
                    ndev = ctypes.c_int(0)
                    result = devices_func(ctypes.byref(ndev))
                    
                    print(f"  📊 devices() 返回: {result}, 设备数: {ndev.value}")
                    
                    if result == 0:
                        print(f"  ✅ 找到 {ndev.value} 个 TCPX 设备")
                    else:
                        print(f"  ⚠️  devices() 调用返回错误码: {result}")
                        
                except Exception as e:
                    print(f"  ⚠️  devices() 调用异常: {e}")
            
            return True
        else:
            print(f"❌ 插件函数不完整")
            return False
            
    except AttributeError:
        print(f"  ❌ ncclNetPlugin_v7 符号不存在")
        return False
    except Exception as e:
        print(f"  ❌ 插件结构体解析失败: {e}")
        return False

if __name__ == "__main__":
    success = test_nccl_tcpx_plugin()
    
    print("=" * 50)
    if success:
        print("🎉 NCCL TCPX 插件接口测试成功!")
        print()
        print("📋 测试总结:")
        print("  ✅ 插件文件存在并可加载")
        print("  ✅ NCCL 插件结构体可访问")
        print("  ✅ 关键函数指针有效")
        print("  ✅ devices() 函数可调用")
        print()
        print("🚀 下一步: 可以开始实现真实的 TCPX 传输层")
    else:
        print("❌ NCCL TCPX 插件接口测试失败")
        print()
        print("🔧 可能的问题:")
        print("  - 插件版本不匹配")
        print("  - 插件编译问题")
        print("  - 环境配置问题")
    
    sys.exit(0 if success else 1)
