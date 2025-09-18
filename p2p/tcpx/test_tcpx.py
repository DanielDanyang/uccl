#!/usr/bin/env python3
"""
TCPX 引擎基本功能测试 - 对应 p2p/tests/test_engine_metadata.py
"""

import sys
import os

print("🧪 TCPX 引擎基本功能测试")
print("==================================================")

# 尝试导入 TCPX 模块
try:
    import p2p
    print("✅ 成功导入 p2p 模块")
except ImportError as e:
    print(f"❌ 导入 p2p 模块失败: {e}")
    sys.exit(1)

# 测试全局函数
try:
    oob_ip = p2p.get_oob_ip()
    print(f"✅ OOB IP: {oob_ip}")
except Exception as e:
    print(f"❌ 获取 OOB IP 失败: {e}")

# 创建 TCPX 引擎
try:
    print("🔄 创建 TCPX 引擎...")
    engine = p2p.Endpoint(0, 4)  # GPU 0, 4 CPUs
    print("✅ 引擎创建成功")
except Exception as e:
    print(f"❌ 引擎创建失败: {e}")
    sys.exit(1)

# 测试设备数量
try:
    print("🔄 获取设备数量...")
    device_count = engine.get_device_count()
    print(f"✅ 设备数量: {device_count}")
    
    if device_count > 0:
        print("✅ 发现真实的 TCPX 设备")
    else:
        print("❌ 没有发现有效的 TCPX 设备")
except Exception as e:
    print(f"❌ 获取设备数量失败: {e}")

# 测试元数据生成
try:
    print("🔄 生成元数据...")
    metadata = engine.get_metadata()
    print(f"✅ 元数据生成成功: {len(metadata)} 字节")
except Exception as e:
    print(f"❌ 元数据生成失败: {e}")

print("==================================================")
print("🎉 TCPX 引擎基本功能测试完成!")
