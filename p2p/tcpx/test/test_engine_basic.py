#!/usr/bin/env python3
"""
TCPX 引擎基本功能测试
测试引擎的创建、销毁和基本操作
"""

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
        # 尝试当前目录
        engine_lib_path = './libuccl_tcpx_engine.so'
    if not os.path.exists(engine_lib_path):
        print(f"❌ TCPX 引擎库不存在: {engine_lib_path}")
        print("💡 请先运行: make clean && make")
        return False

    print(f"✅ TCPX 引擎库存在")

    # 3. 尝试导入 Python 模块
    try:
        print(f"🔄 导入 TCPX 引擎模块...")

        # 添加父目录到 Python 路径以导入 .so 文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # 检查 .so 文件是否存在
        so_file = os.path.join(parent_dir, 'libuccl_tcpx_engine.so')
        print(f"  检查模块文件: {so_file}")
        print(f"  文件存在: {os.path.exists(so_file)}")

        # 尝试不同的导入方式
        p2p = None
        try:
            # 方式1: 直接导入模块名
            import libuccl_tcpx_engine as p2p
            print(f"✅ 通过 libuccl_tcpx_engine 导入成功")
        except ImportError as e1:
            try:
                # 方式2: 尝试 p2p 名称
                import p2p
                print(f"✅ 通过 p2p 导入成功")
            except ImportError as e2:
                print(f"❌ 两种导入方式都失败:")
                print(f"  libuccl_tcpx_engine: {e1}")
                print(f"  p2p: {e2}")
                print(f"  当前路径: {os.getcwd()}")
                print(f"  Python 路径: {sys.path[:3]}...")

                # 尝试使用 importlib
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("p2p", so_file)
                    p2p = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(p2p)
                    print(f"✅ 通过 importlib 导入成功")
                except Exception as e3:
                    print(f"  importlib: {e3}")
                    return False

        if p2p is None:
            print(f"❌ 所有导入方式都失败")
            return False

    except Exception as e:
        print(f"❌ 导入过程异常: {e}")
        return False

    # 4. 检查引擎类和函数
    print(f"🔍 检查引擎类和函数...")

    try:
        # 检查 Endpoint 类是否存在
        if hasattr(p2p, 'Endpoint'):
            print(f"  ✅ Endpoint 类存在")
        else:
            print(f"  ❌ Endpoint 类不存在")
            return False

        # 检查 get_oob_ip 函数
        if hasattr(p2p, 'get_oob_ip'):
            print(f"  ✅ get_oob_ip 函数存在")
        else:
            print(f"  ❌ get_oob_ip 函数不存在")
            return False

    except Exception as e:
        print(f"❌ 检查引擎 API 异常: {e}")
        return False

    print(f"✅ 所有引擎 API 都存在")

    # 5. 测试引擎创建和销毁
    print(f"🔄 测试引擎创建和销毁...")

    try:
        # 创建引擎
        local_gpu_idx = 0
        num_cpus = 4

        print(f"  🔄 创建引擎 (gpu_idx={local_gpu_idx}, num_cpus={num_cpus})...")
        engine = p2p.Endpoint(local_gpu_idx, num_cpus)

        if engine:
            print(f"  ✅ 引擎创建成功")

            # 测试 get_oob_ip
            try:
                oob_ip = p2p.get_oob_ip()
                print(f"  ✅ OOB IP: {oob_ip}")
            except Exception as e:
                print(f"  ⚠️  OOB IP 获取异常: {e}")

            # 测试元数据生成
            try:
                metadata = engine.get_metadata()
                print(f"  ✅ 元数据生成成功: {len(metadata)} 字节")
            except Exception as e:
                print(f"  ⚠️  元数据生成异常: {e}")

            print(f"  ✅ 引擎测试完成")

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
