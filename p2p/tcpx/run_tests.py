#!/usr/bin/env python3
"""
TCPX 引擎测试运行器
按顺序运行所有测试，提供清晰的结果报告
"""

import os
import sys
import subprocess
import time

def run_test(test_name, test_path, description):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"📝 {description}")
    print(f"📁 {test_path}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=True, text=True, timeout=30)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} - 成功 ({elapsed:.1f}s)")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_name} - 失败 ({elapsed:.1f}s)")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} - 超时 (30s)")
        return False
    except Exception as e:
        print(f"💥 {test_name} - 异常: {e}")
        return False

def main():
    """主测试流程"""
    
    print("🚀 TCPX 引擎测试套件")
    print("=" * 60)
    
    # 检查当前目录
    if not os.path.exists('Makefile'):
        print("❌ 请在 p2p/tcpx 目录下运行此脚本")
        sys.exit(1)
    
    # 测试列表
    tests = [
        {
            'name': '插件加载测试',
            'path': 'test/test_minimal.py',
            'description': '测试 NCCL TCPX 插件是否可以加载和访问'
        },
        {
            'name': 'NCCL 接口测试', 
            'path': 'test/test_nccl_plugin.py',
            'description': '测试 NCCL 插件结构体和函数指针'
        }
    ]
    
    # 检查是否需要编译
    engine_lib = 'libuccl_tcpx_engine.so'
    if os.path.exists(engine_lib):
        tests.append({
            'name': '引擎功能测试',
            'path': 'test/test_engine_basic.py', 
            'description': '测试引擎创建、销毁和基本操作'
        })
    else:
        print(f"⚠️  引擎库 {engine_lib} 不存在，跳过引擎测试")
        print(f"💡 运行 'make clean && make' 编译后再测试")
    
    # 运行测试
    results = []
    for test in tests:
        if os.path.exists(test['path']):
            success = run_test(test['name'], test['path'], test['description'])
            results.append((test['name'], success))
        else:
            print(f"⚠️  测试文件不存在: {test['path']}")
            results.append((test['name'], False))
    
    # 总结报告
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("\n🚀 下一步建议:")
        if not os.path.exists(engine_lib):
            print("  1. 编译引擎: make clean && make")
            print("  2. 运行引擎测试: python test/test_engine_basic.py")
        else:
            print("  1. 开始实现真实的 TCPX 插件集成")
            print("  2. 查看 docs/CURRENT_STATUS.md 了解下一步计划")
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        print("\n🔧 可能的问题:")
        print("  - TCPX 插件路径不正确")
        print("  - 缺少必要的依赖库")
        print("  - 环境变量未设置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
