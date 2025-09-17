#!/usr/bin/env python3
"""重用标准引擎写入单元测试的 TCPX 冒烟测试。"""

from __future__ import annotations

import os
import sys

try:
    from p2p.tests.test_engine_write import test_local
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"导入 test_local 失败: {exc}\n")
    raise


def main() -> int:
    plugin_path = os.environ.get("UCCL_TCPX_PLUGIN_PATH")
    dev_idx = os.environ.get("UCCL_TCPX_DEV")

    if not plugin_path:
        print("[警告] UCCL_TCPX_PLUGIN_PATH 未设置；将应用默认加载器规则")
    else:
        print(f"使用 TCPX 插件: {plugin_path}")
    if dev_idx:
        print(f"使用 TCPX 设备: {dev_idx}")

    os.environ.setdefault("UCCL_RCMODE", "1")

    try:
        test_local()
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
