# TCPX NIXL 测试手册

本文档记录了在云 GPU 上验证基于 TCPX 的 UCCL 引擎所需的最低要求。在具有 NCCL GPUDirectTCPX 插件访问权限的单个主机上运行以下步骤（多 GPU 或多节点设置遵循相同的大纲，只需在对等节点之间共享元数据）。

## 1. 前置条件
- 安装了 PyTorch 的 CUDA 可见 H100（或兼容）GPU。
- 从 [google/nccl-plugin-gpudirecttcpx](https://github.com/google/nccl-plugin-gpudirecttcpx) 生成的 NCCL GPUDirectTCPX 插件（`libnccl-net.so`）。
- 检出此仓库并确保 `python -c "from uccl import p2p"` 成功执行。

## 2. 环境准备
```bash
export UCCL_TCPX_PLUGIN_PATH=/abs/path/to/libnccl-net.so   # 必需
export UCCL_TCPX_DEV=0                                     # 选择 TCPX 网卡索引
export UCCL_RCMODE=1                                       # 在测试中启用单侧操作
```
确认插件路径存在（`ls $UCCL_TCPX_PLUGIN_PATH`）。

## 3. 构建（如需要）
大多数工作流程只需要通过 `python -m build` 生成的 wheel 包或开发期间使用的可编辑安装。如果您正在迭代 C++ 代码：
```bash
cd /path/to/uccl/p2p
make  # 镜像 build.sh 的可选辅助工具
```

## 4. 冒烟测试
运行重用 `p2p/tests/test_engine_write.py` 的 TCPX 冒烟测试：
```bash
python -m p2p.tcpx.test_tcpx_write
```
预期结果：
- 端点创建成功。
- 元数据往返产生 IPv4/IPv6 地址、端口和 GPU 索引。
- 写入测试在退出 0 之前打印 `Local RDMA-WRITE test passed`。

## 5. 故障排除速查表
- **导入失败：** 重新构建/安装 UCCL（`python -m build` 或 `pip install -e .`）。
- **插件未找到：** 重新检查 `UCCL_TCPX_PLUGIN_PATH` 和文件系统权限。
- **无 TCPX 设备：** 调整 `UCCL_TCPX_DEV` 以匹配枚举的网卡索引。
- **CUDA 错误：** 确认 `nvidia-smi` 报告 GPU 且 PyTorch 能够识别它（`python -c "import torch; print(torch.cuda.is_available())"`）。

## 6. 扩展覆盖范围
一旦冒烟测试通过，在两个主机上运行更丰富的 NIXL 基准测试（参见 `p2p/benchmarks/benchmark_nixl.py`），同时保持导出 TCPX 环境变量。
