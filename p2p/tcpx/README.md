# TCPX NIXL 插件

此目录包含实验性的胶水代码，用于在 NCCL GPUDirectTCPX 传输层之上运行 UCCL 的点对点引擎。该实现镜像了 RDMA 引擎（`p2p/uccl_engine.cc`），但将传输调用替换为 TCPX 插件入口点。

## 目录结构
- `uccl_engine_tcpx_nixl.cc` - 轻量级的 TCPX 支持引擎，目前覆盖写入路径。
- `test_tcpx_write.py` - 标准引擎写入测试的简单包装器，包含 TCPX 特定的环境检查。
- `TESTING.md` - 云主机的端到端验证步骤。
- `Makefile` - 本地迭代时构建共享对象的可选辅助工具。

## 前置条件
- 目标主机上可用的 CUDA + PyTorch（冒烟测试使用 `torch.cuda`）。
- 构建为共享对象的 NCCL GPUDirectTCPX 插件（`libnccl-net.so` 或自定义路径）。
- 在加载 `uccl` 之前导出环境变量 `UCCL_TCPX_PLUGIN_PATH` 和 `UCCL_TCPX_DEV`。

## 快速开始
1. 构建/安装 UCCL，确保 `python -c "from uccl import p2p"` 成功执行。
2. 设置 TCPX 环境变量，例如：
   ```bash
   export UCCL_TCPX_PLUGIN_PATH=/opt/tcpx/libnccl-net.so
   export UCCL_TCPX_DEV=0
   ```
3. 运行 TCPX 冒烟测试：
   ```bash
   python -m p2p.tcpx.test_tcpx_write
   ```
4. 检查输出中的 "Local RDMA-WRITE test passed" 消息。

详细的检查清单（多节点验证、故障排除）请参见 `TESTING.md`。

## 后续步骤
- 一旦写入路径稳定，扩展 `uccl_engine_tcpx_nixl.cc` 以支持 accept/recv/read 路径。
- 将引擎集成到 `p2p/benchmarks` 中的更高级别 NIXL 集成测试中。
