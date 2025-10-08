## 项目目标与环境
- 方向调整（2025-10-08）：暂时弃用单进程 Orchestrator 路线，转为在多进程基线基础上推进；每 GPU 打开 4 条 TCPX 连接（1 channel≈1 TCPX 连接）。以单 200Gbps NIC + ~8 连接为 per-NIC 上限参考，GPU 固定到 NUMA-local NIC（现阶段可静态映射）。

- 目标：在 GCP A3-high（2 节点、每节点 8×H100、4×gVNIC）上实现基于 TCPX 的 GPU-P2P 高效传输，作为 NIXL 插件数据面，兼容 nccl-plugin-gpudirecttcpx 接口
- 约束：仅 TCPX（无 RDMA）；单进程/多进程两套路径；网络配置脚本化（scripts/node_ips/tcpx.txt）

## 时间线与里程碑（含多进程）
1) 多网卡与通道管理（完成）
- 建立 GPU→NIC 映射：GPU0–3 使用 eth1/eth2；GPU4–7 使用 eth3/eth4
- 修复“各 GPU 通道数一致”的假设，按实际通道工作；引入 ChannelManager 管理 per-GPU/per-channel 资源

2) 多进程基准（test_tcpx_perf_multi，完成且作为对照）
- 采用多进程/每 GPU 独立进程的 NCCL/TCPX 测试路径
- 核心策略：异步 post + 持续 progress（process_completed_chunk 非阻塞轮询；wait_for_channel_capacity 阻塞释放）
- 结果：端到端链路能稳定跑通，作为后续单进程 Orchestrator 对齐参考

3) 控制面重构与单进程 Orchestrator（完成）
- 单进程统一调度 8×GPU，避免 devmem 冲突；握手、端点建立、mhandle/comm 注册完成
- 增强健壮性：通道计数校验、除零保护、日志修复

4) 数据面升级与滑动窗口（进行中）
- 从同步阻塞切到异步+滑动窗口，遵守 TCPX 每 comm 最多 16 个 in-flight 请求
- 引入 SlidingWindow 控制窗口大小；对齐多进程基准的进度驱动策略

## 关键问题、根因与修复
1) 带宽与缓冲区问题（早期）
- 带宽统计 8×膨胀：改为每 GPU 独立统计
- 固定 64MB 缓冲区有溢出风险：改为可配置并校验 test_size

2) 滑动窗口初版使用错误
- 服务器端在 recv 完成前调用 tcpx_irecv_consumed，导致后台不工作
- 修复：先 tcpx_test() 确认 done，再调用 consumed（recv 路径）；并将接口改为三态 try_release_oldest()
  - 返回值：0=释放成功；1=未就绪（非错误，稍后重试）；-1=真错误

3) TCPX 队列顺序约束误判
- tcpx_test 仅能作用于 TCPX 内部队列的 rq.next_transmitting()
- 修复：SlidingWindow 严格按 FIFO 仅检查队首；遇到 rc!=0/或 done=0 视为“未就绪”而非致命

4) 关键进度驱动缺失（根因类，单进程特有）
- 现象：首批 16 个请求 post 满后卡住；tcpx_test 持续 rc=0, done=0，窗口无法释放
- 原因：仅在窗口满时轮询，且只盯队首；tcpxCommProgress 缺少持续驱动
- 修复（对齐多进程基准的“血泪经验”）：
  - 每次 tcpx_irecv/tcpx_isend 后立即对当前通道做非阻塞 drain（progress_channel(blocking=false)）
  - 顺手轮询其它通道（opportunistic drain）
  - 窗口满时改为阻塞 drain，直到释放出至少一个槽位（progress_channel(blocking=true)）
  - 增加详细 DEBUG：打印 tcpx_test 的 rc/done/size（收发两端），必要时启用 TCPX TRACE 验证 tcpxCommProgress

## 与多进程基准的对齐点
- 进度模型：非阻塞轮询 + 阻塞释放 完全等价于 process_completed_chunk + wait_for_channel_capacity
- 队列策略：FIFO 只测队首，避免越权访问
- 轮询范围：当前通道优先，辅以“顺手轮询”所有通道，保证系统整体前进

## 当前状态
- 单进程 Orchestrator 已引入 progress_channel，完成“持续进度驱动”改造；编译通过
- 预计能解决 Iteration 0 卡死问题（首批窗口堆满）；待在双节点实测确认
- 已提供 DEBUG 指南与 TRACE 开关，便于现场定位

## 下一步计划
- 以最小配置回归：UCCL_TCPX_NUM_CHANNELS=1；确认 Iteration 0 正常完成
- 如仍异常：开启 TRACE（NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE，NCCL_DEBUG=INFO，NCCL_DEBUG_SUBSYS=NET,INIT）
- 逐步恢复多通道/全量负载，记录带宽基线；必要时补齐 unpack 元数据处理（当前 memcpy 模式无需 kernel）

## 风险与依赖
- 单进程模式对轮询频度敏感：sleep 策略、CPU 占用可能影响收敛
- TCPX 内部队列推进受系统负载影响；需要保持足够的 progress 调用密度
- 若启用 unpack kernel：需严格遵守 ready_flag/threshold→kernel→consumed 的顺序

## 相关产物
## 外部联系与问题清单（已准备）
- 我们已整理并发送给 Google TCPX 团队的问询邮件草案（中英文），涵盖：
  - channel→socket 映射与 MAX_SOCKETS 关系、NUMA 建议
  - 进度机制（tcpx_test/CommProgress）调用节奏与语义
  - recv 生命周期与 consumed 时机（含 unpack 元数据路径）
  - 多 NIC 调度/绑定与 full-mesh 并发的最佳实践
  - 参数调优（窗口/chunk/睡眠/环境变量）与诊断（TRACE 指标）
- 待回信后按建议进一步对齐实现与参数。

- 多进程基准：tests/test_tcpx_perf_multi.cc（对照参考）
- 单进程 Orchestrator：tests/test_tcpx_perf_orchestrator.cc（已完成进度驱动改造）
- 滑动窗口：include/sliding_window.h，src/sliding_window.cc（try_release_oldest 三态语义）
- 调试文档：DEBUG_GUIDE.md，DEBUG_TCPX_TEST_ANALYSIS.md，STEP3_* 系列修复说明

