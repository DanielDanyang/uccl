# TCPX P2P 文档索引

**最后更新**: 2025-10-02  
**状态**: ✅ 滑动窗口已修复，功能正常，性能待优化

---

## 📖 快速导航

### 🚀 新手入门

1. **[TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md)** ⭐⭐⭐
   - 完整的代码注释和解释
   - 1100+ 行详细中文注释
   - 理解整个测试程序的最佳起点

2. **[CURRENT_SETUP.md](CURRENT_SETUP.md)** ⭐⭐
   - 当前环境配置
   - IP 地址：10.65.74.150 (Server), 10.64.113.77 (Client)
   - 网络接口：eth1,eth2,eth3,eth4

### 🔧 技术细节

3. **[SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md)** ⭐⭐⭐
   - 滑动窗口机制的可视化解释
   - 为什么需要滑动窗口
   - 如何避免 TCPX 请求池耗尽

4. **[SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md)** ⭐⭐⭐
   - **最新修复**：Server 端只处理 17 个 chunks 的问题
   - 根本原因：滑动窗口检查在 `tcpx_irecv` 之后
   - 修复方案：将检查移到 `tcpx_irecv` 之前
   - **状态**: ✅ 已修复，所有 128 个 chunks 都能处理

5. **[TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md)** ⭐⭐
   - TCPX API 与 NCCL 插件的映射关系
   - 理解 TCPX 内部实现的关键

### 🐛 问题诊断

6. **[SERVER_17_CHUNKS_BUG.md](SERVER_17_CHUNKS_BUG.md)** ⭐⭐
   - 详细的 bug 分析
   - 为什么 Server 端只处理 17 个 chunks
   - 已通过 SLIDING_WINDOW_FIX_FINAL.md 修复

7. **[COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md)** ⭐⭐
   - 常见错误和解决方案
   - 避免踩坑的最佳实践

8. **[LOG_ANALYSIS_SUMMARY.md](LOG_ANALYSIS_SUMMARY.md)** ⭐
   - 日志分析总结
   - 如何从日志中诊断问题

### 📊 性能优化

9. **[PERF_DIARY.md](PERF_DIARY.md)** ⭐⭐
   - 性能优化历程
   - Kernel 模式从 100× 慢到正常的修复过程

### 🗂️ 历史文档（已过时）

10. **[BUG_ANALYSIS_20251002.md](BUG_ANALYSIS_20251002.md)** ⚠️ 已过时
    - 早期的 bug 分析
    - 已被 SLIDING_WINDOW_FIX_FINAL.md 取代

11. **[DEBUG_PLAN_20251002.md](DEBUG_PLAN_20251002.md)** ⚠️ 已过时
    - 调试计划
    - 问题已解决

12. **[TIMEOUT_FIX_20251002.md](TIMEOUT_FIX_20251002.md)** ⚠️ 已过时
    - 超时问题的修复
    - 后来发现不是超时问题，而是滑动窗口问题

---

## 🎯 当前状态总结

### ✅ 已解决的问题

1. **Kernel 性能 100× 慢** → 修复：将 stream/launcher 创建移到循环外
2. **Server 端只处理 17 个 chunks** → 修复：将滑动窗口检查移到 `tcpx_irecv` 之前
3. **超时导致提前退出** → 修复：移除 10 秒超时限制

### ⚠️ 当前问题

**性能慢**：
- Server 端：100.155 ms/iter, 0.62 GB/s
- Client 端：156.767 ms/iter, 0.40 GB/s
- **预期**：~20-30 ms/iter, ~20 GB/s（四网卡聚合）

**原因分析**：
- ✅ 滑动窗口工作正常（所有 128 个 chunks 都被处理）
- ❓ 性能瓶颈在哪里？需要进一步分析

---

## 📝 阅读顺序建议

### 对于新手

1. **TEST_TCPX_PERF_EXPLAINED.md** - 理解代码
2. **SLIDING_WINDOW_VISUAL.md** - 理解滑动窗口
3. **CURRENT_SETUP.md** - 了解环境配置
4. **COMMON_MISTAKES_AND_FIXES.md** - 避免常见错误

### 对于调试问题

1. **SLIDING_WINDOW_FIX_FINAL.md** - 最新修复
2. **SERVER_17_CHUNKS_BUG.md** - 问题分析
3. **LOG_ANALYSIS_SUMMARY.md** - 日志分析方法

### 对于性能优化

1. **PERF_DIARY.md** - 性能优化历程
2. **TEST_TCPX_PERF_EXPLAINED.md** - 代码细节
3. **TCPX_LOGIC_MAPPING.md** - TCPX 内部实现

---

## 🗑️ 可以删除的文档

以下文档已过时，可以删除：

- `BUG_ANALYSIS_20251002.md` - 已被 SLIDING_WINDOW_FIX_FINAL.md 取代
- `DEBUG_PLAN_20251002.md` - 问题已解决
- `TIMEOUT_FIX_20251002.md` - 不是真正的问题

---

## 📚 核心文档（保留）

### 必读文档

1. **TEST_TCPX_PERF_EXPLAINED.md** - 代码注释
2. **SLIDING_WINDOW_VISUAL.md** - 滑动窗口机制
3. **SLIDING_WINDOW_FIX_FINAL.md** - 最新修复
4. **CURRENT_SETUP.md** - 环境配置

### 参考文档

5. **TCPX_LOGIC_MAPPING.md** - TCPX API 映射
6. **COMMON_MISTAKES_AND_FIXES.md** - 常见错误
7. **PERF_DIARY.md** - 性能优化历程
8. **SERVER_17_CHUNKS_BUG.md** - Bug 分析
9. **LOG_ANALYSIS_SUMMARY.md** - 日志分析

### 其他文档

10. **FINAL_STRUCTURE.md** - 项目结构
11. **tcpx_transfer.md** - TCPX 传输流程

---

**总计**: 11 个核心文档 + 3 个过时文档（可删除）

**建议**: 删除过时文档，保留 11 个核心文档

