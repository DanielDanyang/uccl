# 快速开始 - TCPX Performance Test

## 📋 测试目标

定位 GPU 直收模式下的 "rx no cmsg" 问题，通过 4 轮测试逐步排查。

---

## 🚀 运行步骤（超级简单）

### 准备工作（只需做一次）

```bash
cd /home/daniel/uccl/p2p/tcpx
mkdir -p logs
```

---

## 测试轮次

### ✅ Test 1: Host 接收模式（已验证通过，跳过）

这个测试已经成功，带宽 ~7.75 GB/s。

---

### 🔍 Test 2: GPU 直收 + 多网卡 + kernel 解包

**目的**: 复现 "rx no cmsg" 问题

#### 在 Node 1 (10.64.52.73) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test2_server.sh
```

#### 在 Node 2 (10.64.113.74) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test2_client.sh
```

**日志位置**:
- Server: `logs/test2_gpu_multi_server.log`
- Client: `logs/test2_gpu_multi_client.log`

---

### 🔍 Test 3: GPU 直收 + 单网卡 + kernel 解包

**目的**: 排除多 NIC 问题

#### 在 Node 1 (10.64.52.73) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test3_server.sh
```

#### 在 Node 2 (10.64.113.74) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test3_client.sh
```

**日志位置**:
- Server: `logs/test3_gpu_single_server.log`
- Client: `logs/test3_gpu_single_client.log`

---

### 🔍 Test 4: GPU 直收 + 单网卡 + d2d 解包

**目的**: 排除 kernel 解包问题

#### 在 Node 1 (10.64.52.73) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test4_server.sh
```

#### 在 Node 2 (10.64.113.74) 运行：
```bash
cd /home/daniel/uccl/p2p/tcpx
./run_test4_client.sh
```

**日志位置**:
- Server: `logs/test4_gpu_d2d_server.log`
- Client: `logs/test4_gpu_d2d_client.log`

---

## ⚠️ 重要提示

1. **每次测试的顺序**:
   - 先启动 Server（Node 1）
   - 等待看到 "Listening on port..." 后
   - 再启动 Client（Node 2）

2. **每次测试前清理进程**:
   ```bash
   pkill -9 test_tcpx_perf
   ```

3. **测试完成后**:
   - 把 `logs/` 目录下的所有日志文件发给我
   - 或者直接把关键错误信息截图/复制给我

---

## 📊 如何判断测试结果

### ✅ 成功的标志
```
[PERF] Avg: X.XXX ms, BW: XX.XX GB/s
```

### ❌ 失败的标志

**Server 端**:
```
fatal, ... rx no cmsg
```

**Client 端**:
```
Connection reset by peer
[ERROR] Send timeout at iteration ...
```

---

## 🔧 如果遇到问题

### 问题 1: 找不到脚本
```bash
cd /home/daniel/uccl/p2p/tcpx
ls -la run_test*.sh
# 如果没有执行权限，运行：
chmod +x run_test*.sh
```

### 问题 2: Server 启动失败
```bash
# 检查端口是否被占用
netstat -tuln | grep 12345
# 杀掉旧进程
pkill -9 test_tcpx_perf
```

### 问题 3: Client 连接失败
```bash
# 检查网络连通性
ping 10.64.52.73
# 检查 Server 是否在运行
ps aux | grep test_tcpx_perf
```

---

## 📝 测试记录表

| 测试 | 配置 | 结果 | 带宽 | 备注 |
|------|------|------|------|------|
| Test 1 | Host + 多网卡 | ✅ 成功 | 7.75 GB/s | 已验证 |
| Test 2 | GPU + 多网卡 + kernel | ⏳ 待测 | - | - |
| Test 3 | GPU + 单网卡 + kernel | ⏳ 待测 | - | - |
| Test 4 | GPU + 单网卡 + d2d | ⏳ 待测 | - | - |

---

## 🎯 下一步

完成 Test 2-4 后，把所有日志发给我，我会：
1. 分析哪个配置成功/失败
2. 定位 "rx no cmsg" 的根本原因
3. 提供针对性的修复方案
4. 如果需要，在插件层添加更详细的 cmsg 日志

---

## 📞 需要帮助？

如果有任何问题，直接把：
- 运行的命令
- 完整的错误信息
- 日志文件内容

发给我，我会立即分析！

