# 当前环境配置

## 节点 IP 地址

| 角色 | IP 地址 | 说明 |
|------|---------|------|
| **Server (Node 1)** | `10.65.74.150` | 接收端，运行 server 模式 |
| **Client (Node 2)** | `10.64.113.77` | 发送端，运行 client 模式 |

配置文件: `scripts/node_ips/tcpx.txt`

---

## 快速启动命令

### 单次测试 (64MB)

**在 Server 节点 (10.65.74.150) 上运行:**
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh server 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

**在 Client 节点 (10.64.113.77) 上运行:**
```bash
cd /home/daniel/uccl/p2p/tcpx
./bench_p2p.sh client 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --size=67108864 --no-unix
```

### 完整性能扫描 (4KB → 256MB)

**在 Server 节点上运行:**
```bash
./bench_p2p_sweep_server.sh 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
```

**在 Client 节点上运行:**
```bash
./bench_p2p_sweep_client.sh 10.65.74.150 0 --ifaces=eth1,eth2,eth3,eth4 --no-unix
```

结果会保存在: `logs/p2p_sweep_YYYYMMDD_HHMMSS.md`

---

## 网络配置

### 网卡接口
```bash
# 数据传输网卡 (4 个 25Gbps NICs)
eth1, eth2, eth3, eth4

# 控制网卡
eth1 (默认)
```

### 端口范围
```bash
# TCPX 使用的端口范围
50000 - 60000
```

### 环境变量
```bash
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
```

---

## 验证连接

### 1. 检查网络可达性

**从 Client 节点 ping Server:**
```bash
ping -c 3 10.65.74.150
```

**预期输出:**
```
64 bytes from 10.65.74.150: icmp_seq=1 ttl=64 time=0.xxx ms
```

### 2. 检查端口开放

**在 Server 节点上:**
```bash
# 检查防火墙规则
sudo iptables -L | grep 50000

# 或者临时开放端口范围
sudo iptables -I INPUT -p tcp --dport 50000:60000 -j ACCEPT
```

### 3. 检查 devmem-tcp

**在两个节点上都运行:**
```bash
dmesg | grep devmem
```

**预期输出:**
```
[    X.XXXXXX] TCP: devmem-tcp enabled
```

### 4. 检查网卡状态

**在两个节点上都运行:**
```bash
# 检查网卡是否启用 tcp-data-split
for iface in eth1 eth2 eth3 eth4; do
  echo "=== $iface ==="
  ethtool -k $iface | grep tcp-data-split
  ip addr show $iface | grep "inet "
done
```

**预期输出:**
```
=== eth1 ===
tcp-data-split: on
    inet 10.65.x.x/20 ...
```

---

## 故障排查

### 问题 1: "Connection refused"

**可能原因:**
- Server 端未启动
- 防火墙阻止连接
- IP 地址错误

**解决方案:**
```bash
# 1. 确认 Server 端已启动并在监听
# 2. 检查 IP 地址是否正确
# 3. 检查防火墙
sudo iptables -L -n | grep 50000
```

### 问题 2: "rx no cmsg"

**可能原因:**
- devmem-tcp 未启用
- 使用了错误的 IP 地址范围

**解决方案:**
```bash
# 检查 devmem-tcp
dmesg | grep devmem

# 确保使用正确的 IP 地址
# Server: 10.65.74.150 (不是 localhost)
# Client: 连接到 10.65.74.150
```

### 问题 3: 性能低于预期

**预期性能 (64MB):**
- 延迟: 8-25 ms
- 带宽: 15-20 GB/s (4-NIC 配置)

**如果性能低:**
```bash
# 1. 检查是否使用了所有 4 个网卡
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# 2. 检查 unpack 实现
export UCCL_TCPX_UNPACK_IMPL=kernel  # 应该使用 kernel 模式

# 3. 检查 chunk 大小
export UCCL_TCPX_CHUNK_BYTES=524288  # 默认 512KB
```

---

## 性能基准

### 预期性能 (4-NIC 配置)

| 消息大小 | 延迟 (ms) | 带宽 (GB/s) |
|---------|----------|------------|
| 4 KB | 0.1 | 0.04 |
| 64 KB | 0.5 | 0.13 |
| 1 MB | 2 | 0.5 |
| 16 MB | 10 | 1.6 |
| 64 MB | 25 | 2.5 |
| 256 MB | 80 | 3.2 |

**注意:** 实际性能可能因网络条件和系统负载而异。

---

## 日志位置

```
p2p/tcpx/logs/
├── bench_server_YYYYMMDD_HHMMSS.log  # Server 端日志
├── bench_client_YYYYMMDD_HHMMSS.log  # Client 端日志
└── p2p_sweep_YYYYMMDD_HHMMSS.md      # 性能扫描结果
```

---

## 更新历史

- **2025-10-02**: 更新 IP 地址
  - Server: 10.64.52.73 → 10.65.74.150
  - Client: 10.64.113.74 → 10.64.113.77
- **2025-10-01**: 初始配置

---

**最后更新**: 2025-10-02  
**维护者**: Daniel

