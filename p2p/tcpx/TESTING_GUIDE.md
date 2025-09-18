# TCPX Testing Guide

## 🎯 Overview

This guide provides step-by-step instructions for testing TCPX functionality on your H100 machines (10.0.0.107 and 10.0.1.25).

## 📋 Prerequisites

- Two H100 machines with TCPX plugin installed
- TCPX plugin at `/usr/local/tcpx/lib64/libnccl-net-tcpx.so`
- Network connectivity between nodes
- Shared filesystem or manual file copying capability

## 🧪 Test Sequence

### Phase 1: Basic Functionality ✅

#### Test 1: Device Discovery
```bash
# On any node
cd /mnt/user_storage/uccl/p2p/tcpx
make test_device_discovery
./tests/test_device_discovery
```

**Expected Result**: Should find 4 TCPX devices (eth1-eth4)

#### Test 2: Data Transfer APIs
```bash
# On any node
make test_data_transfer
./tests/test_data_transfer
```

**Expected Result**: 
- TCPX plugin loads successfully
- Memory registration APIs are available (but fail with invalid handles)
- Data transfer APIs are available (but fail with invalid handles)

### Phase 2: Connection Testing 🚧

#### Test 3: Full Connection with Data Transfer

**Step 1: Compile the test**
```bash
make test_connection
```

**Step 2: Run Server (Node 1: 10.0.0.107)**
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection server
```

**Step 3: Run Client (Node 2: 10.0.1.25)**
```bash
export UCCL_TCPX_DEBUG=1
./tests/test_connection client 10.0.0.107
```

**Expected Flow (Network-based Handle Exchange)**:
1. Server creates TCPX listening socket and handle
2. Server creates bootstrap TCP server on port 12345
3. Server waits for client to connect to bootstrap server
4. Client connects to server's bootstrap socket (port 12345)
5. Server sends TCPX handle to client via bootstrap connection
6. Client receives handle and extracts connection info
7. Client connects to server using TCPX handle
8. Server accepts TCPX connection
9. **NEW**: Server posts receive request for data
10. **NEW**: Client sends test message "Hello from TCPX client!"
11. **NEW**: Both sides verify data transfer completion

**Key Improvements**:
- ✅ **No shared filesystem required** - uses TCP sockets for handle exchange
- ✅ **Network-based coordination** - similar to RDMA's bootstrap mechanism
- ✅ **Automatic retry logic** - client retries bootstrap connection
- ✅ **Proper error handling** - graceful failure and cleanup

## 🔍 What Each Test Validates

### test_device_discovery
- ✅ TCPX plugin loading
- ✅ Device enumeration (should find 4 devices)
- ✅ Basic plugin initialization

### test_data_transfer  
- ✅ Memory registration API availability
- ✅ Data transfer API availability
- ✅ Proper error handling for invalid parameters

### test_connection
- ✅ Connection establishment (tcpx_listen/tcpx_connect_v5/tcpx_accept_v5)
- ✅ Handle exchange mechanism
- 🆕 Memory registration (tcpx_reg_mr/tcpx_dereg_mr)
- 🆕 Asynchronous data transfer (tcpx_isend/tcpx_irecv)
- 🆕 Completion polling (tcpx_test)

## 🚨 Troubleshooting

### Connection Issues

**Problem**: Client cannot connect to bootstrap server
```
✗ FAILED: Cannot connect to bootstrap server
```

**Solution**:
1. Ensure server is running and has created bootstrap server
2. Check network connectivity: `ping 10.0.0.107`
3. Verify port 12345 is not blocked by firewall
4. Check if port is already in use: `netstat -an | grep 12345`

**Problem**: Connection timeout or failure
```
✗ FAILED: tcpx_connect_v5 returned -1
```

**Solutions**:
1. Check network connectivity: `ping 10.0.0.107`
2. Verify TCPX plugin version compatibility
3. Check firewall settings
4. Ensure both nodes have same TCPX configuration

### Data Transfer Issues

**Problem**: Memory registration fails
```
✗ WARNING: tcpx_reg_mr failed with rc=-1
```

**Analysis**: This might be expected behavior - some TCPX implementations may not require explicit memory registration for CPU memory.

**Problem**: Send/Receive timeout
```
✗ TIMEOUT: No data received after 1000 polls
```

**Solutions**:
1. Increase polling timeout
2. Check if connection handles are valid
3. Verify tag matching between send and receive
4. Enable more detailed TCPX debugging

## 📊 Success Criteria

### Minimum Success (Connection Only)
- ✅ Device discovery finds 4 devices
- ✅ Server can listen and accept connections
- ✅ Client can connect to server
- ✅ No crashes or stack overflows

### Full Success (Data Transfer)
- ✅ All connection functionality works
- ✅ Memory registration succeeds (or gracefully handles failure)
- ✅ Client successfully sends test message
- ✅ Server successfully receives test message
- ✅ Data integrity verified (message content matches)

## 🎯 Next Steps After Success

Once connection and data transfer tests pass:

1. **Endpoint Integration**: Integrate TCPX into the main Endpoint class
2. **GPU Memory Support**: Test with CUDA memory instead of CPU memory
3. **Performance Testing**: Measure bandwidth and latency
4. **Multi-connection Testing**: Test multiple simultaneous connections
5. **Production Integration**: Replace RDMA calls in the main codebase

## 🔧 Debug Environment Variables

```bash
export UCCL_TCPX_DEBUG=1          # Enable our debug output
export NCCL_DEBUG=INFO            # Enable NCCL debug output
export NCCL_DEBUG_SUBSYS=NET      # Focus on network subsystem
```

## 📝 Logging

All test output should be captured for analysis:
```bash
./tests/test_connection server 2>&1 | tee server.log
./tests/test_connection client 10.0.0.107 2>&1 | tee client.log
```

This comprehensive testing approach will validate that TCPX can successfully replace RDMA for your P2P communication needs.
