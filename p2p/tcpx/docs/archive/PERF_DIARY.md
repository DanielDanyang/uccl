# UCCL TCPX Perf Diary

This doc records the intent, attempts, results, and current status for the TCPX point-to-point perf test, so the next engineer can quickly continue.

## Goal
- Measure end-to-end TCPX bandwidth/latency between 2 H100 nodes using our TCPX (GPUDirect TCP) path.
- Keep the perf implementation faithful to the working single-shot transfer path (test_tcpx_transfer), but support large payloads and iterations.

## Environment
- 2 nodes, each with H100 GPUs; TCPX only (no RDMA).
- Bootstrap over TCP (separate control socket) to exchange NCCL handle.
- Data plane over TCPX (Google GPUDirect TCPX plugin), device memory only.
- Node list in scripts/node_ips/tcpx.txt

## What NCCL test does (reference)
- See collective/rdma/run_nccl_test_tcpx.sh
  - Uses mpirun across nodes with HOSTFILE=scripts/node_ips/tcpx.txt
  - Key envs: NCCL_P2P_NET_CHUNKSIZE=524288, NCCL_BUFFSIZE=8388608, NCCL_ALGO=Ring, NCCL_PROTO=Simple, NCCL_SOCKET_IFNAME=eth0, plus TCPX-specific bindings.
- We mirrored the “chunked transfer” idea from NCCL by introducing an adjustable chunk size for our perf test.

## Our perf test design (tests/test_tcpx_perf)
- Parameters via env:
  - UCCL_TCPX_PERF_SIZE (default 4MB)
  - UCCL_TCPX_PERF_ITERS (default 10)
  - UCCL_TCPX_CHUNK_BYTES (optional)
    - If unset, falls back to NCCL_P2P_NET_CHUNKSIZE; default 512KB.
- Server (per iteration):
  - For offset in [0..size) step chunk:
    - Post tcpx_irecv(dst = recv_buf + offset, size = chunk)
    - Poll tcpx_test until done
    - Unpack (using device unpack kernel) from bounce buffer into dst
    - tcpx_irecv_consumed to release resources
  - Aggregate timing
- Client (per iteration):
  - For offset in [0..size) step chunk:
    - tcpx_isend(src = send_buf + offset, size = chunk)
    - Poll tcpx_test until done
  - Aggregate timing

## Key lessons discovered during development
1. Unpack is required on server after tcpx_test reports done=1; otherwise data is never materialized into user buffer.
2. Single huge irecv/isend for large payloads risks TCPX bounce buffer backpressure; chunked streaming is necessary.
3. test_tcpx_transfer succeeds because payload is tiny; perf on large payload must stream and consume per-chunk.

## Recent logs and analysis
- Client log (p2p/tcpx/client.log):
  - Iter 0: total bytes=4194304, chunk_bytes=524288
  - Timeout at offset=3145728 (i.e., during the 7th 512KB chunk)
- Server log (p2p/tcpx/server.log):
  - Shows startup and “Iteration 0 … chunk_bytes=524288” line; no further output captured.

Interpretation:
- We progressed through multiple chunks then stalled, which is consistent with backpressure or matching issues between send/recv operations.
- Reusing a single tag for all chunks may confuse plugin-side matching under certain states.

## Fixes just applied (commit in workspace)
- Implemented per-chunk tags to avoid reuse collisions:
  - tag = kTransferTag + iter*10000 + chunk_idx
  - Applied consistently on both client and server sides.
- Added server/client per-chunk debug prints (chunk_idx, tag, size, offset).
- Kept chunked streaming model (default 512KB via NCCL_P2P_NET_CHUNKSIZE or UCCL_TCPX_CHUNK_BYTES).

## What to run next
1. On server node:
   - ./tests/test_tcpx_perf server <gpu_id>
2. On client node:
   - ./tests/test_tcpx_perf client <server_ip> <gpu_id>
3. Optional envs to tune:
   - export NCCL_P2P_NET_CHUNKSIZE=524288
   - export UCCL_TCPX_PERF_SIZE=$((64*1024*1024))  # try 64MB
   - export UCCL_TCPX_PERF_ITERS=10

## Expected outcome
- With per-chunk unique tags and 512KB chunking, the send/recv should progress through all chunks without timeouts.
- The logs will show [PERF][SERVER]/[PERF][CLIENT] lines per chunk; if a stall occurs, we’ll know the exact chunk and tag.

## If issues persist
- Increase poll allowance to rule out transient latency (e.g., raise from 200k to 1,000,000 loops).
- Try smaller/larger chunk sizes (256KB, 1MB) to see sensitivity.
- Verify server sees tcpx_irecv_consumed after each chunk completes.
- Consider reintroducing a lightweight READY/ACK just for pacing if needed (though NCCL pattern usually not needed).
- Compare env differences with run_nccl_test_tcpx.sh (e.g., NIC iface, CPU affinity bindings) if performance is lower than expected.

## Status summary
- Before: Single-shot perf with large payloads hung on tcpx_test due to backpressure.
- Now: Chunked per-iteration streaming with per-chunk unique tags; compiled and ready; next action is to rerun and check logs.

## Quick checklist for the next engineer
- Confirm env: NCCL_P2P_NET_CHUNKSIZE or UCCL_TCPX_CHUNK_BYTES set; size/iters set.
- Run server/client as above; capture p2p/tcpx/server.log and p2p/tcpx/client.log.
- If a timeout occurs, note the reported offset/chunk_idx/tag, and check whether the opposite side posted/polled that chunk.
- Adjust chunk size or poll budget; ensure both sides use the same base tag logic (already coded).
