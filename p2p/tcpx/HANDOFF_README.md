# Handoff – TCPX Single-Process Orchestrator and Multi-Process Baseline

This doc gives the next developer/AI a fast start: what’s done, what’s pending, how to run, how to debug, and where to look in code.

## 1) Goal & Context
- Goal: High-throughput GPU↔GPU P2P over GPUDirect TCPX on GCP A3-high (2× nodes, 8× H100 each, 4× gVNIC)
- Scope: Single-process orchestrator (primary) + Multi-process baseline (reference)
- Interface: Compatible with nccl-plugin-gpudirecttcpx (Google’s TCPX plugin)

## 2) Current Status (short)
- Control-plane: handshake/listen/connect/registrations complete (single-process)
- Data-plane: Sliding window + continuous progress added; matches multi-process logic
- Build: test_tcpx_perf_orchestrator builds; detailed DEBUG/TRACE ready
- Pending: On-hardware re-run to confirm Iteration 0 completes under CHANNELS=1


## 2.5) New Guidance from Google (2025-10-08)
- One channel ≈ one TCPX connection in NCCL plugin; in NIXL we can operate in terms of TCPX connections directly
- Preferred per-NIC max test: single 200Gbps NIC with ~8 TCPX connections (≈21.26 GB/s ceiling)
- NUMA binding: each GPU should stick to its NUMA-local NICs; OK to hardcode a static GPU→NIC mapping for now (e.g., GPU0,1→NIC0; GPU2,3→NIC1; ...)
- vLLM usage: one process per GPU; each process operates one NIC
- Keep NCCL’s threading mechanism for now
- No app-level ACK required; simple send/recv is sufficient

## Pivot (2025-10-08)
- We are deprecating the single-process orchestrator path for now.
- Focus on the multi-process baseline and configure each GPU to open 4 TCPX connections.
- How to run:
  - Server (node0): `UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi server 0`
  - Client (node1): `UCCL_TCPX_NUM_CHANNELS=4 ./tests/test_tcpx_perf_multi client <SERVER_IP> 0`
- Notes:
  - One TCPX connection ≈ one channel; prefer single 200Gbps NIC + ~8 conns for per-NIC max testing
  - Keep NCCL threading as-is; no app-level ACK
  - Expect send/recv asymmetry due to pipeline depth and lifecycle differences


## 3) Key Files
- Single-process test: tests/test_tcpx_perf_orchestrator.cc (progress_channel added with detailed comments)
- Multi-process test: tests/test_tcpx_perf_multi.cc (reference – opportunistic progress and blocking drain)
- Sliding window: include/sliding_window.h, src/sliding_window.cc (try_release_oldest returns {0,1,-1})
- Channel manager: src/channel_manager.{h,cc}
- Bootstrap/control: src/bootstrap.{h,cc}
- Debug guides: DEBUG_GUIDE.md, DEBUG_TCPX_TEST_ANALYSIS.md
- Summary: REPORT_EXEC_SUMMARY_CN.md

## 4) How to Run (two nodes)
```
# Server (Node 0)
cd p2p/tcpx
./test_step3_bandwidth.sh server

# Client (Node 1)
./test_step3_bandwidth.sh client <SERVER_IP>
```
Environment knobs you may set:
```
export UCCL_TCPX_NUM_CHANNELS=1          # start minimal
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_GPUDIRECTTCPX_DEBUG_LEVEL=TRACE
```

## 5) What Good Looks Like
- Server posts first 16 recvs, then progress log shows requests being released
- Client shows “released send request …” repeatedly
- Iteration 0 completes and bandwidth is printed

## 6) If It Stalls (most common)
Symptoms:
- Server window reaches 16/16 and stays; tcpx_test logs show rc=0, done=0 repeatedly
- Client prints a few “opportunistically released …” and then nothing
Actions:
- Confirm code paths call progress_channel() non-blocking after each post (current + all other channels)
- Confirm window-full path uses progress_channel(blocking=true)
- Grep rc/done/size logs from SlidingWindow::try_release_oldest
- Enable TRACE (see env above) and check for next_transmitting changes

## 7) Code Reading Checklist (critical logic)
- tests/test_tcpx_perf_orchestrator.cc:
  - Server: see progress_channel (recv side) and its uses after tcpx_irecv and when window is full
  - Client: same pattern with send side
  - Only FIFO head is tested in try_release_oldest (matches TCPX rq.next_transmitting)
- tests/test_tcpx_perf_multi.cc (reference):
  - process_completed_chunk(): while-loop polling with blocking/non-blocking semantics
  - After each post: opportunistic drain all channels

## 8) Semantics to Respect
- tcpx_test():
  - Should be called on FIFO head; rc!=0 -> real error in baseline; rc=0, done in {0,1}
  - In single-process, frequent calls are necessary to drive tcpxCommProgress
- Recv lifecycle:
  - irecv → test(done=1) → unpack metadata (if kernel path) → consumed
  - memcpy path: consumed after done=1 (no kernel), but see plugin behavior if metadata exists

## 9) Email to Google (questions)
- See questions prepared around channel→socket mapping, progress cadence, multi-NIC scheduling, recv lifecycle, tuning, diagnostics, and full-mesh. Please incorporate replies into code/comments.

## 10) Next Steps (suggested)
1. Re-run minimal case (CHANNELS=1), verify Iteration 0 completes
2. Scale channels (e.g., 2, 4, 8) and record bandwidth per GPU
3. If stalls persist: collect TRACE and share with Google; double-check consumed timing
4. Consider a dedicated progress thread per-comm if single-process cadence is still insufficient

## 11) Known Risks
- Single-process is sensitive to polling cadence (sleep too small: CPU burn; too large: slow progress)
- TCP/socket backpressure may interact with window=16; adjust chunk size/sleep as needed

## 12) Quick Log Scrapes
```
ls -lt logs/*.log | head -6
grep -n "SlidingWindow.*tcpx_test" logs/singleproc_server_*.log | head -50
grep -n "opportunistically released" logs/singleproc_client_*.log | head -50
```

Good luck – the core behavior now matches the working multi-process reference; the remaining gap is mostly runtime cadence/tuning under single-process.

