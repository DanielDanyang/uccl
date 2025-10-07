# Next Steps - Action Plan

**Date**: 2025-10-07  
**Status**: Diagnostics analysis complete, ready for optimization  
**Current Performance**: P2P 2.75 GB/s vs NCCL 19.176 GB/s bus BW

---

## Key Finding from Diagnostics Analysis

❌ **IRQ affinity is NOT the bottleneck** - NCCL uses default IRQ distribution  
✅ **Thread CPU affinity IS important** - NCCL pins threads to NUMA-local cores  
✅ **Multi-NIC parallelism IS critical** - NCCL uses all 4 NICs simultaneously  
✅ **Process architecture matters** - Single process enables NIC sharing

**Read full analysis**: `p2p/tcpx/docs/DIAGNOSTICS_ANALYSIS.md`

---

## Recommended Optimization Path

### Phase 1: Quick Wins (1-2 days)
**Goal**: Improve single-NIC P2P performance by 20-50%

#### Task 1.1: Add Thread CPU Affinity
**Effort**: 2-3 hours  
**Expected gain**: 10-30% bandwidth improvement

**Implementation**:
1. Modify `run_p2p_fullmesh.sh` to export NCCL binding environment variables
2. Match NCCL's NUMA-local core allocation pattern
3. Test and measure impact

**Environment variables to add**:
```bash
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
```

**Success criteria**:
- Bandwidth increases to >3.5 GB/s (server) or >1.5 GB/s (client)
- No errors or warnings
- CPU usage concentrated on bound cores (verify with `mpstat`)

#### Task 1.2: Tune Window Size
**Effort**: 1 hour  
**Expected gain**: 5-15% bandwidth improvement

**Test matrix**:
| Window Size | Expected Behavior |
|-------------|-------------------|
| 16 (current) | Baseline |
| 32 | More pipelining, may improve |
| 64 | Even more pipelining |
| 128 | May saturate, diminishing returns |

**Command**:
```bash
UCCL_TCPX_WINDOW_SIZE=32 ./run_p2p_fullmesh.sh server
```

**Success criteria**:
- Find optimal window size (highest bandwidth)
- Document sweet spot for future use

#### Task 1.3: Tune Chunk Size
**Effort**: 1 hour  
**Expected gain**: 5-10% bandwidth improvement

**Test matrix**:
| Chunk Size | Expected Behavior |
|------------|-------------------|
| 262144 (256 KB) | Smaller chunks, more syscalls |
| 524288 (512 KB, current) | Baseline |
| 1048576 (1 MB) | Larger chunks, fewer syscalls |
| 2097152 (2 MB) | May be too large |

**Success criteria**:
- Find optimal chunk size
- Balance between syscall overhead and memory pressure

---

### Phase 2: Architecture Evaluation (2-3 days)
**Goal**: Determine if single-process architecture enables multi-NIC

#### Task 2.1: Prototype Single-Process P2P
**Effort**: 1 day  
**Expected gain**: Enables multi-NIC (4x theoretical)

**Implementation**:
1. Create new test program: `test_tcpx_perf_single_process.cc`
2. Initialize all 8 GPUs in one process (like NCCL)
3. Create channels for each GPU
4. Test with 1 NIC first (verify no regression)
5. Test with 2 NICs (check for devmem conflicts)
6. Test with 4 NICs (full multi-NIC)

**Key questions to answer**:
- Does single-process eliminate devmem conflicts?
- Can multiple GPUs share the same NIC?
- What is the performance with 4 NICs?

**Success criteria**:
- No devmem conflicts with multi-NIC
- Bandwidth scales with number of NICs
- Performance >10 GB/s per GPU with 4 NICs

#### Task 2.2: Compare Architectures
**Effort**: 4 hours  
**Expected gain**: Decision on path forward

**Comparison matrix**:
| Architecture | Pros | Cons | Performance |
|--------------|------|------|-------------|
| 8-process (current) | Simple, isolated | No NIC sharing | 2.75 GB/s |
| 1-process (new) | NIC sharing, multi-NIC | More complex | TBD |

**Decision criteria**:
- If single-process enables multi-NIC: **Proceed with refactor**
- If single-process still has conflicts: **Investigate plugin limitations**

---

### Phase 3: Full Optimization (3-5 days)
**Goal**: Approach NCCL performance (>15 GB/s bus BW)

#### Task 3.1: Implement Multi-NIC P2P
**Effort**: 2 days  
**Expected gain**: 2-4x bandwidth improvement

**Prerequisites**:
- Single-process architecture working
- No devmem conflicts confirmed

**Implementation**:
1. Modify ChannelManager to support multiple NICs per GPU
2. Distribute channels across NICs (e.g., 2 channels on eth1, 2 on eth2)
3. Implement NIC selection logic (NUMA-aware)
4. Test with 2, 3, 4 NICs

**Success criteria**:
- Bandwidth scales linearly with NICs (2 NICs → 2x, 4 NICs → 4x)
- No devmem conflicts or errors
- Performance >10 GB/s per GPU

#### Task 3.2: Implement Multi-Channel
**Effort**: 1 day  
**Expected gain**: 1.5-2x bandwidth improvement

**Implementation**:
1. Increase `UCCL_TCPX_NUM_CHANNELS` to 4 or 8
2. Distribute channels across NICs
3. Test channel distribution strategies

**Success criteria**:
- Channels distributed evenly across NICs
- No performance degradation (unlike previous multi-channel attempt)
- Performance >15 GB/s per GPU

#### Task 3.3: Final Tuning
**Effort**: 1 day  
**Expected gain**: 10-20% final polish

**Tuning parameters**:
- Window size (per channel)
- Chunk size
- Number of channels
- NIC distribution strategy
- CPU affinity fine-tuning

**Success criteria**:
- Performance within 20% of NCCL (>15 GB/s bus BW)
- Stable and reproducible
- Well-documented configuration

---

## Decision Points

### Decision Point 1: After Phase 1
**Question**: Did thread CPU affinity improve performance significantly?

- **If YES (>20% improvement)**: Continue with Phase 2
- **If NO (<10% improvement)**: Skip to Phase 2 immediately (architecture is the bottleneck)

### Decision Point 2: After Task 2.1
**Question**: Does single-process architecture enable multi-NIC?

- **If YES**: Proceed with Phase 3 (full optimization)
- **If NO**: Investigate plugin limitations or consider alternative approaches

### Decision Point 3: After Phase 3
**Question**: Did we achieve target performance (>15 GB/s bus BW)?

- **If YES**: Document, optimize, and consider upstreaming
- **If NO**: Analyze remaining bottlenecks (profiling, kernel-level investigation)

---

## Timeline Estimate

| Phase | Tasks | Effort | Calendar Time |
|-------|-------|--------|---------------|
| Phase 1 | CPU affinity, window/chunk tuning | 4-5 hours | 1 day |
| Phase 2 | Single-process prototype, evaluation | 1.5 days | 2 days |
| Phase 3 | Multi-NIC, multi-channel, tuning | 4 days | 5 days |
| **Total** | | **~6 days** | **8 days** (with buffer) |

---

## Success Metrics

### Phase 1 Success
- [ ] Thread CPU affinity implemented
- [ ] Bandwidth >3.5 GB/s (server) or >1.5 GB/s (client)
- [ ] Optimal window size identified
- [ ] Optimal chunk size identified

### Phase 2 Success
- [ ] Single-process P2P working
- [ ] Multi-NIC capability confirmed (no devmem conflicts)
- [ ] Decision made on architecture

### Phase 3 Success
- [ ] Multi-NIC P2P working (4 NICs)
- [ ] Multi-channel working (4-8 channels per GPU)
- [ ] Performance >15 GB/s bus BW
- [ ] Within 20% of NCCL performance

### Overall Success
- [ ] P2P benchmark achieves >10 GB/s per GPU
- [ ] Understand root cause of any remaining gap
- [ ] Clear documentation of what works and what doesn't
- [ ] Reproducible configuration

---

## Risk Mitigation

### Risk 1: Thread CPU affinity doesn't help
**Mitigation**: Skip to Phase 2 immediately (don't waste time)

### Risk 2: Single-process still has devmem conflicts
**Mitigation**: Investigate TCPX plugin source code or contact Google for guidance

### Risk 3: Performance plateaus below target
**Mitigation**: Profile with `perf`, analyze kernel-level bottlenecks, consider alternative approaches

### Risk 4: Multi-channel degrades performance (like before)
**Mitigation**: Ensure channels are distributed across NICs, not on same NIC

---

## Immediate Next Action

**START HERE**: 

1. **Read the diagnostics analysis**:
   ```bash
   cat p2p/tcpx/docs/DIAGNOSTICS_ANALYSIS.md
   ```

2. **Implement Task 1.1 (Thread CPU Affinity)**:
   ```bash
   # Edit run_p2p_fullmesh.sh
   vim p2p/tcpx/run_p2p_fullmesh.sh
   
   # Add these lines before launching GPU processes:
   export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
   export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
   ```

3. **Test and measure**:
   ```bash
   # Node 0
   ./run_p2p_fullmesh.sh server
   
   # Node 1
   ./run_p2p_fullmesh.sh client <NODE0_IP>
   
   # Check results
   grep "PERF.*Avg.*BW:" logs/fullmesh_*.log
   ```

4. **Compare to baseline**:
   - Baseline: 2.75 GB/s (server), 1.17 GB/s (client)
   - Target: >3.5 GB/s (server), >1.5 GB/s (client)

5. **Document results**:
   - Create `PHASE1_RESULTS.md` with findings
   - Decide on next steps based on improvement

---

## Questions to Answer

### Phase 1 Questions
- [ ] Does thread CPU affinity improve bandwidth?
- [ ] What is the optimal window size?
- [ ] What is the optimal chunk size?
- [ ] Are there any CPU bottlenecks visible in `mpstat`?

### Phase 2 Questions
- [ ] Does single-process eliminate devmem conflicts?
- [ ] Can multiple GPUs share the same NIC in single-process?
- [ ] What is the performance overhead of single-process vs multi-process?

### Phase 3 Questions
- [ ] How does bandwidth scale with number of NICs?
- [ ] What is the optimal channel distribution strategy?
- [ ] What is the final performance gap vs NCCL?
- [ ] What are the remaining bottlenecks?

---

## Resources

### Documentation
- **Diagnostics Analysis**: `p2p/tcpx/docs/DIAGNOSTICS_ANALYSIS.md`
- **Project Status**: `p2p/tcpx/docs/PROJECT_STATUS.md`
- **IRQ Investigation Plan**: `p2p/tcpx/docs/IRQ_BINDING_INVESTIGATION_PLAN.md` (now obsolete)

### Code
- **Current P2P benchmark**: `p2p/tcpx/run_p2p_fullmesh.sh`
- **Test program**: `p2p/tcpx/tests/test_tcpx_perf_multi.cc`
- **Channel manager**: `p2p/tcpx/src/channel_manager.cc`

### Reference
- **NCCL test**: `collective/rdma/run_nccl_test_tcpx.sh`
- **Diagnostics data**: `diagnostics/`

---

**Last Updated**: 2025-10-07  
**Status**: Ready to start Phase 1  
**Next Action**: Implement thread CPU affinity (Task 1.1)

