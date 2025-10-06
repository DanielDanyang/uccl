# TCPX Multi-Channel Documentation Index

**Last Updated**: 2025-10-06

---

## 🚨 CRITICAL - Read First

| Document | Purpose | Status |
|----------|---------|--------|
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Current bugs and immediate action required | ⚠️ **URGENT** |
| [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) | GPU-NIC topology analysis and lessons learned | ⚠️ **IMPORTANT** |

**TL;DR**: Code has a bug that needs revert. GPU can only use NICs on same PCIe root complex.

---

## 📚 Core Documentation

### Getting Started (5-10 minutes)

| Document | Purpose | Lines |
|----------|---------|-------|
| [../QUICKSTART.md](../QUICKSTART.md) | Build and run in 5 minutes | ~100 |
| [../HANDOFF.md](../HANDOFF.md) | Complete project overview | ~500 |
| [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md) | Common issues and solutions | ~300 |

### Understanding the Code (1-2 hours)

| Document | Purpose | Lines |
|----------|---------|-------|
| [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md) | Detailed code walkthrough | 1100+ |
| [SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md) | Sliding window visualization | ~200 |
| [TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md) | TCPX API reference | ~400 |

---

## 🐛 Bug Fixes and Issues

### Critical Fixes

| Document | Issue | Status |
|----------|-------|--------|
| [SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md) | Sliding window check must be BEFORE tcpx_irecv | ✅ Fixed |
| [SERVER_17_CHUNKS_BUG.md](SERVER_17_CHUNKS_BUG.md) | Server only processed 17/128 chunks | ✅ Fixed |
| [FIXES_APPLIED.md](FIXES_APPLIED.md) | 7 code quality fixes (ODR, nullptr, etc.) | ✅ Fixed |
| [CRITICAL_FIXES.md](CRITICAL_FIXES.md) | Multi-channel fixes (one reverted) | ⚠️ Partial |

### Current Issues

| Issue | Document | Priority |
|-------|----------|----------|
| GPU-NIC topology mismatch | [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) | 🔴 **HIGH** |
| Code needs revert | [CURRENT_STATUS.md](CURRENT_STATUS.md) | 🔴 **HIGH** |

---

## 🏗️ Architecture and Design

### Project Milestones

| Document | Milestone | Status |
|----------|-----------|--------|
| [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) | Infrastructure modules (SlidingWindow, Bootstrap, ChannelManager) | ✅ Complete |
| [PHASE3_COMPLETE.md](PHASE3_COMPLETE.md) | Multi-channel performance test | ✅ Complete |

### Key Concepts

| Concept | Document | Description |
|---------|----------|-------------|
| Sliding Window | [SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md) | Flow control (max 16 inflight per channel) |
| Multi-Channel | [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) | Multiple TCPX connections for multi-NIC |
| GPU-NIC Topology | [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) | PCIe constraints on GPU-NIC pairing |
| Unpack Kernel | [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md) | GPU kernel for reassembling packets |

---

## 🔧 Reference and Troubleshooting

### Quick Reference

| Topic | Document | Use Case |
|-------|----------|----------|
| Common mistakes | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Debugging checklist |
| TCPX API | [TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md) | API usage reference |
| Environment setup | [CURRENT_SETUP.md](CURRENT_SETUP.md) | GCP H100 configuration |
| Performance history | [PERF_DIARY.md](PERF_DIARY.md) | Optimization timeline |

### Error Messages

| Error | Document | Section |
|-------|----------|---------|
| "rx no cmsg" (topology) | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Error 5 |
| "rx no cmsg" (devmem) | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Error 6 |
| "unable to allocate requests" | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Error 2 |
| Kernel 100× slower | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Error 1 |
| Data verification failed | [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) | Error 3 |

---

## 📊 Document Statistics

### Total Documents: 15

**By Category**:
- Current Status: 2 (CURRENT_STATUS, TOPOLOGY_FIX)
- Getting Started: 3 (QUICKSTART, HANDOFF, TROUBLESHOOTING)
- Code Explanation: 3 (TEST_TCPX_PERF_EXPLAINED, SLIDING_WINDOW_VISUAL, TCPX_LOGIC_MAPPING)
- Bug Fixes: 4 (SLIDING_WINDOW_FIX_FINAL, SERVER_17_CHUNKS_BUG, FIXES_APPLIED, CRITICAL_FIXES)
- Milestones: 2 (PHASE1_COMPLETE, PHASE3_COMPLETE)
- Reference: 3 (COMMON_MISTAKES_AND_FIXES, CURRENT_SETUP, PERF_DIARY)

**By Priority**:
- 🔴 Critical (read immediately): 2
- 🟡 Important (read soon): 5
- 🟢 Reference (as needed): 8

---

## 🎯 Reading Paths

### Path 1: Fix Current Bug (30 minutes)

1. [CURRENT_STATUS.md](CURRENT_STATUS.md) - Understand the bug
2. [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) - Understand why it happened
3. Revert code in `channel_manager.cc`
4. Test single-channel mode

### Path 2: New Developer Onboarding (4-6 hours)

**Day 1** (2 hours):
1. [CURRENT_STATUS.md](CURRENT_STATUS.md) - Current state
2. [../QUICKSTART.md](../QUICKSTART.md) - Build and run
3. [../HANDOFF.md](../HANDOFF.md) - Project overview

**Day 2** (2 hours):
1. [SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md) - Core concept
2. [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md) - Code walkthrough
3. [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - Architecture

**Day 3** (2 hours):
1. [SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md) - Critical bug
2. [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Avoid pitfalls
3. [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) - Recent lessons

### Path 3: Debugging Specific Issue (15-30 minutes)

**"rx no cmsg" error**:
1. [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Error 5 or 6
2. [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) - If topology-related

**Performance issue**:
1. [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Error 1
2. [PERF_DIARY.md](PERF_DIARY.md) - Optimization history

**Hanging/timeout**:
1. [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Error 2 or 4
2. [SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md) - Sliding window

---

## 📝 Document Maintenance

### Recent Changes (2025-10-06)

**Added**:
- CURRENT_STATUS.md
- TOPOLOGY_FIX.md
- INDEX.md (this file)

**Updated**:
- README.md - Reorganized for current state
- FIXES_APPLIED.md - Added fixes #6 and #7
- CRITICAL_FIXES.md - Marked fix #2 as reverted
- COMMON_MISTAKES_AND_FIXES.md - Added GPU-NIC topology error
- PHASE3_COMPLETE.md - Corrected chunk size info

**Removed** (outdated/merged):
- DEBUG_LOGS_ADDED.md, DEBUG_LOGS_REMOVED.md
- CHUNK_SIZE_OPTIMIZATION.md
- MULTI_CHANNEL_DESIGN.md, MULTI_CHANNEL_IMPLEMENTATION_DETAILS.md
- TRANSFER_TEST_REFACTOR.md, tcpx_transfer.md
- PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md
- ADDITIONAL_FIXES.md

### Maintenance Guidelines

**When to update**:
- CURRENT_STATUS.md: When project status changes
- TOPOLOGY_FIX.md: If new topology insights discovered
- COMMON_MISTAKES_AND_FIXES.md: When new errors encountered
- TEST_TCPX_PERF_EXPLAINED.md: When code changes significantly

**When to create new docs**:
- Major bug fixes: Create analysis document
- New features: Update existing or create new
- Performance optimizations: Add to PERF_DIARY.md

**When to archive**:
- Information superseded: Move to docs/archive/
- Temporary documents: Delete after merging

---

## 🔗 External References

- **NCCL Plugin**: https://github.com/google/nccl-plugin-gpudirecttcpx
- **TCPX Paper**: [Google's GPUDirect TCPX whitepaper]
- **GCP H100 Docs**: [GCP A3 instance documentation]

---

**For questions or updates, see [README.md](README.md)**

