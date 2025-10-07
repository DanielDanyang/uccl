# TCPX P2P Documentation

This directory contains detailed technical documentation for the TCPX P2P project.

---

## 🚨 START HERE

### Current Status (2025-10-06)

⚠️ **IMPORTANT**: The code currently has a bug that needs to be reverted. See:
1. **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - Current state and immediate action required
2. **[TOPOLOGY_FIX.md](TOPOLOGY_FIX.md)** - Analysis of the bug and lessons learned

---

## 📚 Essential Documents (Read These First)

### For Getting Started

1. **[../QUICKSTART.md](../QUICKSTART.md)** - Get up and running in 5 minutes
2. **[../HANDOFF.md](../HANDOFF.md)** - Complete project overview and handoff guide
3. **[../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)** - Common issues and solutions

### For Understanding the Code

4. **[TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md)** - Detailed code explanation (1100+ lines)
5. **[SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md)** - Sliding window mechanism visualization

---

## 📖 Reference Documents

### Current Status and Issues

- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - ⚠️ Current state, bugs, and action items
- **[TOPOLOGY_FIX.md](TOPOLOGY_FIX.md)** - ⚠️ GPU-NIC topology analysis and lessons learned

### Bug Fixes and Optimizations

- **[SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md)** - Critical sliding window bug fix
- **[SERVER_17_CHUNKS_BUG.md](SERVER_17_CHUNKS_BUG.md)** - Analysis of 17-chunk bug
- **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - All 7 code quality fixes
- **[CRITICAL_FIXES.md](CRITICAL_FIXES.md)** - ⚠️ Multi-channel fixes (one reverted)

### Technical Reference

- **[TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md)** - TCPX API mapping and logic
- **[COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md)** - Common mistakes and how to fix them

### Project Milestones

- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** - Infrastructure modules (SlidingWindow, Bootstrap, ChannelManager)
- **[PHASE3_COMPLETE.md](PHASE3_COMPLETE.md)** - Multi-channel performance test

### Historical

- **[PERF_DIARY.md](PERF_DIARY.md)** - Performance optimization history

### Setup

- **[CURRENT_SETUP.md](CURRENT_SETUP.md)** - Current environment setup

---

## 🗂️ Document Organization

### By Purpose

**Current Status** (⚠️ READ FIRST):
- CURRENT_STATUS.md - Current state and action items
- TOPOLOGY_FIX.md - GPU-NIC topology lessons learned

**Getting Started**:
- QUICKSTART.md (in parent directory)
- HANDOFF.md (in parent directory)

**Troubleshooting**:
- TROUBLESHOOTING.md (in parent directory)
- COMMON_MISTAKES_AND_FIXES.md

**Understanding the Code**:
- TEST_TCPX_PERF_EXPLAINED.md (detailed annotations)
- SLIDING_WINDOW_VISUAL.md (visual explanation)
- TCPX_LOGIC_MAPPING.md (API reference)

**Bug Fixes**:
- SLIDING_WINDOW_FIX_FINAL.md (critical fix)
- SERVER_17_CHUNKS_BUG.md (bug analysis)
- FIXES_APPLIED.md (7 code quality fixes)
- CRITICAL_FIXES.md (multi-channel fixes, one reverted)

**Project Milestones**:
- PHASE1_COMPLETE.md (infrastructure modules)
- PHASE3_COMPLETE.md (multi-channel test)

**Historical**:
- PERF_DIARY.md (optimization history)

**Setup**:
- CURRENT_SETUP.md (environment configuration)

---

## 🎯 Reading Path for New Developers

### Day 1: Understand Current State

1. ⚠️ Read [CURRENT_STATUS.md](CURRENT_STATUS.md) - **START HERE**
2. ⚠️ Read [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md) - Understand the current bug
3. Read [../QUICKSTART.md](../QUICKSTART.md) - Build and run

### Day 2: Understand the System

1. Read [../HANDOFF.md](../HANDOFF.md) - Complete overview
2. Read [SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md) - Understand sliding window
3. Skim [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md) - Code structure

### Day 3: Deep Dive

1. Read [SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md) - Critical bug fix
2. Read [TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md) - TCPX API details
3. Read [FIXES_APPLIED.md](FIXES_APPLIED.md) - Code quality improvements

### Day 4+: Start Contributing

1. Read [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Avoid pitfalls
2. Read [PERF_DIARY.md](PERF_DIARY.md) - Learn from history
3. Fix the current bug (see CURRENT_STATUS.md)

---

## 🔍 Quick Reference

### Find Information About...

**Current Issues** (⚠️ START HERE):
- CURRENT_STATUS.md - Current bugs and action items
- TOPOLOGY_FIX.md - GPU-NIC topology constraints

**Sliding window**:
- SLIDING_WINDOW_VISUAL.md - Visual explanation
- SLIDING_WINDOW_FIX_FINAL.md - Critical bug fix
- TEST_TCPX_PERF_EXPLAINED.md (lines 300-400)

**Multi-channel architecture**:
- PHASE1_COMPLETE.md - Infrastructure modules
- PHASE3_COMPLETE.md - Performance test
- CURRENT_STATUS.md - Current state

**Unpack kernel**:
- TEST_TCPX_PERF_EXPLAINED.md (lines 600-700)
- PERF_DIARY.md (kernel performance fix)

**TCPX API**:
- TCPX_LOGIC_MAPPING.md - API reference
- TEST_TCPX_PERF_EXPLAINED.md (throughout)

**Bug fixes**:
- FIXES_APPLIED.md - 7 code quality fixes
- CRITICAL_FIXES.md - Multi-channel fixes (one reverted)
- SLIDING_WINDOW_FIX_FINAL.md - Sliding window fix
- SERVER_17_CHUNKS_BUG.md - 17-chunk bug analysis

**Performance**:
- PERF_DIARY.md - Optimization history

**Setup**:
- CURRENT_SETUP.md - Environment configuration
- ../QUICKSTART.md - Quick start guide

**Troubleshooting**:
- ../TROUBLESHOOTING.md - Common issues
- COMMON_MISTAKES_AND_FIXES.md - Common mistakes

---

## 📞 Need Help?

1. **Current bug**: Read [CURRENT_STATUS.md](CURRENT_STATUS.md) and [TOPOLOGY_FIX.md](TOPOLOGY_FIX.md)
2. **Quick question**: Check [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
3. **Understanding code**: Read [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md)
4. **Project overview**: Read [../HANDOFF.md](../HANDOFF.md)

---

## 📝 Document Maintenance

### Recent Changes (2025-10-06)

**Added**:
- CURRENT_STATUS.md - Current state and action items
- TOPOLOGY_FIX.md - GPU-NIC topology analysis

**Updated**:
- FIXES_APPLIED.md - Added fixes #6 and #7
- CRITICAL_FIXES.md - Marked fix #2 as reverted
- README.md (this file) - Reorganized for current state

**Removed** (outdated/merged):
- DEBUG_LOGS_ADDED.md, DEBUG_LOGS_REMOVED.md
- CHUNK_SIZE_OPTIMIZATION.md (incorrect optimization)
- MULTI_CHANNEL_DESIGN.md, MULTI_CHANNEL_IMPLEMENTATION_DETAILS.md
- TRANSFER_TEST_REFACTOR.md, tcpx_transfer.md
- PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md
- ADDITIONAL_FIXES.md (merged into FIXES_APPLIED.md)

---

**Last Updated**: 2025-10-06

