# TCPX P2P Documentation

This directory contains detailed technical documentation for the TCPX P2P project.

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

### Bug Fixes and Optimizations

- **[SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md)** - Critical sliding window bug fix
- **[SERVER_17_CHUNKS_BUG.md](SERVER_17_CHUNKS_BUG.md)** - Analysis of 17-chunk bug
- **[CHUNK_SIZE_OPTIMIZATION.md](CHUNK_SIZE_OPTIMIZATION.md)** - Chunk size optimization (512KB → 2MB)
- **[DEBUG_LOGS_REMOVED.md](DEBUG_LOGS_REMOVED.md)** - Debug log removal for performance

### Technical Reference

- **[TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md)** - TCPX API mapping and logic
- **[COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md)** - Common mistakes and how to fix them

### Historical

- **[PERF_DIARY.md](PERF_DIARY.md)** - Performance optimization history
- **[PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md](PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md)** - Performance analysis

### Setup

- **[CURRENT_SETUP.md](CURRENT_SETUP.md)** - Current environment setup

---

## 🗂️ Document Organization

### By Purpose

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

**Optimizations**:
- CHUNK_SIZE_OPTIMIZATION.md
- DEBUG_LOGS_REMOVED.md
- PERF_DIARY.md

**Setup**:
- CURRENT_SETUP.md

---

## 🎯 Reading Path for New Developers

### Day 1: Get Running

1. Read [../QUICKSTART.md](../QUICKSTART.md)
2. Build and run the test
3. If issues, check [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

### Day 2: Understand the System

1. Read [../HANDOFF.md](../HANDOFF.md) - Complete overview
2. Read [SLIDING_WINDOW_VISUAL.md](SLIDING_WINDOW_VISUAL.md) - Understand sliding window
3. Skim [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md) - Code structure

### Day 3: Deep Dive

1. Read [SLIDING_WINDOW_FIX_FINAL.md](SLIDING_WINDOW_FIX_FINAL.md) - Critical bug fix
2. Read [TCPX_LOGIC_MAPPING.md](TCPX_LOGIC_MAPPING.md) - TCPX API details
3. Read [CHUNK_SIZE_OPTIMIZATION.md](CHUNK_SIZE_OPTIMIZATION.md) - Optimization details

### Day 4+: Start Contributing

1. Read [COMMON_MISTAKES_AND_FIXES.md](COMMON_MISTAKES_AND_FIXES.md) - Avoid pitfalls
2. Read [PERF_DIARY.md](PERF_DIARY.md) - Learn from history
3. Start debugging multi-NIC issue (see [../HANDOFF.md](../HANDOFF.md))

---

## 📝 Document Maintenance

### When to Update

- **HANDOFF.md**: When project status changes
- **TROUBLESHOOTING.md**: When new issues are discovered
- **TEST_TCPX_PERF_EXPLAINED.md**: When code changes significantly
- **PERF_DIARY.md**: When new optimizations are made

### When to Create New Documents

- **Bug fixes**: Create a new document explaining the bug and fix
- **Optimizations**: Create a new document explaining the optimization
- **New features**: Update TEST_TCPX_PERF_EXPLAINED.md or create new doc

### When to Archive Documents

- When information is outdated or superseded
- Move to `docs/archive/` directory (create if needed)

---

## 🔍 Quick Reference

### Find Information About...

**Sliding window**:
- SLIDING_WINDOW_VISUAL.md
- SLIDING_WINDOW_FIX_FINAL.md
- TEST_TCPX_PERF_EXPLAINED.md (lines 300-400)

**Chunking**:
- CHUNK_SIZE_OPTIMIZATION.md
- TEST_TCPX_PERF_EXPLAINED.md (lines 200-250)

**Unpack kernel**:
- TEST_TCPX_PERF_EXPLAINED.md (lines 600-700)
- PERF_DIARY.md (kernel performance fix)

**TCPX API**:
- TCPX_LOGIC_MAPPING.md
- TEST_TCPX_PERF_EXPLAINED.md (throughout)

**Performance**:
- PERF_DIARY.md
- PERFORMANCE_ANALYSIS_AND_NEXT_STEPS.md
- CHUNK_SIZE_OPTIMIZATION.md
- DEBUG_LOGS_REMOVED.md

**Setup**:
- CURRENT_SETUP.md
- ../QUICKSTART.md

**Troubleshooting**:
- ../TROUBLESHOOTING.md
- COMMON_MISTAKES_AND_FIXES.md

---

## 📞 Need Help?

1. **Quick question**: Check [../TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
2. **Understanding code**: Read [TEST_TCPX_PERF_EXPLAINED.md](TEST_TCPX_PERF_EXPLAINED.md)
3. **Project overview**: Read [../HANDOFF.md](../HANDOFF.md)
4. **Specific bug/optimization**: Check relevant document above

---

**Last Updated**: 2025-10-02

