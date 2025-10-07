# TCPX P2P Documentation

**Last Updated**: 2025-10-07  
**Status**: Documentation cleanup complete

---

## Quick Navigation

### 🚀 Getting Started
1. **[../README.md](../README.md)** - Project overview and quick start
2. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current status and timeline
3. **[AI_HANDOFF_PROMPT.md](AI_HANDOFF_PROMPT.md)** - Context for new developers

### 📊 Investigation Results
1. **[DIAGNOSTICS_SUMMARY.md](DIAGNOSTICS_SUMMARY.md)** - IRQ investigation results (concise)
2. **[archive/DIAGNOSTICS_ANALYSIS.md](archive/DIAGNOSTICS_ANALYSIS.md)** - Full detailed analysis

### 📋 Implementation Plan
1. **[SINGLE_PROCESS_PLAN.md](SINGLE_PROCESS_PLAN.md)** - Single-process refactor plan

### 📦 Archive
- **[archive/README.md](archive/README.md)** - Historical documentation index

---

## Core Documents

| Document | Purpose | Length |
|----------|---------|--------|
| **PROJECT_STATUS.md** | Current status, timeline, next steps | ~200 lines |
| **DIAGNOSTICS_SUMMARY.md** | IRQ investigation results (concise) | ~150 lines |
| **SINGLE_PROCESS_PLAN.md** | Single-process refactor plan | ~250 lines |
| **AI_HANDOFF_PROMPT.md** | Context for new AI assistants | ~200 lines |

---

## Reading Paths

### For New Developers
1. Read **../README.md** (project overview)
2. Read **PROJECT_STATUS.md** (current status)
3. Read **DIAGNOSTICS_SUMMARY.md** (investigation results)
4. Read **SINGLE_PROCESS_PLAN.md** (next steps)
5. Run current P2P to verify environment

### For AI Assistants
1. Copy context from **AI_HANDOFF_PROMPT.md**
2. Read **PROJECT_STATUS.md**
3. Read **SINGLE_PROCESS_PLAN.md**
4. Start implementation

### For Quick Reference
- **Current status**: PROJECT_STATUS.md
- **Why IRQ affinity doesn't matter**: DIAGNOSTICS_SUMMARY.md
- **What to do next**: SINGLE_PROCESS_PLAN.md
- **How to run tests**: ../README.md

---

## Key Findings

**Question**: Why does NCCL achieve 19.176 GB/s while P2P achieves only 2.75 GB/s?

**Answer**:
- ❌ **NOT** IRQ affinity (NCCL uses default)
- ✅ **YES** Thread CPU affinity (NCCL pins to NUMA-local cores)
- ✅ **YES** Multi-NIC parallelism (NCCL uses 4 NICs, P2P uses 1)
- ✅ **YES** Process architecture (NCCL: 1 proc enables NIC sharing)

**Next**: Single-process refactor (see SINGLE_PROCESS_PLAN.md)

---

## Documentation Cleanup (2025-10-07)

**Before**: 23 files in docs/, confusing and redundant  
**After**: 5 core files + organized archive

**Changes**:
- ✅ Consolidated 23 docs → 5 core docs + archive
- ✅ Created concise versions (PROJECT_STATUS, DIAGNOSTICS_SUMMARY, AI_HANDOFF_PROMPT)
- ✅ Moved 18 historical docs to archive/
- ✅ Created SINGLE_PROCESS_PLAN.md
- ✅ Created archive/README.md

---

**Last Updated**: 2025-10-07  
**Status**: 5 core docs + organized archive, clear reading path
