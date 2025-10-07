# Archive - Historical Documentation

This directory contains historical documentation from the TCPX P2P project development.

**Date Archived**: 2025-10-07  
**Reason**: Documentation cleanup and consolidation

---

## What's Here

### Investigation Documents (Obsolete)
- **IRQ_BINDING_INVESTIGATION_PLAN.md** - Original IRQ investigation plan (superseded by results)
- **IRQ_BINDING_OPTIMIZATION_PLAN.md** - IRQ optimization plan (not needed)
- **NEXT_STEPS_ACTION_PLAN.md** - Old action plan (replaced by SINGLE_PROCESS_PLAN.md)
- **EXECUTIVE_SUMMARY.md** - Verbose summary (replaced by DIAGNOSTICS_SUMMARY.md)

### Detailed Analysis (Reference)
- **DIAGNOSTICS_ANALYSIS.md** - Full 300+ line IRQ/CPU/NIC analysis (see DIAGNOSTICS_SUMMARY.md for concise version)

### Historical Debug Reports
- **DEBUG_ETH2_RX_NO_CMSG.md** - eth2 "rx no cmsg" debug (resolved: loopback issue)
- **SERVER_17_CHUNKS_BUG.md** - Server 17 chunks bug investigation
- **SLIDING_WINDOW_FIX_FINAL.md** - Sliding window fix documentation
- **SLIDING_WINDOW_VISUAL.md** - Visual explanation of sliding window
- **TOPOLOGY_FIX.md** - Topology fix documentation

### Phase Completion Reports
- **PHASE1_COMPLETE.md** - Phase 1 completion report
- **PHASE3_COMPLETE.md** - Phase 3 completion report

### Reference Documents
- **COMMON_MISTAKES_AND_FIXES.md** - Common mistakes and fixes
- **CRITICAL_FIXES.md** - Critical fixes applied
- **FIXES_APPLIED.md** - List of fixes applied
- **CURRENT_SETUP.md** - Old setup documentation
- **CURRENT_STATUS.md** - Old status documentation
- **PERF_DIARY.md** - Performance diary
- **TCPX_LOGIC_MAPPING.md** - TCPX logic mapping
- **TEST_TCPX_PERF_EXPLAINED.md** - Test program explanation

### Old Versions
- **PROJECT_STATUS_OLD.md** - Previous verbose version (replaced by concise version)
- **AI_HANDOFF_PROMPT_OLD.md** - Previous verbose version (replaced by concise version)

---

## Current Active Documentation

See parent directory (`p2p/tcpx/docs/`) for current documentation:

1. **PROJECT_STATUS.md** - Current project status (concise)
2. **DIAGNOSTICS_SUMMARY.md** - IRQ investigation results (concise)
3. **SINGLE_PROCESS_PLAN.md** - Single-process refactor plan
4. **AI_HANDOFF_PROMPT.md** - AI assistant handoff (concise)

---

## Why Archived?

These documents were archived during a documentation cleanup on 2025-10-07 to:
- Reduce clutter and confusion
- Focus on current actionable information
- Preserve historical context for reference
- Improve onboarding for new developers

---

## When to Reference

**Use archived docs when**:
- Investigating similar issues (e.g., "rx no cmsg" errors)
- Understanding historical decisions
- Learning about past debugging approaches
- Researching detailed technical analysis

**Don't use archived docs for**:
- Current project status (use PROJECT_STATUS.md)
- Next steps planning (use SINGLE_PROCESS_PLAN.md)
- Quick reference (use README.md or DIAGNOSTICS_SUMMARY.md)

---

## Key Learnings Preserved

The most important learnings from these documents are now consolidated in:
- **PROJECT_STATUS.md** → "Key Learnings" section
- **DIAGNOSTICS_SUMMARY.md** → "Key Findings" section
- **SINGLE_PROCESS_PLAN.md** → "Risk Mitigation" section

---

**Last Updated**: 2025-10-07  
**Status**: Archive complete, current docs streamlined

