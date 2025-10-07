# Phase 1: Round-Robin NIC Distribution - README

**Date**: 2025-10-07  
**Status**: ✅ Implemented, Ready for Testing

---

## 🎯 Quick Start

### For Impatient Users:
```bash
# Server (Node 0)
cd /home/daniel/uccl/p2p/tcpx
./test_phase1_4ch.sh server

# Client (Node 1)
./test_phase1_4ch.sh client <SERVER_IP>

# Verify
./verify_nic_distribution.sh
```

**Read**: `QUICKSTART_PHASE1.md` for details

---

## 📚 Documentation Guide

### Start Here:
1. **QUICKSTART_PHASE1.md** - Quick commands and success indicators
2. **PHASE1_CHECKLIST.md** - Step-by-step testing checklist

### For Technical Details:
3. **PHASE1_SUMMARY.md** - Overview of changes and testing plan
4. **PHASE1_ROUNDROBIN_FIX.md** - Deep technical documentation

### For Understanding the Problem:
5. Review logs: `logs/singleproc_server_20251007_083637.log`
   - Lines 70-109: Shows all channels on same NIC (broken)
   - Lines 436-465: Shows accept stall on GPU 5 (broken)

---

## 🐛 What Was Fixed

### The Problem:
- **Before**: Each GPU used only 1 NIC (the best-scoring one)
- **Impact**: GPU 4 and GPU 5 both tried to use 8 channels on eth3
- **Result**: Accept stalled after ~12 channels on same NIC

### The Fix:
- **After**: Channels distributed round-robin across all 4 NICs
- **Impact**: Each NIC gets ~8 channels (4 ch/GPU × 8 GPUs ÷ 4 NICs)
- **Result**: No single NIC is saturated

### Code Changed:
- **File**: `src/channel_manager.cc`
- **Lines**: 266-290 (replaced 266-278)
- **Change**: Round-robin across ALL NICs instead of replicating best NIC

---

## ✅ Expected Results

### With 4 Channels/GPU:
```
✅ All 8 GPUs complete accept
✅ All 4 NICs used (8 channels each)
✅ No "Failed to accept" errors
✅ Total: 32 channels (8 GPUs × 4 channels)
```

### With 8 Channels/GPU (if 4 works):
```
✅ All 8 GPUs complete accept
✅ All 4 NICs used (16 channels each)
✅ No "Failed to accept" errors
✅ Total: 64 channels (8 GPUs × 8 channels)
✅ Bandwidth >5 GB/s (better than 2.75 GB/s baseline)
```

---

## 🔧 Tools Provided

### Testing:
- `test_phase1_4ch.sh` - Run test with 4 channels/GPU
- `run_p2p_singleproc.sh` - General launcher (use UCCL_TCPX_NUM_CHANNELS env var)

### Verification:
- `verify_nic_distribution.sh` - Analyze logs for NIC distribution
- Checks: NIC usage, accept status, errors

### Documentation:
- `QUICKSTART_PHASE1.md` - Quick reference
- `PHASE1_CHECKLIST.md` - Testing checklist
- `PHASE1_SUMMARY.md` - Overview
- `PHASE1_ROUNDROBIN_FIX.md` - Technical details

---

## ⚠️ Known Limitations

### 1. Cross-NUMA Traffic
**Issue**: Round-robin uses ALL 4 NICs, including cross-NUMA NICs  
**Impact**: GPU 0 (NUMA 0) may use eth3/eth4 (NUMA 1)  
**Workaround**: Accept for now, Phase 2 will fix  
**Performance**: Expect 10-20% below optimal

### 2. Not Production-Ready
**Issue**: This is a quick fix to unblock testing  
**Impact**: Performance may be suboptimal  
**Workaround**: Use for testing only  
**Production**: Wait for Phase 2 (NUMA-aware selection)

---

## 📊 Testing Matrix

| Test | Channels/GPU | Total Channels | Channels/NIC | Status |
|------|--------------|----------------|--------------|--------|
| Phase 1a | 4 | 32 | 8 | 🔄 To Test |
| Phase 1b | 8 | 64 | 16 | ⏳ After 1a |
| Stress | 16 | 128 | 32 | ⏳ Optional |

---

## 🚀 Next Steps

### After Phase 1 Passes:
1. **Option A**: Proceed to Step 3 (Data Plane)
   - Add actual data transfer
   - Measure bandwidth
   - Compare with NCCL baseline

2. **Option B**: Implement Phase 2 (NUMA-aware)
   - Select only NUMA-local NICs
   - Optimize for production
   - Maximize performance

**Recommendation**: Option A (Step 3) to unblock progress

---

## 📞 Getting Help

### If Tests Fail:
1. Check `PHASE1_CHECKLIST.md` for troubleshooting
2. Run `verify_nic_distribution.sh` to analyze logs
3. Review logs in `logs/singleproc_*.log`
4. Check `PHASE1_ROUNDROBIN_FIX.md` for technical details

### Common Issues:
- **"Failed to accept"**: Check NIC distribution in logs
- **"Connection refused"**: Ensure server started first
- **"Permission denied"**: Run `chmod +x *.sh`

---

## 📈 Success Criteria

### Phase 1a (4 channels/GPU):
- [x] Code implemented
- [x] Code compiled
- [ ] All 8 GPUs accept successfully
- [ ] All 4 NICs used evenly
- [ ] No errors in logs

### Phase 1b (8 channels/GPU):
- [ ] All 8 GPUs accept successfully
- [ ] All 4 NICs used evenly
- [ ] Bandwidth >5 GB/s
- [ ] No errors in logs

---

## 🎓 Key Insights

### Why This Fix Works:
1. **Even Distribution**: No single NIC is saturated
2. **Simple Logic**: Round-robin is easy to understand and debug
3. **Unblocks Testing**: Allows us to proceed with data plane work

### Why This Isn't Perfect:
1. **Cross-NUMA**: GPU 0 (NUMA 0) uses eth3/eth4 (NUMA 1)
2. **Suboptimal**: Performance could be 10-20% better with NUMA-aware selection
3. **Not Production**: Need Phase 2 for production deployment

### The Trade-off:
- **Phase 1**: Quick fix, unblocks testing, suboptimal performance
- **Phase 2**: Proper fix, optimal performance, more complex

**Decision**: Phase 1 first to validate approach, Phase 2 later for optimization

---

## 📁 File Structure

```
p2p/tcpx/
├── src/
│   └── channel_manager.cc          # Modified (lines 266-290)
├── tests/
│   └── test_tcpx_perf_orchestrator.cc  # Unchanged
├── test_phase1_4ch.sh              # NEW: Test script
├── verify_nic_distribution.sh      # NEW: Verification script
├── PHASE1_README.md                # NEW: This file
├── QUICKSTART_PHASE1.md            # NEW: Quick reference
├── PHASE1_CHECKLIST.md             # NEW: Testing checklist
├── PHASE1_SUMMARY.md               # NEW: Overview
└── PHASE1_ROUNDROBIN_FIX.md        # NEW: Technical docs
```

---

## ✅ Pre-Flight Checklist

Before testing:
- [x] Code modified
- [x] Code compiled
- [x] Test scripts created
- [x] Verification tools created
- [x] Documentation written
- [ ] Tests run on GCP
- [ ] Results verified

---

**Ready to test!** Start with `QUICKSTART_PHASE1.md` or `PHASE1_CHECKLIST.md`.

**Questions?** Check `PHASE1_SUMMARY.md` or `PHASE1_ROUNDROBIN_FIX.md`.

**Good luck!** 🚀

