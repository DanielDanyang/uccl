#!/bin/bash
# Analyze bandwidth from Step 3 test logs

set -e

SERVER_LOG="${1:-}"
CLIENT_LOG="${2:-}"

if [[ -z "$SERVER_LOG" ]]; then
  # Find the most recent server log
  SERVER_LOG=$(ls -t logs/singleproc_server_*.log 2>/dev/null | head -1)
fi

if [[ -z "$CLIENT_LOG" ]]; then
  # Find the most recent client log
  CLIENT_LOG=$(ls -t logs/singleproc_client_*.log 2>/dev/null | head -1)
fi

if [[ ! -f "$SERVER_LOG" ]]; then
  echo "Error: Server log not found: $SERVER_LOG"
  echo "Usage: $0 [server_log] [client_log]"
  exit 1
fi

if [[ ! -f "$CLIENT_LOG" ]]; then
  echo "Error: Client log not found: $CLIENT_LOG"
  echo "Usage: $0 [server_log] [client_log]"
  exit 1
fi

echo "=========================================="
echo "Bandwidth Analysis"
echo "=========================================="
echo "Server log: $SERVER_LOG"
echo "Client log: $CLIENT_LOG"
echo ""

# Extract test configuration
echo "=== Test Configuration ==="
grep "Test size:" "$SERVER_LOG" | head -1
grep "Iterations:" "$SERVER_LOG" | head -1
grep "Chunk size:" "$SERVER_LOG" | head -1
grep "Total channels:" "$SERVER_LOG" | head -1

echo ""
echo "=== Server Performance ==="

# Extract server bandwidth
if grep -q "Performance Summary" "$SERVER_LOG"; then
  grep -A 3 "Performance Summary" "$SERVER_LOG" | grep -E "Average time:|Average bandwidth:|Total channels"
else
  echo "⚠️  No performance summary found in server log"
  echo "Checking for iteration results..."
  grep "bandwidth:" "$SERVER_LOG" | tail -5
fi

echo ""
echo "=== Client Performance ==="

# Extract client bandwidth
if grep -q "Performance Summary" "$CLIENT_LOG"; then
  grep -A 3 "Performance Summary" "$CLIENT_LOG" | grep -E "Average time:|Average bandwidth:|Total channels"
else
  echo "⚠️  No performance summary found in client log"
  echo "Checking for iteration results..."
  grep "bandwidth:" "$CLIENT_LOG" | tail -5
fi

echo ""
echo "=== Bandwidth Comparison ==="

# Extract average bandwidth values
SERVER_BW=$(grep "Average bandwidth:" "$SERVER_LOG" 2>/dev/null | sed 's/.*: \([0-9.]*\).*/\1/' || echo "N/A")
CLIENT_BW=$(grep "Average bandwidth:" "$CLIENT_LOG" 2>/dev/null | sed 's/.*: \([0-9.]*\).*/\1/' || echo "N/A")

echo "Server: $SERVER_BW GB/s"
echo "Client: $CLIENT_BW GB/s"

# Compare with baseline
BASELINE_BW=2.75
NCCL_BW=19.176

if [[ "$SERVER_BW" != "N/A" ]]; then
  echo ""
  echo "=== Performance vs Baseline ==="
  echo "Baseline (multi-process, 1 ch/GPU): $BASELINE_BW GB/s"
  echo "Current (single-process): $SERVER_BW GB/s"
  
  # Calculate improvement
  if command -v bc &> /dev/null; then
    improvement=$(echo "scale=2; ($SERVER_BW / $BASELINE_BW - 1) * 100" | bc)
    echo "Improvement: ${improvement}%"
    
    nccl_percent=$(echo "scale=2; ($SERVER_BW / $NCCL_BW) * 100" | bc)
    echo "NCCL baseline: $NCCL_BW GB/s"
    echo "Current vs NCCL: ${nccl_percent}%"
  fi
fi

echo ""
echo "=== Error Check ==="

# Check for errors
SERVER_ERRORS=$(grep -c "\[ERROR\]" "$SERVER_LOG" || true)
CLIENT_ERRORS=$(grep -c "\[ERROR\]" "$CLIENT_LOG" || true)

if [[ $SERVER_ERRORS -eq 0 && $CLIENT_ERRORS -eq 0 ]]; then
  echo "✅ No errors detected"
else
  echo "❌ Errors found:"
  echo "   Server errors: $SERVER_ERRORS"
  echo "   Client errors: $CLIENT_ERRORS"
  echo ""
  echo "Server error details:"
  grep "\[ERROR\]" "$SERVER_LOG" || echo "  (none)"
  echo ""
  echo "Client error details:"
  grep "\[ERROR\]" "$CLIENT_LOG" || echo "  (none)"
fi

echo ""
echo "=== NIC Distribution ==="

# Check NIC usage
for netdev in 0 1 2 3; do
  count=$(grep "Channel.*→ netDev $netdev" "$SERVER_LOG" | wc -l)
  nic_name=$(grep "Channel.*→ netDev $netdev" "$SERVER_LOG" | head -1 | sed -n 's/.*(\(eth[0-9]\).*/\1/p' || echo "unknown")
  if [[ -z "$nic_name" ]]; then
    nic_name="unknown"
  fi
  echo "netDev $netdev ($nic_name): $count channels"
done

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="

# Summary
echo ""
if [[ "$SERVER_BW" != "N/A" ]]; then
  if (( $(echo "$SERVER_BW > 5.0" | bc -l) )); then
    echo "✅ SUCCESS: Bandwidth >5 GB/s achieved!"
  elif (( $(echo "$SERVER_BW > $BASELINE_BW" | bc -l) )); then
    echo "⚠️  PARTIAL: Bandwidth improved but <5 GB/s target"
  else
    echo "❌ FAIL: Bandwidth not improved over baseline"
  fi
else
  echo "⚠️  WARNING: Could not extract bandwidth from logs"
fi

