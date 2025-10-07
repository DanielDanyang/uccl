#!/bin/bash
# Verify NIC distribution from orchestrator logs
# This script analyzes the log files to confirm:
# 1. All 4 NICs (eth1, eth2, eth3, eth4) are being used
# 2. Channels are distributed evenly across NICs
# 3. No single NIC is saturated

set -e

LOG_FILE="${1:-}"

if [[ -z "$LOG_FILE" ]]; then
  # Find the most recent server log
  LOG_FILE=$(ls -t logs/singleproc_server_*.log 2>/dev/null | head -1)
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "Error: Log file not found: $LOG_FILE"
  echo "Usage: $0 [log_file]"
  exit 1
fi

echo "=========================================="
echo "NIC Distribution Analysis"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Extract channel assignments
echo "=== Channel → NIC Mapping ==="
grep "Channel.*→ netDev" "$LOG_FILE" | head -40

echo ""
echo "=== NIC Usage Summary ==="

# Count channels per NIC (netDev)
for netdev in 0 1 2 3; do
  count=$(grep "Channel.*→ netDev $netdev" "$LOG_FILE" | wc -l)
  nic_name=$(grep "Channel.*→ netDev $netdev" "$LOG_FILE" | head -1 | sed -n 's/.*(\(eth[0-9]\).*/\1/p')
  if [[ -z "$nic_name" ]]; then
    nic_name="unknown"
  fi
  echo "netDev $netdev ($nic_name): $count channels"
done

echo ""
echo "=== GPU → NIC Distribution ==="

# Show distribution per GPU
for gpu in {0..7}; do
  echo -n "GPU $gpu: "
  grep "\[GPU $gpu\]" "$LOG_FILE" -A 50 | grep "Channel.*→ netDev" | \
    sed 's/.*netDev \([0-9]\).*/\1/' | tr '\n' ',' | sed 's/,$/\n/'
done

echo ""
echo "=== Accept Status ==="

# Check if all GPUs completed accept
for gpu in {0..7}; do
  if grep -q "\[GPU $gpu\] Accepted [0-9]* connections" "$LOG_FILE"; then
    channels=$(grep "\[GPU $gpu\] Accepted" "$LOG_FILE" | sed 's/.*Accepted \([0-9]*\).*/\1/')
    echo "✅ GPU $gpu: Accepted $channels connections"
  else
    echo "❌ GPU $gpu: Accept FAILED or incomplete"
  fi
done

echo ""
echo "=== Error Summary ==="

# Check for errors
error_count=$(grep -c "\[ERROR\]" "$LOG_FILE" || true)
failed_count=$(grep -c "Failed to accept" "$LOG_FILE" || true)

if [[ $error_count -eq 0 && $failed_count -eq 0 ]]; then
  echo "✅ No errors detected"
else
  echo "❌ Errors found:"
  echo "   [ERROR] lines: $error_count"
  echo "   Failed accepts: $failed_count"
  echo ""
  echo "Error details:"
  grep -E "\[ERROR\]|Failed to accept" "$LOG_FILE" || true
fi

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="

