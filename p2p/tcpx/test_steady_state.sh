#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="logs/bench_client_20251001_232734.log"
SIZE=4096
SKIP_WARMUP=1

ITER_LINES=$(grep -E "\[PERF\] Iter [0-9]+ time=([0-9]+\.?[0-9]*)ms" -n "${LOG_FILE}" || true)
echo "Found iter lines: $(echo "$ITER_LINES" | wc -l)"

SS_AVG_MS=$(echo "${ITER_LINES}" | awk -v skip="${SKIP_WARMUP}" '
  {
    split($0,a,":");
    line=a[2];
    match(line, /Iter ([0-9]+) time=([0-9]+\.?[0-9]*)ms/, m);
    iter=m[1]+0; val=m[2]+0.0;
    if (iter>=skip) { sum+=val; n+=1; }
  }
  END { if (n>0) printf "%.6f", sum/n; else print ""; }
')

echo "SS_AVG_MS: ${SS_AVG_MS}"

SS_BW_GBPS=$(python3 -c "
size_gb = float(${SIZE})/(1024**3)
avg_ms = float(${SS_AVG_MS})
print(f'{size_gb / (avg_ms/1000.0):.3f}')
")

echo "SS_BW_GBPS: ${SS_BW_GBPS}"
echo "[STEADY-STATE] Avg time: ${SS_AVG_MS} ms, BW: ${SS_BW_GBPS} GB/s"

