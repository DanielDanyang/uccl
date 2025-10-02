#!/usr/bin/env python3
"""
Parse bench_p2p logs and generate performance table.
Usage: python3 parse_bench_logs.py <log_dir> [--skip-warmup N]
"""
import re
import sys
import os
from pathlib import Path

def parse_log(log_path, skip_warmup=1):
    """Parse a single bench log and extract steady-state BW."""
    with open(log_path) as f:
        lines = f.readlines()
    
    # Extract config
    size_bytes = None
    iters = None
    chunk_bytes = None
    nsocks = None
    nthreads = None
    ifaces = None
    impl = None
    
    for line in lines:
        if m := re.search(r'Size\s+:\s+(\d+)\s+bytes', line):
            size_bytes = int(m.group(1))
        elif m := re.search(r'Iters\s+:\s+(\d+)', line):
            iters = int(m.group(1))
        elif m := re.search(r'Chunk bytes\s+:\s+(\d+)', line):
            chunk_bytes = int(m.group(1))
        elif m := re.search(r'nsocks/thrds:\s+(\d+)/(\d+)', line):
            nsocks, nthreads = int(m.group(1)), int(m.group(2))
        elif m := re.search(r'Ifaces\s+:\s+(.+)', line):
            ifaces = m.group(1).strip()
        elif m := re.search(r'Impl\s+:\s+(\w+)', line):
            impl = m.group(1)
    
    # Extract per-iteration times
    iter_times = []
    for line in lines:
        if m := re.search(r'\[PERF\] Iter (\d+) time=([0-9.]+)ms', line):
            iter_num = int(m.group(1))
            time_ms = float(m.group(2))
            iter_times.append((iter_num, time_ms))
    
    if not iter_times or size_bytes is None:
        return None
    
    # Compute steady-state (skip warmup)
    steady_times = [t for i, t in iter_times if i >= skip_warmup]
    if not steady_times:
        return None
    
    avg_ms = sum(steady_times) / len(steady_times)
    size_gb = size_bytes / (1024**3)
    bw_gbps = size_gb / (avg_ms / 1000.0)
    
    return {
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024**2),
        'iters': iters,
        'chunk_bytes': chunk_bytes,
        'nsocks': nsocks,
        'nthreads': nthreads,
        'ifaces': ifaces,
        'impl': impl,
        'steady_avg_ms': avg_ms,
        'steady_bw_gbps': bw_gbps,
        'log_file': log_path.name
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_bench_logs.py <log_dir> [--skip-warmup N]")
        sys.exit(1)
    
    log_dir = Path(sys.argv[1])
    skip_warmup = 1
    
    for arg in sys.argv[2:]:
        if arg.startswith('--skip-warmup='):
            skip_warmup = int(arg.split('=')[1])
    
    # Find all bench_client logs
    client_logs = sorted(log_dir.glob('bench_client_*.log'))
    
    if not client_logs:
        print(f"No bench_client_*.log files found in {log_dir}")
        sys.exit(1)
    
    results = []
    for log_path in client_logs:
        result = parse_log(log_path, skip_warmup)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results parsed")
        sys.exit(1)
    
    # Sort by size
    results.sort(key=lambda x: x['size_bytes'])
    
    # Print CSV
    print("size_bytes,size_mb,iters,chunk_bytes,nsocks,nthreads,ifaces,impl,steady_avg_ms,steady_bw_gbps,log_file")
    for r in results:
        print(f"{r['size_bytes']},{r['size_mb']:.2f},{r['iters']},{r['chunk_bytes']},"
              f"{r['nsocks']},{r['nthreads']},\"{r['ifaces']}\",{r['impl']},"
              f"{r['steady_avg_ms']:.6f},{r['steady_bw_gbps']:.3f},{r['log_file']}")
    
    print("\n# TCPX P2P Performance Table\n")
    print("| Size (Bytes) | Size (MB) | Iters | Steady Avg (ms) | BW (GB/s) | Log File |")
    print("| ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in results:
        print(f"| {r['size_bytes']:,} | {r['size_mb']:.2f} | {r['iters']} | "
              f"{r['steady_avg_ms']:.6f} | {r['steady_bw_gbps']:.3f} | {r['log_file']} |")

if __name__ == '__main__':
    main()

