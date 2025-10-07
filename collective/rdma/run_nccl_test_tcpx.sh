# !/bin/bash

source ../../scripts/shared.sh

# Usage ./run_nccl_test.sh [nccl] [# of Nodes] [# of GPUs per process] [allreduce/alltoall: 0/1] [procs per node] [collect_diagnostics: 0/1]

TEST=${1:-nccl}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_PROC=${3:-8}
PROG_OPTION=${4:-0}
PROCS_PER_NODE=${5:-1}
COLLECT_DIAGNOSTICS=${6:-0}
HOSTFILE="${UCCL_HOME}/scripts/node_ips/tcpx.txt"

# Create diagnostics directory if collecting data
if [ "$COLLECT_DIAGNOSTICS" -eq 1 ]; then
    DIAG_DIR="${UCCL_HOME}/diagnostics/nccl_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${DIAG_DIR}"
    echo "Diagnostics will be saved to: ${DIAG_DIR}"
fi

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

if [ "$PROG_OPTION" -eq 0 ]; then
    PROG_NAME=all_reduce_perf
elif [ "$PROG_OPTION" -eq 1 ]; then
    PROG_NAME=alltoall_perf
else
    PROG_NAME=sendrecv_perf
fi

if [ "$TEST" = "nccl" ]; then
    echo "Running NCCL test"
else
    echo "Unsupport benchmark type."
    exit 1
fi

CUDA_LIB_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
echo "Running test: ${PROG_NAME}, ${TEST}, ${NUM_PROCS} nodes, ${NUM_GPUS_PER_PROC} GPUs per process, $((NUM_PROCS * NUM_GPUS_PER_PROC)) GPUs in total."

echo $NUM_PROCS
echo $PROCS_PER_NODE

# ============================================================================
# PRE-TEST DIAGNOSTICS COLLECTION
# ============================================================================
if [ "$COLLECT_DIAGNOSTICS" -eq 1 ]; then
    echo "=== Collecting pre-test diagnostics ==="

    # 1. IRQ snapshot
    echo "Collecting IRQ snapshot..."
    cat /proc/interrupts > "${DIAG_DIR}/irq_snapshot_before_$(hostname).txt"

    # 2. IRQ affinity for gVNIC
    echo "Collecting IRQ affinity..."
    {
        echo "=== gVNIC IRQ Affinity ==="
        for irq in $(grep gve /proc/interrupts | awk '{print $1}' | tr -d ':'); do
            affinity=$(cat /proc/irq/$irq/smp_affinity 2>/dev/null || echo 'N/A')
            affinity_list=$(cat /proc/irq/$irq/smp_affinity_list 2>/dev/null || echo 'N/A')
            echo "IRQ $irq: mask=$affinity cores=$affinity_list"
        done
    } > "${DIAG_DIR}/irq_affinity_before_$(hostname).txt"

    # 3. NIC-to-IRQ mapping and stats
    echo "Collecting NIC information..."
    {
        for nic in eth1 eth2 eth3 eth4; do
            echo "=== $nic ==="
            echo "Driver info:"
            ethtool -i $nic 2>/dev/null || echo "  N/A"
            echo ""
            echo "PCI and NUMA:"
            pci_addr=$(ethtool -i $nic 2>/dev/null | grep bus-info | awk '{print $2}')
            if [ -n "$pci_addr" ]; then
                numa_node=$(cat /sys/class/net/$nic/device/numa_node 2>/dev/null || echo "N/A")
                echo "  PCI: $pci_addr"
                echo "  NUMA: $numa_node"
            fi
            echo ""
            echo "Statistics (selected):"
            ethtool -S $nic 2>/dev/null | grep -E "rx_packets|tx_packets|rx_bytes|tx_bytes|rx_devmem" | head -20
            echo ""
        done
    } > "${DIAG_DIR}/nic_info_before_$(hostname).txt"

    # 4. CPU topology
    echo "Collecting CPU topology..."
    lscpu > "${DIAG_DIR}/cpu_topology_$(hostname).txt"

    # 5. NUMA topology
    echo "Collecting NUMA topology..."
    numactl --hardware > "${DIAG_DIR}/numa_topology_$(hostname).txt" 2>&1

    # 6. Current environment variables
    echo "Collecting environment variables..."
    env | grep -E "NCCL|CUDA|TCPX|UCCL" | sort > "${DIAG_DIR}/env_vars_$(hostname).txt"

    echo "Pre-test diagnostics collection complete."
    echo ""
fi

# ============================================================================
# BACKGROUND MONITORING (if diagnostics enabled)
# ============================================================================
if [ "$COLLECT_DIAGNOSTICS" -eq 1 ]; then
    echo "Starting background monitoring..."

    # Monitor CPU usage
    mpstat -P ALL 1 > "${DIAG_DIR}/cpu_usage_during_test_$(hostname).txt" 2>&1 &
    MPSTAT_PID=$!

    # Monitor IRQ counts (sample every 5 seconds)
    {
        while true; do
            echo "=== $(date +%H:%M:%S) ==="
            cat /proc/interrupts | grep gve
            sleep 5
        done
    } > "${DIAG_DIR}/irq_monitoring_$(hostname).txt" 2>&1 &
    IRQ_MON_PID=$!

    echo "Background monitoring started (mpstat PID: $MPSTAT_PID, IRQ monitor PID: $IRQ_MON_PID)"
    echo ""
fi

# ============================================================================
# RUN NCCL TEST
# ============================================================================
# adapted from https://github.com/skypilot-org/skypilot/blob/master/examples/gcp_gpu_direct_tcpx/nccl_tcpx_gcpvm_h100.yaml

# Save output to file if collecting diagnostics
if [ "$COLLECT_DIAGNOSTICS" -eq 1 ]; then
    NCCL_OUTPUT="${DIAG_DIR}/nccl_test_output.log"
    echo "NCCL output will be saved to: ${NCCL_OUTPUT}"
    echo "Running NCCL test..."
else
    NCCL_OUTPUT="/dev/stdout"
fi

mpirun --allow-run-as-root -np ${NUM_PROCS} -N ${PROCS_PER_NODE} \
    -hostfile ${HOSTFILE} --map-by ppr:${PROCS_PER_NODE}:node \
    --mca btl tcp,self \
    --mca btl_tcp_if_include eth0 \
    --mca plm_rsh_args "-p 2222" \
    -x PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH" \
    -x LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH" \
    -x NCCL_IGNORE_CPU_AFFINITY=1 \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=Simple \
    -x NCCL_MAX_NCHANNELS=8 \
    -x NCCL_MIN_NCHANNELS=8 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_CROSS_NIC=0 \
    -x NCCL_NSOCKS_PERTHREAD=4 \
    -x NCCL_SOCKET_NTHREADS=1 \
    -x NCCL_DYNAMIC_CHUNK_SIZE=524288 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_P2P_PCI_CHUNKSIZE=524288 \
    -x NCCL_P2P_NVL_CHUNKSIZE=1048576 \
    -x NCCL_BUFFSIZE=8388608 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4 \
    -x NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0 \
    -x NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177" \
    -x NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191" \
    -x NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000 \
    -x NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx" \
    -x NCCL_GPUDIRECTTCPX_FORCE_ACK=0 \
    -x NCCL_NET_GDR_LEVEL=PIX \
    -x NCCL_P2P_PXN_LEVEL=0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=ENV \
    ${UCCL_HOME}/thirdparty/nccl-tests/build/${PROG_NAME} -c 0 \
    -b 1K -e 1G \
    -f 2 -w 50 -n 50 \
    -g 1 -t ${NUM_GPUS_PER_PROC} 2>&1 | tee "${NCCL_OUTPUT}"

NCCL_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# POST-TEST DIAGNOSTICS COLLECTION
# ============================================================================
if [ "$COLLECT_DIAGNOSTICS" -eq 1 ]; then
    echo ""
    echo "=== Collecting post-test diagnostics ==="

    # Stop background monitoring
    if [ -n "$MPSTAT_PID" ]; then
        kill $MPSTAT_PID 2>/dev/null
        echo "Stopped mpstat monitoring"
    fi
    if [ -n "$IRQ_MON_PID" ]; then
        kill $IRQ_MON_PID 2>/dev/null
        echo "Stopped IRQ monitoring"
    fi

    # 1. IRQ snapshot after test
    echo "Collecting post-test IRQ snapshot..."
    cat /proc/interrupts > "${DIAG_DIR}/irq_snapshot_after_$(hostname).txt"

    # 2. Calculate IRQ delta
    echo "Calculating IRQ delta..."
    diff "${DIAG_DIR}/irq_snapshot_before_$(hostname).txt" \
         "${DIAG_DIR}/irq_snapshot_after_$(hostname).txt" \
         > "${DIAG_DIR}/irq_delta_$(hostname).txt" 2>&1 || true

    # 3. NIC statistics after test
    echo "Collecting post-test NIC statistics..."
    {
        for nic in eth1 eth2 eth3 eth4; do
            echo "=== $nic ==="
            ethtool -S $nic 2>/dev/null | grep -E "rx_packets|tx_packets|rx_bytes|tx_bytes|rx_devmem" | head -20
            echo ""
        done
    } > "${DIAG_DIR}/nic_stats_after_$(hostname).txt"

    # 4. Extract key metrics from NCCL output
    echo "Extracting NCCL metrics..."
    {
        echo "=== NCCL Performance Summary ==="
        grep -E "Avg bus bandwidth|Out of bounds|size.*time.*algbw.*busbw" "${NCCL_OUTPUT}" | tail -20
        echo ""
        echo "=== NCCL Thread Affinity ==="
        grep -i "thread.*running on.*cpu" "${NCCL_OUTPUT}" | head -50
        echo ""
        echo "=== NCCL Channel Setup ==="
        grep -i "channel" "${NCCL_OUTPUT}" | head -50
        echo ""
        echo "=== NCCL Errors/Warnings ==="
        grep -iE "error|warn|fatal" "${NCCL_OUTPUT}" | head -50
    } > "${DIAG_DIR}/nccl_metrics_summary.txt"

    # 5. Create summary report
    echo "Creating summary report..."
    {
        echo "=== NCCL Test Diagnostics Summary ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Test: ${PROG_NAME}"
        echo "Processes: ${NUM_PROCS}"
        echo "GPUs per process: ${NUM_GPUS_PER_PROC}"
        echo "Exit code: ${NCCL_EXIT_CODE}"
        echo ""
        echo "=== Binding Configuration ==="
        echo "TX_BINDINGS: ${NCCL_GPUDIRECTTCPX_TX_BINDINGS:-Not set}"
        echo "RX_BINDINGS: ${NCCL_GPUDIRECTTCPX_RX_BINDINGS:-Not set}"
        echo ""
        echo "=== Files Collected ==="
        ls -lh "${DIAG_DIR}"
    } > "${DIAG_DIR}/SUMMARY.txt"

    echo ""
    echo "=========================================="
    echo "Diagnostics collection complete!"
    echo "Results saved to: ${DIAG_DIR}"
    echo "=========================================="
    cat "${DIAG_DIR}/SUMMARY.txt"
fi

exit ${NCCL_EXIT_CODE:-0}

# -x NCCL_P2P_DISABLE=1 \
# -x LD_LIBRARY_PATH="/mnt/user_storage/uccl-yang/thirdparty/nccl/build/lib" \
# -x NCCL_NET_PLUGIN="/mnt/user_storage/tcpx-yang/build/libnccl-net.so" \
  