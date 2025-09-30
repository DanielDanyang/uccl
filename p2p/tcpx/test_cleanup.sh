#!/bin/bash
# Test script to verify the cleanup changes

set -e

echo "=== TCPX Cleanup Verification ==="
echo ""

echo "Step 1: Clean previous build artifacts..."
cd /mnt/user_storage/uccl/p2p/tcpx
make clean

echo ""
echo "Step 2: Verify removed files are gone..."
if [ -f "rx/rx_cmsg_parser.h" ]; then
    echo "ERROR: rx_cmsg_parser.h still exists!"
    exit 1
fi
if [ -f "rx/rx_cmsg_parser.cc" ]; then
    echo "ERROR: rx_cmsg_parser.cc still exists!"
    exit 1
fi
if [ -f "rx/rx_descriptor.cc" ]; then
    echo "ERROR: rx_descriptor.cc still exists!"
    exit 1
fi
echo "✓ Removed files confirmed deleted"

echo ""
echo "Step 3: Verify rx_descriptor.h is simplified..."
if grep -q "class DescriptorBuilder" rx/rx_descriptor.h; then
    echo "ERROR: DescriptorBuilder class still in rx_descriptor.h!"
    exit 1
fi
if grep -q "using UnpackDescriptor = tcpx::plugin::loadMeta" rx/rx_descriptor.h; then
    echo "✓ rx_descriptor.h uses loadMeta alias"
else
    echo "ERROR: rx_descriptor.h doesn't use loadMeta alias!"
    exit 1
fi

echo ""
echo "Step 4: Build core components..."
make core

echo ""
echo "Step 5: Build test_tcpx_transfer..."
make test_tcpx_transfer

echo ""
echo "Step 6: Verify executable exists..."
if [ -f "tests/test_tcpx_transfer" ]; then
    echo "✓ test_tcpx_transfer built successfully"
else
    echo "ERROR: test_tcpx_transfer not found!"
    exit 1
fi

echo ""
echo "=== All verification steps passed! ==="
echo ""
echo "Summary of changes:"
echo "  - Removed: rx/rx_cmsg_parser.h, rx/rx_cmsg_parser.cc"
echo "  - Removed: rx/rx_descriptor.cc"
echo "  - Simplified: rx/rx_descriptor.h (now header-only)"
echo "  - Updated: Makefile (removed RX_OBJS)"
echo "  - Updated: test_tcpx_transfer.cc (uses buildDescriptorBlock)"
echo ""
echo "You can now test with:"
echo "  export UCCL_TCPX_UNPACK_IMPL=d2d"
echo "  ./tests/test_tcpx_transfer server"
echo "  ./tests/test_tcpx_transfer client <server_ip>"

