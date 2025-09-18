#!/bin/bash
# TCPX P2P Transport - Build Verification Script

set -e  # Exit on any error

echo "=== TCPX P2P Transport Build Verification ==="
echo ""

# Check if we're in the right directory
if [[ ! -f "tcpx_interface.h" || ! -f "tcpx_impl.cc" ]]; then
    echo "❌ Error: Please run this script from the p2p/tcpx directory"
    exit 1
fi

echo "📁 Checking file structure..."
REQUIRED_FILES=(
    "tcpx_interface.h"
    "tcpx_impl.cc"
    "Makefile"
    "README.md"
    "tests/test_device_discovery.cc"
    "tests/test_connection.cc"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $file"
    else
        echo "  ❌ Missing: $file"
        exit 1
    fi
done

echo ""
echo "🔧 Building tests..."

# Clean previous builds
make clean > /dev/null 2>&1 || true

# Build core tests
echo "  Building test_device_discovery..."
if make test_device_discovery > build.log 2>&1; then
    echo "  ✅ test_device_discovery built successfully"
else
    echo "  ❌ test_device_discovery build failed"
    echo "Build log:"
    cat build.log
    exit 1
fi

echo "  Building test_connection..."
if make test_connection >> build.log 2>&1; then
    echo "  ✅ test_connection built successfully"
else
    echo "  ❌ test_connection build failed"
    echo "Build log:"
    cat build.log
    exit 1
fi

echo ""
echo "🧪 Running automated tests..."

# Test device discovery
echo "  Running device discovery test..."
if ./tests/test_device_discovery > test.log 2>&1; then
    echo "  ✅ Device discovery test passed"
    # Show key results
    grep -E "(SUCCESS|Found.*devices)" test.log | head -2
else
    echo "  ❌ Device discovery test failed"
    echo "Test log:"
    cat test.log
    exit 1
fi

echo ""
echo "📋 Build verification summary:"
echo "  ✅ All required files present"
echo "  ✅ Core tests compile successfully"
echo "  ✅ Device discovery test passes"
echo ""
echo "🎯 Ready for PR submission!"
echo ""
echo "📝 Next steps:"
echo "  1. Test connection between two nodes:"
echo "     Node 1: ./tests/test_connection server"
echo "     Node 2: ./tests/test_connection client <node1_ip>"
echo ""
echo "  2. If connection test passes, the implementation is ready"
echo ""

# Clean up log files
rm -f build.log test.log

echo "✨ Verification complete!"
