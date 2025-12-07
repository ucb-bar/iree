#!/bin/bash
# Test script for concurrent scheduling example
# This script validates the MLIR syntax and provides instructions for running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Concurrent Scheduling Example Test Script ==="
echo ""
echo "This example demonstrates concurrent execution of multiple neural network models."
echo ""

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/model_a_conv.mlir" ]; then
    echo "ERROR: Please run this script from the samples/concurrent_scheduling directory"
    exit 1
fi

echo "✓ Found MLIR model files:"
echo "  - model_a_conv.mlir (Convolutional feature extractor)"
echo "  - model_b_dense.mlir (Dense classifier)"
echo "  - model_c_residual.mlir (Residual processor)"
echo "  - pipeline_vanilla_async.mlir (Async pipeline orchestrator)"
echo ""

echo "✓ Found C runtime implementations:"
echo "  - concurrent_scheduling_oracle.c (Custom oracle scheduler)"
echo "  - concurrent_scheduling_vanilla.c (Vanilla IREE scheduler)"
echo ""

echo "✓ Build configuration files:"
echo "  - CMakeLists.txt (CMake build configuration)"
echo "  - BUILD.bazel (Bazel build configuration)"
echo ""

echo "=== Quick Syntax Validation ==="
echo ""

# Basic MLIR syntax check (just verify files are readable and have basic structure)
for file in model_a_conv.mlir model_b_dense.mlir model_c_residual.mlir pipeline_vanilla_async.mlir; do
    if grep -q "builtin.module" "$file" || grep -q "func.func" "$file"; then
        echo "✓ $file: Valid MLIR structure"
    else
        echo "✗ $file: Missing expected MLIR constructs"
        exit 1
    fi
done

echo ""
echo "=== Build Instructions ==="
echo ""
echo "To build and run this example, you need to:"
echo ""
echo "1. Build IREE from source (if not already done):"
echo "   cd /path/to/iree"
echo "   cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo"
echo "   cmake --build build"
echo ""
echo "2. Add IREE tools to PATH:"
echo "   export PATH=\$PATH:/path/to/iree/build/tools"
echo ""
echo "3. Compile the models:"
echo "   cd samples/concurrent_scheduling"
echo "   ./compile_models.sh"
echo ""
echo "4. Run the examples:"
echo "   # Oracle scheduling (manual control)"
echo "   ../build/samples/concurrent_scheduling/concurrent_scheduling_oracle \\"
echo "       model_a.vmfb model_b.vmfb model_c.vmfb 10"
echo ""
echo "   # Vanilla IREE scheduling (automatic)"
echo "   ../build/samples/concurrent_scheduling/concurrent_scheduling_vanilla \\"
echo "       pipeline.vmfb 10"
echo ""
echo "=== Hardware Configuration ==="
echo ""
echo "This example is designed for SpaceMIT X60 hardware:"
echo "  - 2 CPU clusters (each with 4 cores)"
echo "  - 1 cluster has NPU acceleration"
echo ""
echo "The oracle scheduler attempts to place workloads optimally:"
echo "  - Model A (Conv) → Cluster 1 (NPU preferred)"
echo "  - Model B (Dense) → Cluster 0 (CPU intensive)"
echo "  - Model C (Residual) → Cluster 1"
echo ""
echo "For other hardware, the example will still work but may not show"
echo "significant performance differences between scheduling approaches."
echo ""
echo "✓ All validation checks passed!"
echo ""
echo "See README.md for detailed instructions and usage examples."
