#!/bin/bash
# Compilation script for concurrent scheduling models
# This script compiles all MLIR models to VMFB bytecode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compiling Concurrent Scheduling Models ==="
echo ""

# Check if iree-compile is available
if ! command -v iree-compile &> /dev/null; then
    echo "ERROR: iree-compile not found in PATH"
    echo ""
    echo "Please build IREE first and add the tools to your PATH:"
    echo "  cd /path/to/iree"
    echo "  cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    echo "  cmake --build build"
    echo "  export PATH=\$PATH:/path/to/iree/build/tools"
    exit 1
fi

echo "Found iree-compile: $(which iree-compile)"
echo ""

# Compilation flags
TARGET_DEVICE="local"
TARGET_BACKEND="llvm-cpu"
ASYNC_MODEL="async-external"

echo "=== Compiling Individual Models (for Oracle Scheduler) ==="
echo ""

# Compile Model A
echo "Compiling Model A (Convolutional Feature Extractor)..."
iree-compile \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_a_conv.mlir \
    -o=model_a.vmfb
echo "✓ model_a.vmfb created"

# Compile Model B
echo "Compiling Model B (Dense Classifier)..."
iree-compile \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_b_dense.mlir \
    -o=model_b.vmfb
echo "✓ model_b.vmfb created"

# Compile Model C
echo "Compiling Model C (Residual Processor)..."
iree-compile \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_c_residual.mlir \
    -o=model_c.vmfb
echo "✓ model_c.vmfb created"

echo ""
echo "=== Compiling Async Pipeline (for Vanilla Scheduler) ==="
echo ""

# Compile individual models with async support for vanilla pipeline
echo "Compiling Model A with async support..."
iree-compile \
    --iree-execution-model=${ASYNC_MODEL} \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_a_conv.mlir \
    -o=model_a_async.vmfb
echo "✓ model_a_async.vmfb created"

echo "Compiling Model B with async support..."
iree-compile \
    --iree-execution-model=${ASYNC_MODEL} \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_b_dense.mlir \
    -o=model_b_async.vmfb
echo "✓ model_b_async.vmfb created"

echo "Compiling Model C with async support..."
iree-compile \
    --iree-execution-model=${ASYNC_MODEL} \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    model_c_residual.mlir \
    -o=model_c_async.vmfb
echo "✓ model_c_async.vmfb created"

# Compile the pipeline orchestrator
echo "Compiling Pipeline Orchestrator..."
iree-compile \
    --iree-execution-model=${ASYNC_MODEL} \
    --iree-hal-target-device=${TARGET_DEVICE} \
    --iree-hal-local-target-device-backends=${TARGET_BACKEND} \
    pipeline_vanilla_async.mlir \
    -o=pipeline.vmfb
echo "✓ pipeline.vmfb created"

echo ""
echo "=== Compilation Complete ==="
echo ""
echo "Generated files:"
echo "  Oracle scheduler:"
echo "    - model_a.vmfb (for oracle execution)"
echo "    - model_b.vmfb (for oracle execution)"
echo "    - model_c.vmfb (for oracle execution)"
echo ""
echo "  Vanilla scheduler:"
echo "    - model_a_async.vmfb (with async support)"
echo "    - model_b_async.vmfb (with async support)"
echo "    - model_c_async.vmfb (with async support)"
echo "    - pipeline.vmfb (orchestrator)"
echo ""
echo "You can now run the examples:"
echo ""
echo "  Oracle scheduling:"
echo "    ./concurrent_scheduling_oracle model_a.vmfb model_b.vmfb model_c.vmfb 10"
echo ""
echo "  Vanilla IREE scheduling:"
echo "    ./concurrent_scheduling_vanilla pipeline.vmfb 10"
echo ""
