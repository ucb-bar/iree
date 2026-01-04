#!/bin/bash

# --- Configuration ---
# Adjust paths if necessary
BUILD_DIR="/scratch2/agustin/merlin/build-bar-iree-host-deb-tracy"
RUNNER_BIN="$BUILD_DIR/samples/promise_devices_layer/multi-core-runner"
VMFB_FILE="samples/promise_devices_layer/multi_core_dispatch.vmfb"

# Ensure ASan libraries are found (optional, depending on your build)
CLANG_RT_DIR=$(${CONDA_PREFIX}/bin/clang -print-resource-dir)/lib/x86_64-conda-linux-gnu
export LD_LIBRARY_PATH="${CLANG_RT_DIR}:${LD_LIBRARY_PATH}"
# Uncomment line below if you need ASan preloading
# export LD_PRELOAD="${CLANG_RT_DIR}/libclang_rt.asan-x86_64.so"

echo "========================================================"
echo " Starting Multi-Core Runner in Background..."
echo "========================================================"

# 1. Run the program in the background
$RUNNER_BIN $VMFB_FILE main > /dev/null 2>&1 &
RUNNER_PID=$!

if [ -z "$RUNNER_PID" ]; then
    echo "❌ Failed to start process."
    exit 1
fi

echo "✅ Process started with PID: $RUNNER_PID"
echo "⏳ Waiting 1 second for worker threads to initialize..."
sleep 1

# Check if process is still alive
if ! ps -p $RUNNER_PID > /dev/null; then
   echo "❌ Process died prematurely. Did you remove the infinite loop from main.c?"
   exit 1
fi

echo "========================================================"
echo " 🔍 Capturing CPU Affinity Masks (taskset)"
echo "========================================================"

# 2. Capture taskset output
# Output format is usually: "pid <TID>'s current affinity mask: <mask>"
TASKSET_OUTPUT=$(taskset -a -p $RUNNER_PID)

echo "$TASKSET_OUTPUT"
echo "--------------------------------------------------------"

# 3. Analyze Results
# We look for specific hex masks at the end of the line
FOUND_MASK_1=0 # Binary 0001 (Core 0)
FOUND_MASK_2=0 # Binary 0010 (Core 1)
FOUND_MASK_3=0 # Binary 0011 (Core 0 & 1)

if echo "$TASKSET_OUTPUT" | grep -q "mask: 1$"; then FOUND_MASK_1=1; fi
if echo "$TASKSET_OUTPUT" | grep -q "mask: 2$"; then FOUND_MASK_2=1; fi
if echo "$TASKSET_OUTPUT" | grep -q "mask: 3$"; then FOUND_MASK_3=1; fi

echo "📊 Analysis:"

if [ $FOUND_MASK_1 -eq 1 ]; then
    echo "  ✅ Success: Found thread pinned to CORE 0 (Mask 1) -> Device A"
else
    echo "  ❌ Failure: Missing thread for Device A (Expected Mask 1)"
fi

if [ $FOUND_MASK_2 -eq 1 ]; then
    echo "  ✅ Success: Found thread pinned to CORE 1 (Mask 2) -> Device B"
else
    echo "  ❌ Failure: Missing thread for Device B (Expected Mask 2)"
fi

if [ $FOUND_MASK_3 -eq 1 ]; then
    echo "  ✅ Success: Found thread on CORES 0+1   (Mask 3) -> Device AB"
else
    echo "  ❌ Failure: Missing thread for Device AB (Expected Mask 3)"
fi

echo "========================================================"

# 4. Cleanup
echo "🧹 Killing process $RUNNER_PID..."
kill $RUNNER_PID