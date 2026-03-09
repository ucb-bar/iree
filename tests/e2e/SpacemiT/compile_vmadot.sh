#!/bin/bash
set -e # Exit on error

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file.mlir>"
    exit 1
fi

SRC_FILE="$1"
FILENAME=$(basename -- "$SRC_FILE")
BASENAME="${FILENAME%.*}"
BUILD_DIR="/scratch2/agustin/merlin/build/vanilla/host/debug/iree-spacemit-3.10.0.dev"
OUTPUT_DIR="/scratch2/agustin/merlin/third_party/iree_bar/tests/e2e/SpacemiT/tmp/${BASENAME}"
OUTPUT_VMFB="${OUTPUT_DIR}/${BASENAME}.vmfb"
IR_LOG="${OUTPUT_DIR}/mlir_ir_dump.log"

# Clean up previous dumps
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/phases"

echo "========================================================"
echo "Step 1: Compiling End-to-End (${SRC_FILE})"
echo "========================================================"

# TODO: Check this PR update for next iREE version. It should solve the conv issue: https://github.com/iree-org/iree/pull/23278
# We redirect stderr (2>) to IR_LOG because --mlir-print-ir-after-all prints to stderr.
${BUILD_DIR}/install/bin/iree-compile \
    "${SRC_FILE}" \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot" \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-opt-data-tiling=true \
    --iree-llvmcpu-enable-ukernels=none \
    --iree-codegen-mmt4d-use-intrinsics \
    --iree-llvmcpu-enable-vector-contract-custom-kernels=true \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-quantized-conv-to-conv, iree-preprocessing-convert-conv2d-to-img2col))" \
    --iree-hal-dump-executable-intermediates-to="${OUTPUT_DIR}" \
    --dump-compilation-phases-to="${OUTPUT_DIR}/phases" \
    --iree-hal-dump-executable-files-to="${OUTPUT_DIR}/files" \
    --mlir-print-ir-after-all \
    --mlir-disable-threading \
    -o "${OUTPUT_VMFB}" 2> "${IR_LOG}"

echo "  [OK] Generated: ${OUTPUT_VMFB}"
echo "  [OK] IR Dump saved to: ${IR_LOG}"
echo "  [OK] Phases saved to: ${OUTPUT_DIR}/phases/"

echo "========================================================"
echo "Step 2: Verifying Assembly (Looking for vmadot)"
echo "========================================================"
BINARY_FILE=$(find "${OUTPUT_DIR}" \( -name "*.so" -o -name "*.elf" -o -name "*.o" \) | head -n 1)

if [ -z "$BINARY_FILE" ]; then
    echo "  [FAIL] No binary file found."
    exit 1
fi

${BUILD_DIR}/llvm-project/bin/llvm-objdump \
    -d \
    --mattr=+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot \
    "$BINARY_FILE" | grep -E "vmadot" -C 2 || {
        echo "  [FAIL] vmadot / vfmadot NOT found in the assembly!"
        echo "  --> Check ${IR_LOG} to see what happened to the vector.contract"
        exit 1
    }

echo "  [SUCCESS] SpacemiT IME instruction found in assembly!"
echo "========================================================"
