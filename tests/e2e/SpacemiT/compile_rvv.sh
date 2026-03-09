#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file.mlir>"
    exit 1
fi

SRC_FILE="$1"
FILENAME=$(basename -- "$SRC_FILE")
BASENAME="${FILENAME%.*}"
BUILD_DIR="/scratch2/agustin/merlin/build/vanilla/host/debug/iree-spacemit-3.10.0.dev"
OUTPUT_DIR="/scratch2/agustin/merlin/third_party/iree_bar/tests/e2e/SpacemiT/tmp/${BASENAME}_rvv"
OUTPUT_VMFB="${OUTPUT_DIR}/${BASENAME}.vmfb"
IR_LOG="${OUTPUT_DIR}/mlir_ir_dump.log"

# Tunables:
# - Lower LMUL caps often perform better on X60 for generic RVV i8 paths
#   because they reduce vector-pipeline occupancy for vrgather/vmacc chains.
# - Override at invocation time, e.g. RVV_LMUL_MAX=2 ./compile_rvv.sh ...
RVV_LMUL_MAX="${RVV_LMUL_MAX:-1}"
VECTOR_WIDTH_BYTES="${VECTOR_WIDTH_BYTES:-32}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/phases"

echo "========================================================"
echo "Step 1: Compiling Standard RVV (${SRC_FILE})"
echo "  RVV_LMUL_MAX=${RVV_LMUL_MAX}"
echo "  VECTOR_WIDTH_BYTES=${VECTOR_WIDTH_BYTES}"
echo "========================================================"

${BUILD_DIR}/install/bin/iree-compile \
    "${SRC_FILE}" \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b" \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-opt-data-tiling=true \
    --iree-codegen-mmt4d-use-intrinsics \
    --iree-llvmcpu-enable-ukernels=none \
    --iree-llvmcpu-enable-vector-contract-custom-kernels=true \
    --iree-llvmcpu-link-ukernel-bitcode=false \
    --iree-llvmcpu-skip-intermediate-roundings=true \
    --riscv-v-fixed-length-vector-lmul-max="${RVV_LMUL_MAX}" \
    --iree-llvmcpu-target-vector-width-in-bytes="${VECTOR_WIDTH_BYTES}" \
    --iree-llvmcpu-loop-vectorization=true \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-quantized-conv-to-conv, iree-preprocessing-convert-conv2d-to-img2col))" \
    --iree-hal-dump-executable-intermediates-to="${OUTPUT_DIR}" \
    --dump-compilation-phases-to="${OUTPUT_DIR}/phases" \
    --mlir-print-ir-after-all \
    --mlir-disable-threading \
    -o "${OUTPUT_VMFB}" 2> "${IR_LOG}"

echo "  [OK] Generated: ${OUTPUT_VMFB}"
echo "  [OK] IR Dump saved to: ${IR_LOG}"

echo "========================================================"
echo "Step 2: Verifying Assembly (Looking for Standard RVV)"
echo "========================================================"
BINARY_FILE=$(find "${OUTPUT_DIR}" \( -name "*.so" -o -name "*.elf" -o -name "*.o" \) | head -n 1)

if [ -z "$BINARY_FILE" ]; then
    echo "  [FAIL] No binary file found."
    exit 1
fi

echo "Checking for vector multiply-accumulate instructions..."
${BUILD_DIR}/llvm-project/bin/llvm-objdump \
    -d \
    --mattr=+m,+a,+f,+d,+c,+v,+zvl256b \
    "$BINARY_FILE" | grep -E "vwmacc|vmacc" -C 2 > "${OUTPUT_DIR}/assembly_snippet.txt"

if [ -s "${OUTPUT_DIR}/assembly_snippet.txt" ]; then
    echo "  [SUCCESS] Standard RVV math instructions found!"
    head -n 20 "${OUTPUT_DIR}/assembly_snippet.txt"
else
    echo "  [FAIL] No vector math instructions found."
    echo "  --> Check if the compiler fell back to scalar code."
    exit 1
fi

echo "========================================================"
