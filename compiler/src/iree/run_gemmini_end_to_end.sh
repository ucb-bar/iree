#!/bin/bash
set -e  # Exit on error

if [ -z "$1" ]; then
  echo "Usage: $0 <input_file.mlir>"
  exit 1
fi

SRC_FILE="$1"
FILENAME=$(basename -- "$SRC_FILE")
BASENAME="${FILENAME%.*}"

# Your local IREE build directory
BUILD_DIR="$(pwd)/build"

# Where to dump vmfb + IR + phases
OUTPUT_DIR="/tmp/iree_gemmini/${BASENAME}"
OUTPUT_VMFB="${OUTPUT_DIR}/${BASENAME}.vmfb"
IR_LOG="${OUTPUT_DIR}/mlir_ir_dump.log"

IREE_COMPILE="${BUILD_DIR}/tools/iree-compile"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/phases"

echo "========================================================"
echo "Step 1: Compiling End-to-End (${SRC_FILE}) with Gemmini lowering"
echo "  BUILD_DIR=${BUILD_DIR}"
echo "========================================================"

# Allow the command to fail at the *link* step (missing __mulsf3 / __addsf3),
# but we still want the IR dump, so we don't `set -e` this one.
set +e
"${IREE_COMPILE}" \
  "${SRC_FILE}" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-enable-gemmini-linalg-lowering \
  --mlir-print-ir-after-all \
  --mlir-disable-threading \
  --iree-hal-dump-executable-intermediates-to="${OUTPUT_DIR}" \
  --dump-compilation-phases-to="${OUTPUT_DIR}/phases" \
  -o "${OUTPUT_VMFB}" 2> "${IR_LOG}"
COMPILE_STATUS=$?
set -e

echo "  [INFO] iree-compile exit status: ${COMPILE_STATUS}"
echo "  [OK] IR Dump saved to: ${IR_LOG}"
echo "  [OK] Phases saved to: ${OUTPUT_DIR}/phases/"
echo

echo "========================================================"
echo "Step 2: Pass + Gemmini sanity checks"
echo "========================================================"

echo "[1] Where does convert-linalg-to-gemmini run?"
grep -n "convert-linalg-to-gemmini" "${IR_LOG}" | head || echo "  (no hits)"

echo
echo "[2] Any gemmini.* ops in the IR?"
grep -n "gemmini\." "${IR_LOG}" | head || echo "  (no gemmini ops found)"

echo
echo "========================================================"
echo "Done."
echo "========================================================"