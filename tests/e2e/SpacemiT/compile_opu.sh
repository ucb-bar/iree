#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_file.mlir>"
  exit 1
fi

SRC_FILE="$1"
FILENAME=$(basename -- "$SRC_FILE")
BASENAME="${FILENAME%.*}"

# Your current build tree.
BUILD_DIR="/scratch2/agustin/merlin/build/host-vanilla-release"

# Output folder for this OPU/ZVL128 run.
OUTPUT_DIR="/scratch2/agustin/merlin/third_party/iree_bar/tests/e2e/OPU/tmp/${BASENAME}_xopu_zvl128"
OUTPUT_VMFB="${OUTPUT_DIR}/${BASENAME}.vmfb"
IR_LOG="${OUTPUT_DIR}/mlir_ir_dump.log"
ASM_LOG="${OUTPUT_DIR}/assembly_full.txt"
ASM_HITS="${OUTPUT_DIR}/assembly_opu_hits.txt"

# IMPORTANT:
# Set this to the exact feature string you defined in RISCVFeatures.td.
# Most likely it is +xopu.
# If your tree uses a different spelling (for example +xucbbar), either:
#   XOPU_FEATURE=+xucbbar ./compile_opu_zvl128.sh file.mlir
# or edit the default below.
XOPU_FEATURE="${XOPU_FEATURE:-+xopu}"

BASE_FEATURES="+m,+a,+f,+d,+c,+v,+zvl128b"
TARGET_FEATURES="${BASE_FEATURES},${XOPU_FEATURE}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/phases"
mkdir -p "${OUTPUT_DIR}/files"

echo "========================================================"
echo "Step 1: Compiling OPU path (${SRC_FILE})"
echo "  TARGET_FEATURES=${TARGET_FEATURES}"
echo "========================================================"

"${BUILD_DIR}/install/bin/iree-compile" \
  "${SRC_FILE}" \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu-features="${TARGET_FEATURES}" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-vector-width-in-bytes=16 \
  --iree-opt-data-tiling=true \
  --iree-llvmcpu-enable-ukernels=none \
  --iree-llvmcpu-link-ukernel-bitcode=false \
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
echo "  [OK] IR dump:   ${IR_LOG}"
echo "  [OK] Phases:    ${OUTPUT_DIR}/phases"
echo "  [OK] Files:     ${OUTPUT_DIR}/files"

echo "========================================================"
echo "Step 2: Verifying final assembly (Looking for OPU)"
echo "========================================================"

BINARY_FILE=$(find "${OUTPUT_DIR}" \( -name "*.so" -o -name "*.elf" -o -name "*.o" \) | head -n 1)

if [ -z "${BINARY_FILE}" ]; then
  echo "  [FAIL] No binary file found."
  exit 1
fi

echo "  Using binary: ${BINARY_FILE}"

"${BUILD_DIR}/llvm-project/bin/llvm-objdump" \
  -d \
  --mattr="${TARGET_FEATURES}" \
  "${BINARY_FILE}" > "${ASM_LOG}"

grep -E "opu\.opmvinbcast|opu\.vmv\.rv|opu\.vmv\.vr|opu\.vopacc" -C 2 "${ASM_LOG}" > "${ASM_HITS}" || true

if [ -s "${ASM_HITS}" ]; then
  echo "  [SUCCESS] OPU instructions found in final assembly:"
  cat "${ASM_HITS}"
  echo "========================================================"
  exit 0
fi

echo "  [WARN] No final OPU asm mnemonics found."
echo "========================================================"
echo "Step 3: Checking IR for OPU intrinsics"
echo "========================================================"

grep -E "llvm\.riscv\.opu\.(bcast|vmv\.rv|vmv\.vr|vopacc)" -n "${IR_LOG}" | tee "${OUTPUT_DIR}/ir_opu_hits.txt" || true

if [ -s "${OUTPUT_DIR}/ir_opu_hits.txt" ]; then
  echo "  [INFO] OPU intrinsics are present in IR, but did not survive to final asm."
  echo "  --> This usually means the backend selection/lowering is still not finishing as expected."
  exit 2
fi

echo "  [FAIL] No OPU intrinsics found in IR either."
echo "  --> The custom-kernel match likely did not trigger."
exit 1