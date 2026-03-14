// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

#opu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl128b,+xopu",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_i8_i32_opu(%lhs: tensor<128x256xi8>, %rhs: tensor<256x128xi8>) -> tensor<128x128xi32> attributes {hal.executable.target = #opu_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi8>, tensor<256x128xi8>) outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [16, 16, 0]
// CHECK-LABEL: func.func @matmul_i8_i32_opu
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

#opu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl128b,+xopu",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_i8_i32_opu_narrow_m(%lhs: tensor<8x256xi8>, %rhs: tensor<256x128xi8>) -> tensor<8x128xi32> attributes {hal.executable.target = #opu_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<8x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<8x128xi32>) -> tensor<8x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<8x256xi8>, tensor<256x128xi8>) outs(%fill : tensor<8x128xi32>) -> tensor<8x128xi32>
  return %res : tensor<8x128xi32>
}

// CHECK-DAG:     #[[CONFIG_NARROW:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [8, 16, 0]
// CHECK-LABEL: func.func @matmul_i8_i32_opu_narrow_m
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG_NARROW]]

// -----

#opu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl128b,+xopu",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_fp8_f16_opu(%lhs: tensor<128x256xf8E4M3FN>, %rhs: tensor<256x128xf8E4M3FN>) -> tensor<128x128xf16> attributes {hal.executable.target = #opu_target} {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x128xf16>
  %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) outs(%fill : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %res : tensor<128x128xf16>
}

// CHECK-DAG:     #[[CONFIG_FP8:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [16, 16, 0], vector_reduction = [0, 0, 8]
// CHECK-LABEL: func.func @matmul_fp8_f16_opu
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG_FP8]]
