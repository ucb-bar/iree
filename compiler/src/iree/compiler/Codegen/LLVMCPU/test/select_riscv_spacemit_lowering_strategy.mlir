// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

// ============================================================================
// Test 1: Integer Matmul (i8 inputs -> i32 accumulation)
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_i8_i32(%lhs: tensor<128x256xi8>, %rhs: tensor<256x128xi8>) -> tensor<128x128xi32> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi8>, tensor<256x128xi8>) outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4, 4, 0]
// CHECK-LABEL: func.func @matmul_i8_i32
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 2: FP32 Matmul (f32 inputs -> f32 accumulation)
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_f32(%lhs: tensor<128x256xf32>, %rhs: tensor<256x128xf32>) -> tensor<128x128xf32> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x128xf32>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4, 4, 0]
// CHECK-LABEL: func.func @matmul_f32
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 3: FP8 Matmul (f8E4M3FN inputs -> f16 accumulation)
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_fp8(%lhs: tensor<128x256xf8E4M3FN>, %rhs: tensor<256x128xf8E4M3FN>) -> tensor<128x128xf16> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x128xf16>
  %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xf8E4M3FN>, tensor<256x128xf8E4M3FN>) outs(%fill : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %res : tensor<128x128xf16>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4, 4, 0]
// CHECK-LABEL: func.func @matmul_fp8
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 4: BF16 Matmul (bf16 inputs -> f32 accumulation)
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_bf16(%lhs: tensor<128x256xbf16>, %rhs: tensor<256x128xbf16>) -> tensor<128x128xf32> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xbf16>, tensor<256x128xbf16>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4, 4, 0]
// CHECK-LABEL: func.func @matmul_bf16
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 5: Convolution (NHWC)
// Note: Input is 34x34 to produce a 32x32 output (which is divisible by 4)
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @conv_spacemit(%input: tensor<1x34x34x8xi8>, %filter: tensor<3x3x8x16xi8>) -> tensor<1x32x32x16xi32> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<1x32x32x16xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<1x32x32x16xi32>) -> tensor<1x32x32x16xi32>
  
  %res = linalg.conv_2d_nhwc_hwcf 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%input, %filter : tensor<1x34x34x8xi8>, tensor<3x3x8x16xi8>) 
    outs(%fill : tensor<1x32x32x16xi32>) -> tensor<1x32x32x16xi32>
    
  return %res : tensor<1x32x32x16xi32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [1, 1, 4, 4, 0, 0, 0]
// CHECK-LABEL: func.func @conv_spacemit
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 6: Generic Reduction
// ============================================================================

#spacemit_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

#map_input = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_weight = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_output = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @generic_contraction(%lhs: tensor<16x64xi8>, %rhs: tensor<16x64xi8>) -> tensor<16x16xi32> attributes {hal.executable.target = #spacemit_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<16x16xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<16x16xi32>) -> tensor<16x16xi32>
  
  %res = linalg.generic {
    indexing_maps = [#map_input, #map_weight, #map_output], 
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<16x64xi8>, tensor<16x64xi8>) outs(%fill : tensor<16x16xi32>) {
  ^bb0(%in: i8, %in_0: i8, %out: i32):
    %ext_lhs = arith.extsi %in : i8 to i32
    %ext_rhs = arith.extsi %in_0 : i8 to i32
    %mul = arith.muli %ext_lhs, %ext_rhs : i32
    %add = arith.addi %out, %mul : i32
    linalg.yield %add : i32
  } -> tensor<16x16xi32>
  
  return %res : tensor<16x16xi32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4, 4, 0]
// CHECK-LABEL: func.func @generic_contraction
// CHECK:       linalg.generic
// CHECK-SAME:    lowering_config = #[[CONFIG]]

// -----

// ============================================================================
// Test 7: Fallback Regression (Standard RISC-V without +xsmtvdot)
// Logic: Ensure normal RISC-V uses vector-heavy lengths like [8, 32, 0].
// ============================================================================

#generic_riscv_target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {
  cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", 
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", 
  native_vector_size = 32 : index, 
  target_triple = "riscv64-unknown-unknown-elf"
}>

func.func @matmul_generic_riscv(%lhs: tensor<128x256xi8>, %rhs: tensor<256x128xi8>) -> tensor<128x128xi32> attributes {hal.executable.target = #generic_riscv_target} {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi8>, tensor<256x128xi8>) outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}

// CHECK-DAG:     #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [8, 32, 0]
// CHECK-LABEL: func.func @matmul_generic_riscv
// CHECK:       linalg.matmul
// CHECK-SAME:    lowering_config = #[[CONFIG]]