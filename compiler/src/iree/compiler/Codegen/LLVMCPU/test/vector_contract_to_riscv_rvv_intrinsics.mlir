// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-vector-contract-custom-kernels))" %s | FileCheck %s

// -----

func.func @mmt_8x16x1_i8i8i32_rvv(
    %lhs: vector<8x1xi8>,
    %rhs: vector<16x1xi8>,
    %acc: vector<8x16xi32>) -> vector<8x16xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {
    target_triple = "riscv64-unknown-linux-gnu",
    cpu_features = "+v,+zvl256b"
  }>
} {
  %lhs_wide = arith.extsi %lhs : vector<8x1xi8> to vector<8x1xi32>
  %rhs_wide = arith.extsi %rhs : vector<16x1xi8> to vector<16x1xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc :
      vector<8x1xi32>, vector<16x1xi32> into vector<8x16xi32>
  return %res : vector<8x16xi32>
}

// CHECK-LABEL: func.func @mmt_8x16x1_i8i8i32_rvv(
// CHECK-NOT: vector.contract
// CHECK-NOT: arith.extsi
// CHECK: vector.shape_cast
// CHECK: vector.extract_strided_slice
// CHECK: llvm.call_intrinsic "llvm.vector.insert"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwadd.w"
// CHECK: llvm.call_intrinsic "llvm.vector.extract"
// CHECK: vector.insert_strided_slice
// CHECK: vector.shape_cast
// CHECK: return

// -----

func.func @mmt_1x32x1_i8i8i32_rvv_matvec(
    %lhs: vector<1x1xi8>,
    %rhs: vector<32x1xi8>,
    %acc: vector<1x32xi32>) -> vector<1x32xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {
    target_triple = "riscv64-unknown-linux-gnu",
    cpu_features = "+v,+zvl256b"
  }>
} {
  %lhs_wide = arith.extsi %lhs : vector<1x1xi8> to vector<1x1xi32>
  %rhs_wide = arith.extsi %rhs : vector<32x1xi8> to vector<32x1xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc :
      vector<1x1xi32>, vector<32x1xi32> into vector<1x32xi32>
  return %res : vector<1x32xi32>
}

// CHECK-LABEL: func.func @mmt_1x32x1_i8i8i32_rvv_matvec(
// CHECK-NOT: vector.contract
// CHECK-NOT: arith.extsi
// CHECK: vector.shape_cast
// CHECK: llvm.call_intrinsic "llvm.vector.insert"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwadd.w"
// CHECK: llvm.call_intrinsic "llvm.vector.extract"
// CHECK: return

// -----

func.func @mmt_i32i32i32_no_rvv_custom_kernel(
    %lhs: vector<8x1xi32>,
    %rhs: vector<16x1xi32>,
    %acc: vector<8x16xi32>) -> vector<8x16xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {
    target_triple = "riscv64-unknown-linux-gnu",
    cpu_features = "+v,+zvl256b"
  }>
} {
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs, %rhs, %acc :
      vector<8x1xi32>, vector<16x1xi32> into vector<8x16xi32>
  return %res : vector<8x16xi32>
}

// CHECK-LABEL: func.func @mmt_i32i32i32_no_rvv_custom_kernel(
// CHECK: vector.contract
// CHECK: return