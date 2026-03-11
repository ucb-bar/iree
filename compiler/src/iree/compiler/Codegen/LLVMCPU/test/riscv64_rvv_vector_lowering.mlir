// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmcpu-mmt4d-vector-lowering, iree-codegen-llvmcpu-vector-lowering-pipeline))" --split-input-file | FileCheck %s
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmcpu-mmt4d-vector-lowering{vector-contract-custom-kernels=false}))" --split-input-file | FileCheck %s -check-prefix=CHECK-KERNEL-OFF
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-llvmcpu-mmt4d-vector-lowering{vector-contract-custom-kernels=true}))" --split-input-file | FileCheck %s -check-prefix=CHECK-KERNEL-ON

// -----

#target_rvv = #hal.executable.target<"xyz", "xyz", {
  target_triple = "riscv64-unknown-linux-gnu",
  cpu_features = "+v,+zvl256b"
}>
#translation_info = #iree_codegen.translation_info<pipeline = Mmt4dTilingExpert>

func.func @simple_i8_mmt4d_no_custom_kernel(
    %lhs: vector<1x1x8x1xi8>,
    %rhs: vector<1x1x16x1xi8>,
    %acc: vector<1x1x8x16xi32>) -> vector<1x1x8x16xi32>
    attributes {hal.executable.target = #target_rvv, translation_info = #translation_info} {
  %lhs_wide = arith.extsi %lhs : vector<1x1x8x1xi8> to vector<1x1x8x1xi32>
  %rhs_wide = arith.extsi %rhs : vector<1x1x16x1xi8> to vector<1x1x16x1xi32>
  %res = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
      ],
      iterator_types = [
        "parallel", "parallel", "reduction",
        "parallel", "parallel", "reduction"
      ],
      kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc :
      vector<1x1x8x1xi32>, vector<1x1x16x1xi32> into vector<1x1x8x16xi32>
  return %res : vector<1x1x8x16xi32>
}

// CHECK-LABEL: func.func @simple_i8_mmt4d_no_custom_kernel(
// CHECK: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwadd.w"

// CHECK-KERNEL-OFF-LABEL: func.func @simple_i8_mmt4d_no_custom_kernel(
// CHECK-KERNEL-OFF-NOT: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK-KERNEL-OFF-NOT: llvm.call_intrinsic "llvm.riscv.vwadd.w"

// CHECK-KERNEL-ON-LABEL: func.func @simple_i8_mmt4d_no_custom_kernel(
// CHECK-KERNEL-ON: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK-KERNEL-ON: llvm.call_intrinsic "llvm.riscv.vwadd.w"

// -----

#target_rvv = #hal.executable.target<"xyz", "xyz", {
  target_triple = "riscv64-unknown-linux-gnu",
  cpu_features = "+v,+zvl256b"
}>

func.func @mmt4d_kernel_dispatch(
    %lhs: memref<1x1x8x1xi8>,
    %rhs: memref<1x1x16x1xi8>,
    %dst: memref<1x1x8x16xi32>) attributes {hal.executable.target = #target_rvv} {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<1x1x8x16xi32>
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  vector.transfer_write %cst, %dst[%c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true]} :
      vector<1x1x8x16xi32>, memref<1x1x8x16xi32>

  %res = scf.for %k = %c0 to %c1 step %c1
      iter_args(%acc_iter = %cst) -> (vector<1x1x8x16xi32>) {
    %lhs_read = vector.transfer_read %lhs[%c0, %k, %c0, %c0], %c0_i8
        {in_bounds = [true, true, true, true]} :
        memref<1x1x8x1xi8>, vector<1x1x8x1xi8>
    %rhs_read = vector.transfer_read %rhs[%c0, %k, %c0, %c0], %c0_i8
        {in_bounds = [true, true, true, true]} :
        memref<1x1x16x1xi8>, vector<1x1x16x1xi8>

    %lhs_wide = arith.extsi %lhs_read : vector<1x1x8x1xi8> to vector<1x1x8x1xi32>
    %rhs_wide = arith.extsi %rhs_read : vector<1x1x16x1xi8> to vector<1x1x16x1xi32>

    %next = vector.contract {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
        ],
        iterator_types = [
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction"
        ],
        kind = #vector.kind<add>
    } %lhs_wide, %rhs_wide, %acc_iter :
        vector<1x1x8x1xi32>, vector<1x1x16x1xi32> into vector<1x1x8x16xi32>

    scf.yield %next : vector<1x1x8x16xi32>
  }

  vector.transfer_write %res, %dst[%c0, %c0, %c0, %c0]
      {in_bounds = [true, true, true, true]} :
      vector<1x1x8x16xi32>, memref<1x1x8x16xi32>
  return
}

// CHECK-LABEL: func.func @mmt4d_kernel_dispatch(
// CHECK-NOT: vector.contract
// CHECK: %[[RHSVEC:.+]] = vector.load %{{.*}} : memref<1x1x16xi8, strided<[16, 16, 1]>>, vector<16xi8>
// CHECK: %[[RVV_RHS:.+]] = llvm.call_intrinsic "llvm.vector.insert"(%{{.*}}, %[[RHSVEC]], %{{.*}}) : (vector<[4]xi8>, vector<16xi8>, i64) -> vector<[4]xi8>
// CHECK: %[[LHS0:.+]] = memref.load %{{.*}} : memref<1x1x8xi8, strided<[8, 8, 1]>>
// CHECK-COUNT-8: llvm.call_intrinsic "llvm.riscv.vwmul"
// CHECK: llvm.call_intrinsic "llvm.riscv.vwadd.w"
// CHECK: llvm.call_intrinsic "llvm.vector.extract"
// CHECK-COUNT-8: vector.store %{{.*}} : memref<8x16xi32>, vector<16xi32>